"""
TCG-Net Stage 1 — fMRI Connectome Encoder pretraining  (§7.1).

Architecture follows §3 exactly:
  §3.1  FC input: Fisher z-transformed Pearson FC matrix A_f ∈ R^{R×R}
  §3.2  Node features: FC profile (§3.2.1) + 6 graph-theoretic features (§3.2.2)
          → X_i ∈ R^{R×72}  (R=66, G=6)
  §3.3  Graph edges: group-level top-20% mask × individual z-FC weights
          stored as sparse edge_index + edge_attr  (no leakage — mask from train set only)
  §3.4  GNN encoder (torch_geometric):
          GCNConv(72→128) → GAT(128→128, heads=4, concat→512) → GCNConv(512→128)
          → mean/max readout → Linear(256→128) → z_f ∈ R^{128}
  §5    Aux head (Stage 1 only): Linear(128→64→32→1) → ŷ_f

Training per §7.1 single-modality pretraining:  A_f → z_f → ŷ_f
Target: UK Biobank Fluid Intelligence field 20016, instance 2 (imaging visit).
Target is z-scored during optimisation; all reported metrics are on original 0-13 scale.

Graph-theoretic features (§3.2.2) are expensive to compute and cached per CV fold
to Data/fmri_graph_features_fold{k}.npz. Delete fold caches if the source FC data changes.
"""

import json
from pathlib import Path

import networkx as nx
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FC_PATH      = '/home/celery/Project/TCG-Net/Data/corr_matrix.mat'
PHENO_CSV    = '/home/celery/uk_BIOBANK_DATA/table/phesantallphenos_04.csv'
CACHE_FEATS  = '/home/celery/Project/TCG-Net/Data/fmri_graph_features.npz'
OUT_WEIGHTS  = '/home/celery/Project/TCG-Net/Data/fmri_stage1_encoder.pt'
OUT_LOG      = '/home/celery/Project/TCG-Net/Data/fmri_stage1_log.json'
OUT_SPLITS   = '/home/celery/Project/TCG-Net/Data/fmri_stage1_splits.json'

# ---------------------------------------------------------------------------
# Cohort filters
# ---------------------------------------------------------------------------
BAD_FC_EIDS  = {1909084, 2364098, 5275169}  # ≥14/66 flat BOLD ROIs
FLUID_IQ_COL = '20016-2.0'                  # imaging visit (instance 2)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED       = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE   = 30
N_FOLDS    = 5           # §8: 5-fold CV stratified by fluid IQ quintile
LR         = 5e-4        # §7.2 Stage-1 (tuning: middle ground between 1e-3 and 3e-4)
WD         = 1e-3        # increased from 1e-4: stronger L2 to combat overfitting
DROPOUT    = 0.5         # increased from 0.2: FC→fluid IQ signal is weak, needs regularisation
N_ROI      = 66
N_GT_FEAT  = 6           # §3.2.2 graph-theoretic features per node
NODE_DIM   = N_ROI + N_GT_FEAT   # 72
EDGE_PCT   = 0.20        # §3.3 top-20% group-level mask
Z_DIM      = 128


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 1. Cohort loading
# ===========================================================================

def load_cohort():
    print('[cohort] loading FC...')
    mat    = sio.loadmat(FC_PATH)
    fc     = mat['FC_Matrices']
    fc_ids = np.array(mat['Patient_IDs']).squeeze().astype(np.int64)
    print(f'[cohort]   raw FC: {fc.shape}')

    pheno    = pd.read_csv(PHENO_CSV, usecols=['eid', FLUID_IQ_COL]).dropna(subset=[FLUID_IQ_COL])
    fluid_iq = dict(zip(pheno['eid'].astype(int), pheno[FLUID_IQ_COL].astype(np.float32)))
    print(f'[cohort]   fluid IQ subjects (instance 2): {len(fluid_iq)}')

    keep, ys = [], []
    for i, e in enumerate(fc_ids):
        e = int(e)
        if e in BAD_FC_EIDS or e not in fluid_iq:
            continue
        keep.append(i)
        ys.append(fluid_iq[e])

    keep = np.asarray(keep, dtype=np.int64)
    fc   = fc[keep].astype(np.float32)
    eids = fc_ids[keep]
    y    = np.asarray(ys, dtype=np.float32)
    print(f'[cohort]   final N={len(eids)}, FC {fc.shape}')
    return fc, eids, y


# ===========================================================================
# 2. Graph-theoretic node features  §3.2.2
# ===========================================================================

def _thresh_abs(fc_mat: np.ndarray, pct: float = 0.20) -> np.ndarray:
    """Per-subject top-pct% absolute FC threshold (single matrix)."""
    R    = fc_mat.shape[0]
    flat = np.abs(fc_mat[np.triu_indices(R, k=1)])
    thr  = np.quantile(flat, 1.0 - pct)
    return np.where(np.abs(fc_mat) >= thr, np.abs(fc_mat), 0.0)



def _floyd_warshall_batch(A: np.ndarray) -> np.ndarray:
    """
    All-pairs shortest paths on a batch of weighted graphs via Floyd-Warshall.
    A: [C, R, R] non-negative edge weights (0 = no edge).
    Returns dist: [C, R, R] shortest path lengths (inf where unreachable).
    """
    R    = A.shape[1]
    eps  = 1e-10
    dist = np.where(A > 0, 1.0 / A.clip(min=eps), np.inf).astype(np.float64)
    dist[:, np.arange(R), np.arange(R)] = 0.0
    for r in range(R):
        np.minimum(dist,
                   dist[:, :, r:r+1] + dist[:, r:r+1, :],
                   out=dist)
    return dist


def compute_graph_features(fc_matrices: np.ndarray, train_idx: np.ndarray,
                            group_mask: np.ndarray, cache_path: str) -> np.ndarray:
    """
    Compute §3.2.2 features for all subjects using batched numpy (no networkx
    in the per-subject loop — ~100× faster than per-subject nx calls).

    Features per node: strength, clustering (Onnela 2005), betweenness centrality
    (Floyd-Warshall), participation coefficient, local efficiency, within-module
    degree z-score.

    Cached to disk after the first run; delete the cache if FC data changes
    or if this function's algorithm changes.
    """
    if Path(cache_path).exists():
        cached = np.load(cache_path)['features']
        if cached.shape[0] == len(fc_matrices):
            print(f'[features] loading cached features from {cache_path}')
            return cached
        # Cohort size changed (e.g. different subject filters) — recompute
        print(f'[features] cache size mismatch '
              f'({cached.shape[0]} cached vs {len(fc_matrices)} current) — recomputing...')

    # --- Group-level community assignments (one networkx call, not per-subject) ---
    # Community labels are used for participation coefficient and within-module
    # degree z-score. Computed from the training-set mean FC only — no leakage.
    # Louvain (per §3 plan) is used rather than greedy modularity because it
    # typically finds higher-quality partitions. seed=SEED ensures reproducibility
    # since Louvain has a random initialisation step.
    print('[features] computing group-level modules from training set...')
    fc_train_mean = np.abs(fc_matrices[train_idx]).mean(axis=0)
    A_group       = _thresh_abs(fc_train_mean, EDGE_PCT)
    communities   = list(nx.community.louvain_communities(
                        nx.from_numpy_array(A_group), seed=SEED))
    module_labels = np.zeros(N_ROI, dtype=int)
    for m, comm in enumerate(communities):
        for node in comm:
            module_labels[node] = m
    print(f'[features]   {len(communities)} modules detected')

    # Neighbour lists derived from the group mask (fixed topology shared by all
    # subjects). Used by local efficiency to define the subgraph of each node's
    # neighbours — must be consistent with the group-masked A built in the loop.
    nbr_lists = [np.where(group_mask[i] & (np.arange(N_ROI) != i))[0]
                 for i in range(N_ROI)]

    # --- Batched numpy computation ---
    N        = len(fc_matrices)
    features = np.zeros((N, N_ROI, N_GT_FEAT), dtype=np.float32)
    CHUNK    = 1000
    eps      = 1e-12

    # Precomputed once: s≠t mask for betweenness validity check
    _s_ne_t = ~np.eye(N_ROI, dtype=bool)                        # [R, R]

    print(f'[features] computing for {N} subjects (batched numpy, no networkx)...')
    for start in tqdm(range(0, N, CHUNK), unit='chunk', dynamic_ncols=True):
        end = min(start + CHUNK, N)

        # Apply the group mask to individual absolute FC — all six graph metrics
        # are computed on this matrix so they share a consistent topology with
        # the GNN edges. Previously a per-subject 20% threshold was used here,
        # which was inconsistent with local efficiency's group-mask neighbour
        # lists and produced different graph topologies across the six features.
        A = (np.abs(fc_matrices[start:end]) *
             group_mask[np.newaxis, :, :]).astype(np.float32)   # [C, R, R]
        C = A.shape[0]

        # 1. Nodal strength — sum of absolute FC weights at each node
        strength = A.sum(axis=2)                                 # [C, R]

        # 2. Clustering coefficient (Onnela 2005 weighted)
        # Measures the density of triangles around each node, scaled by edge
        # weight. The cube-root normalisation keeps the range in [0, 1].
        Amax = A.max(axis=(1, 2), keepdims=True).clip(min=eps)
        W13  = np.cbrt(A / Amax)                                 # [C, R, R]
        tmp  = np.matmul(W13, W13)                               # [C, R, R]
        cyc3 = (tmp * W13.transpose(0, 2, 1)).sum(axis=2)        # [C, R]
        k    = (A > 0).sum(axis=2).astype(np.float64)           # degree
        den  = k * (k - 1)
        clust = np.divide(cyc3, den, out=np.zeros_like(cyc3), where=den > 0)

        # 3. Betweenness centrality (§3.2.2)
        # v lies on the shortest path s→t iff dist[s,v] + dist[v,t] ≈ dist[s,t].
        # For weighted graphs with continuous edge weights, path-length ties are
        # negligible, so this binary test counts σ(s,t|v)/σ(s,t) exactly.
        # Normalised for undirected graphs: divide by (R-1)(R-2) because each
        # unordered pair (s,t) is counted twice (as s→t and t→s).
        dist        = _floyd_warshall_batch(A)                   # [C, R, R]
        betweenness = np.zeros((C, N_ROI), dtype=np.float64)
        for v in range(N_ROI):
            # Path length through v for every (s, t) pair
            via_v   = dist[:, :, v:v+1] + dist[:, v:v+1, :]    # [C, R, R]
            # inf - inf = nan when both endpoints are unreachable; suppress the
            # warning — those pairs are already excluded by the valid mask below
            with np.errstate(invalid='ignore'):
                on_path = np.abs(via_v - dist) < 1e-9           # [C, R, R]
            # Exclude: v itself as source/target, self-pairs, unreachable pairs
            s_ne_v  = np.arange(N_ROI) != v                     # [R]
            valid   = (s_ne_v[np.newaxis, :, np.newaxis] &
                       s_ne_v[np.newaxis, np.newaxis, :] &
                       _s_ne_t[np.newaxis, :, :] &
                       ~np.isinf(dist))
            betweenness[:, v] = (on_path & valid).sum(axis=(1, 2))
        # Normalise: (R-1)(R-2) accounts for counting both (s→t) and (t→s)
        betweenness /= (N_ROI - 1) * (N_ROI - 2)

        # 4. Participation coefficient — how evenly a node's connections are
        # distributed across modules. 0 = all connections within one module,
        # 1 = connections spread uniformly across all modules.
        P = np.zeros((C, N_ROI), dtype=np.float64)
        for m in np.unique(module_labels):
            idx_m  = np.where(module_labels == m)[0]
            k_in   = A[:, :, idx_m].sum(axis=2)                 # strength to module m
            P     += (k_in / (strength + eps)) ** 2
        participation = 1.0 - P

        # 5. Local efficiency — harmonic mean of inverse shortest-path lengths
        # within each node's neighbourhood subgraph. Measures how well a node's
        # neighbours communicate when that node is removed.
        # nbr_lists uses group-mask topology → consistent with A above.
        local_eff = np.zeros((C, N_ROI), dtype=np.float64)
        for i in range(N_ROI):
            nbrs   = nbr_lists[i]
            n_nbrs = len(nbrs)
            if n_nbrs < 2:                                       # need ≥2 neighbours
                continue
            sub_A  = A[:, nbrs[:, None], nbrs[None, :]]         # [C, k, k]
            sub_d  = _floyd_warshall_batch(sub_A)
            inv_d  = np.divide(1.0, sub_d,
                               out=np.zeros_like(sub_d),
                               where=(sub_d > 0) & ~np.isinf(sub_d))
            local_eff[:, i] = inv_d.sum(axis=(1, 2)) / (n_nbrs * (n_nbrs - 1))

        # 6. Within-module degree z-score (Guimerà & Amaral 2005) — z-score of
        # each node's strength to nodes in its OWN module, measured against the
        # other nodes in that module. High values indicate intra-module hubs.
        # (Previously this z-scored total strength, which mislabeled the metric
        # and made it largely redundant with feature 1.)
        within_z = np.zeros((C, N_ROI), dtype=np.float64)
        for m in np.unique(module_labels):
            idx_m = module_labels == m
            if idx_m.sum() < 2:                                  # need ≥2 nodes in module
                continue
            k_in_m = A[:, :, idx_m].sum(axis=2)                 # [C, |m|] within-module strength
            std_m  = k_in_m.std(axis=1, keepdims=True)
            within_z[:, idx_m] = np.where(
                std_m > 0,
                (k_in_m - k_in_m.mean(axis=1, keepdims=True)) / (std_m + eps),
                0.0,
            )

        # Stack order must match §3.2.2 feature list exactly:
        # [strength, clustering, betweenness, participation, local_eff, within_z]
        features[start:end] = np.stack(
            [strength, clust, betweenness, participation, local_eff, within_z],
            axis=2,
        ).astype(np.float32)

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, module_labels=module_labels)
    print(f'[features] saved → {cache_path}')
    return features


# ===========================================================================
# 3. Group-level sparse mask  §3.3
# ===========================================================================

def build_group_mask(fc_matrices: np.ndarray, train_idx: np.ndarray,
                     pct: float = EDGE_PCT) -> np.ndarray:
    """
    M_group: symmetric boolean [R, R] of top-pct% edges from training-set mean FC.
    Only training subjects → no leakage.
    """
    mean_abs = np.abs(fc_matrices[train_idx]).mean(axis=0)
    R        = mean_abs.shape[0]
    thr      = np.quantile(mean_abs[np.triu_indices(R, k=1)], 1.0 - pct)
    mask     = mean_abs >= thr
    return mask | mask.T


# ===========================================================================
# 4. Dataset  — returns torch_geometric Data objects (sparse edge representation)
# ===========================================================================

class ConnectomeDataset:
    """
    Wraps the cohort as a PyG-compatible dataset (returns Data objects).

    Per sample:
      x          : [R, R+G]  standardised node features — Fisher-z FC profile
                              (§3.2.1) concatenated with graph-theoretic features
                              (§3.2.2). R=66, G=6 → shape [66, 72].
      edge_index : [2, E]    shared group-mask topology (both directions, no
                              self-loops). Same for all subjects — topology is
                              fixed by the group-level mask.
      edge_attr  : [E, 1]    raw Fisher-z FC values at the group-mask positions.
                              NOT per-edge z-scored — see __getitem__ for why.
      y          : scalar    fluid IQ z-scored during training; metrics reported
                              on the original 0-13 scale after de-normalisation.

    Call apply_normalization(train_idx) before using the dataset — it computes
    all normalisation statistics from training subjects only (no leakage).
    """

    def __init__(self, fc_matrices: np.ndarray, graph_feats: np.ndarray,
                 y: np.ndarray, group_mask: np.ndarray):
        # Clip before arctanh to avoid inf at ±1.0 (diagonal is already 0.0
        # after FC_corr.py fix, but off-diagonal values can still be ±1.0 for
        # perfectly correlated/anti-correlated time series).
        self.fc_z        = np.arctanh(np.clip(fc_matrices, -0.999999, 0.999999)).astype(np.float32)
        self.graph_feats = graph_feats.astype(np.float32)   # [N, R, 6]
        self.y_raw       = y.astype(np.float32)
        self.y           = y.astype(np.float32)

        # Pre-compute sparse edge indices from the shared group mask.
        # Self-loop guard (rows != cols) is a safety net — after FC_corr.py fix
        # the diagonal is 0.0 so it never enters the group mask, but we keep
        # the guard in case the mask source changes in the future.
        rows, cols   = np.where(group_mask)
        no_self      = rows != cols
        self.rows    = rows[no_self]
        self.cols    = cols[no_self]
        edge_index   = np.stack([self.rows, self.cols], axis=0)          # [2, E]
        self.edge_index = torch.from_numpy(edge_index).long()            # shared tensor

        # Normalisation stats — set by apply_normalization(); None until then.
        self.y_mean   = 0.0
        self.y_std    = 1.0
        self.fc_mu    = None    # [R, R]  per-edge mean of Fisher-z FC (train set)
        self.fc_sigma = None    # [R, R]  per-edge std  of Fisher-z FC (train set)
        self.feat_mu  = None    # [G]     per-feature mean of graph metrics (train set)
        self.feat_sigma = None  # [G]     per-feature std  of graph metrics (train set)

    def apply_normalization(self, train_idx: np.ndarray) -> None:
        y_tr          = self.y_raw[train_idx]
        self.y_mean   = float(y_tr.mean())
        self.y_std    = float(max(y_tr.std(), 1e-4))
        self.y        = ((self.y_raw - self.y_mean) / self.y_std).astype(np.float32)

        fc_tr         = self.fc_z[train_idx]
        self.fc_mu    = fc_tr.mean(axis=0).astype(np.float32)
        self.fc_sigma = np.maximum(fc_tr.std(axis=0), 1e-4).astype(np.float32)

        gf_tr         = self.graph_feats[train_idx].reshape(-1, N_GT_FEAT)
        self.feat_mu  = gf_tr.mean(axis=0).astype(np.float32)
        self.feat_sigma = np.maximum(gf_tr.std(axis=0), 1e-4).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Data:
        assert self.fc_mu is not None, (
            "Call apply_normalization(train_idx) before using the dataset."
        )

        # FC profile node features — standardised per edge position using
        # training-set mean/std, then clipped to ±5.
        # The ±5 clip prevents blow-up at near-constant edges where fc_sigma
        # collapses to the 1e-4 floor: without clipping, value ≈ fc_z / 1e-4
        # which can reach ~72 000 and destabilise GCN aggregation.
        fc_std = np.clip((self.fc_z[idx] - self.fc_mu) / self.fc_sigma,
                         -5.0, 5.0).astype(np.float32)                   # [R, R]

        # Graph-theoretic features — standardised per feature type (pooled
        # across all nodes and subjects in the training set, shape [G=6]).
        gf_std = np.clip((self.graph_feats[idx] - self.feat_mu) / self.feat_sigma,
                         -5.0, 5.0)                                                # [R, 6]

        # Concatenate to final node feature matrix X_i ∈ R^{R×(R+G)} (§3.2)
        node_feats = torch.from_numpy(
            np.concatenate([fc_std, gf_std], axis=1).astype(np.float32)          # [R, 72]
        )

        # Edge attributes: raw Fisher-z FC values at group-mask positions.
        # NOT per-edge z-scored — per-edge standardisation with a 1e-4 sigma
        # floor inflates near-zero edges to ~72 000, crashing GCN aggregation
        # before BatchNorm can rescue them. Raw z-FC is bounded by ≈±7.25.
        edge_attr = torch.from_numpy(
            self.fc_z[idx][self.rows, self.cols].astype(np.float32)
        ).unsqueeze(-1)                                                            # [E, 1]

        return Data(
            x          = node_feats,
            edge_index = self.edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor(self.y[idx]),
        )


# ===========================================================================
# 5. Model  —  torch_geometric GCNConv + GATConv
# ===========================================================================

class ConnectomeEncoder(nn.Module):
    """
    §3.4 GCN-GAT connectome encoder (torch_geometric implementation).

    GCNConv(72→128, weighted)  → BN → GELU → Dropout(DROPOUT)
    GATConv(128→128, heads=4, edge_dim=1, concat→512) → BN → GELU → Dropout(DROPOUT)
    GCNConv(512→128, weighted) → GELU
    global_mean_pool ∥ global_max_pool → [B, 256]
    Linear(256→128) → z_f ∈ R^{128}

    After each forward pass, self.last_attn stores the GAT attention coefficients
    as (edge_index, alpha) where alpha ∈ R^{E×heads}. Used for §11 interpretability
    (brain-region and edge importance scores). Value is overwritten each batch —
    call in eval mode on a single batch to inspect specific subjects.
    """

    def __init__(self, node_feat_dim: int = NODE_DIM, dropout: float = DROPOUT):
        super().__init__()

        # Layer 1: GCN — aggregates weighted neighbourhood information
        self.gcn1 = GCNConv(node_feat_dim, 128, add_self_loops=True, normalize=True)
        self.bn1  = nn.BatchNorm1d(128)

        # Layer 2: GAT — learns which edges to attend to, conditioned on FC weight
        # edge_dim=1 incorporates the Fisher-z FC value into attention (§3.3)
        # fill_value=0.0 assigns zero edge attribute to added self-loops, consistent
        # with the zeroed diagonal in the FC matrices
        self.gat  = GATConv(128, 128, heads=4, concat=True, dropout=dropout,
                            edge_dim=1, add_self_loops=True, fill_value=0.0)
        self.bn2  = nn.BatchNorm1d(512)

        # Layer 3: GCN — compresses the 4-head concatenation back to 128
        self.gcn2 = GCNConv(512, 128, add_self_loops=True, normalize=True)

        self.proj = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout)

        # Cached GAT attention weights from last forward pass (§11 interpretability)
        self.last_attn: tuple | None = None  # (edge_index [2,E], alpha [E, heads])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # GCNConv degree normalisation requires non-negative edge weights.
        # Taking abs() preserves magnitude while satisfying the constraint.
        # Sign information is retained in the node FC-profile features and in
        # the GAT edge_attr, so no information is lost overall.
        ew = edge_attr.abs().squeeze(-1)                         # [E]

        # --- Layer 1: GCN ---
        h = self.gcn1(x, edge_index, edge_weight=ew)            # [N, 128]
        h = F.gelu(self.bn1(h))
        h = self.drop(h)

        # --- Layer 2: GAT ---
        # Clamp edge_attr to [-3, 3] before passing to GAT to prevent attention
        # logit explosion. Raw Fisher-z values reach ±7.25; without clamping the
        # edge term can dominate the attention score and collapse it to near
        # one-hot, destroying the learned attention distribution.
        gat_edge = edge_attr.clamp(-3.0, 3.0)                   # [E, 1]
        h, (attn_ei, attn_w) = self.gat(
            h, edge_index, edge_attr=gat_edge,
            return_attention_weights=True,
        )                                                        # h: [N, 512]
        # Detach before caching — we don't want this to keep the compute graph alive
        self.last_attn = (attn_ei.detach(), attn_w.detach())    # [2,E], [E, heads]
        h = F.gelu(self.bn2(h))
        h = self.drop(h)

        # --- Layer 3: GCN (no BN per §3.4 spec) ---
        h = F.gelu(self.gcn2(h, edge_index, edge_weight=ew))    # [N, 128]

        # --- Global readout ---
        # Concatenate mean and max pooling to capture both average and peak activity
        z = torch.cat([global_mean_pool(h, batch),
                        global_max_pool(h, batch)], dim=1)      # [B, 256]
        return self.proj(z)                                      # [B, 128] = z_f


class AuxHead(nn.Module):
    """
    §5 prediction head: z_f → ŷ_f (scalar fluid intelligence).

    Linear(128→64) → GELU → Dropout(DROPOUT) → Linear(64→32) → GELU → Linear(32→1)

    Stage 1 only — saved for evaluation but NOT loaded into Stage 2.
    Only the encoder weights (ConnectomeEncoder) carry over.
    """

    def __init__(self, z_dim: int = Z_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.GELU(), nn.Dropout(dropout),  # 128 → 64
            nn.Linear(64, 32),    nn.GELU(),                        # 64  → 32
            nn.Linear(32, 1),                                       # 32  → 1 (scalar)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)   # [B, 1] → [B]


# ===========================================================================
# 6. Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(encoder, head, loader, device, loss_fn,
             y_mean: float, y_std: float) -> dict:
    """Loss in z-scored space; MAE/MSE/R²/Pearson reported on original 0-13 scale."""
    encoder.eval(); head.eval()
    preds_z, ys_z, losses = [], [], []
    for batch in loader:
        batch = batch.to(device)
        z    = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        yhat = head(z)
        losses.append(loss_fn(yhat, batch.y).item())
        preds_z.append(yhat.cpu())
        ys_z.append(batch.y.cpu())

    preds_z = torch.cat(preds_z).numpy()
    ys_z    = torch.cat(ys_z).numpy()
    preds   = preds_z * y_std + y_mean
    ys      = ys_z    * y_std + y_mean

    mae    = float(np.mean(np.abs(preds - ys)))
    mse    = float(np.mean((preds - ys) ** 2))
    ss_res = float(np.sum((preds - ys) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    r      = float(np.corrcoef(preds, ys)[0, 1]) if len(preds) > 1 else float('nan')
    return {'loss': float(np.mean(losses)), 'mae': mae, 'mse': mse, 'r2': r2, 'pearson': r}


# ===========================================================================
# 7. Main
# ===========================================================================

def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    fc, eids, y = load_cohort()

    # §8: stratify folds by fluid IQ quintile to keep target distribution balanced
    # duplicates='drop' handles tied bin edges in integer-valued fluid IQ scores
    quintile_labels = pd.qcut(y, q=5, labels=False, duplicates='drop').astype(int)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics    = []   # test metrics for each fold
    fold_enc_states = []   # best encoder weights for each fold
    all_splits      = []   # per-fold eid lists + normalization stats (for Stage 2)

    for fold, (train_val_idx, test_idx) in enumerate(
            skf.split(np.arange(len(eids)), quintile_labels)):

        print(f'\n{"="*60}')
        print(f'FOLD {fold+1}/{N_FOLDS}  '
              f'(train_val={len(train_val_idx)}, test={len(test_idx)})')
        print(f'{"="*60}')

        # Hold out ~10% of total data from train_val as the val set for early
        # stopping. test_size=0.125 of 80% = 10% of the full cohort, giving
        # approximately a 70/10/20 split per fold.
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.125, random_state=SEED,
            stratify=quintile_labels[train_val_idx],
        )
        print(f'[split] train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')

        # Group mask — recomputed from this fold's train subjects only.
        # Known pitfall §3: using test/val subjects here is the single most
        # leakage-prone step; test_idx must never touch this computation.
        group_mask = build_group_mask(fc, train_idx, EDGE_PCT)
        n_edges    = int(group_mask[np.triu_indices(N_ROI, k=1)].sum())
        print(f'[mask]  {n_edges} edges ({100*n_edges/2145:.1f}% of upper triangle)')

        # Graph-theoretic features — cached per fold because the group mask
        # (and therefore the feature topology) differs between folds.
        fold_cache  = CACHE_FEATS.replace('.npz', f'_fold{fold}.npz')
        graph_feats = compute_graph_features(fc, train_idx, group_mask, fold_cache)

        ds = ConnectomeDataset(fc, graph_feats, y, group_mask)
        ds.apply_normalization(train_idx)

        loader_kw    = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        train_loader = DataLoader(Subset(ds, train_idx), shuffle=True,  **loader_kw)
        val_loader   = DataLoader(Subset(ds, val_idx),   shuffle=False, **loader_kw)
        test_loader  = DataLoader(Subset(ds, test_idx),  shuffle=False, **loader_kw)

        encoder = ConnectomeEncoder(node_feat_dim=NODE_DIM, dropout=DROPOUT).to(device)
        head    = AuxHead(z_dim=Z_DIM, dropout=DROPOUT).to(device)
        if fold == 0:
            n_params = (sum(p.numel() for p in encoder.parameters()) +
                        sum(p.numel() for p in head.parameters()))
            print(f'[model] {n_params:,} trainable parameters')

        opt     = AdamW(list(encoder.parameters()) + list(head.parameters()),
                        lr=LR, weight_decay=WD)
        sched   = CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)
        loss_fn = nn.HuberLoss(delta=1.0)

        best_val_mae = float('inf')
        best_state   = None
        bad_epochs   = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            encoder.train(); head.train()
            t_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                yhat  = head(encoder(batch.x, batch.edge_index,
                                     batch.edge_attr, batch.batch))
                loss  = loss_fn(yhat, batch.y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()), 1.0)
                opt.step()
                t_losses.append(loss.item())
            sched.step()

            val_m = evaluate(encoder, head, val_loader, device,
                             loss_fn, ds.y_mean, ds.y_std)
            print(f'  [ep {epoch:3d}] train={np.mean(t_losses):.4f}  '
                  f'val_loss={val_m["loss"]:.4f}  MAE={val_m["mae"]:.4f}  '
                  f'r={val_m["pearson"]:.4f}  lr={opt.param_groups[0]["lr"]:.2e}')

            # §7.1: early stopping on val MAE (original 0-13 scale)
            if val_m['mae'] < best_val_mae - 1e-5:
                best_val_mae = val_m['mae']
                best_state   = {
                    'encoder': {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                    'head':    {k: v.cpu().clone() for k, v in head.state_dict().items()},
                    'epoch':   epoch,
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    print(f'  [early stop] patience {PATIENCE} exhausted at epoch {epoch}')
                    break

        if best_state:
            encoder.load_state_dict(best_state['encoder'])
            head.load_state_dict(best_state['head'])
            print(f'  [best] epoch {best_state["epoch"]}, val_mae={best_val_mae:.4f}')

        test_m = evaluate(encoder, head, test_loader, device,
                          loss_fn, ds.y_mean, ds.y_std)
        print(f'  [test] MAE={test_m["mae"]:.4f}  R²={test_m["r2"]:.4f}  '
              f'r={test_m["pearson"]:.4f}')
        fold_metrics.append(test_m)

        enc_state = (best_state['encoder'] if best_state else
                     {k: v.cpu() for k, v in encoder.state_dict().items()})
        fold_enc_states.append(enc_state)

        fold_ckpt = OUT_WEIGHTS.replace('.pt', f'_fold{fold}.pt')
        Path(fold_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(enc_state, fold_ckpt)
        print(f'  [save] encoder → {fold_ckpt}')

        all_splits.append({
            'fold':       fold,
            'train_eids': eids[train_idx].tolist(),
            'val_eids':   eids[val_idx].tolist(),
            'test_eids':  eids[test_idx].tolist(),
            'y_mean':     ds.y_mean,
            'y_std':      ds.y_std,
            'best_epoch': best_state['epoch'] if best_state else None,
            'best_val_mae': float(best_val_mae),
            'test':       test_m,
        })

    # --- Aggregate CV results (§8 primary/secondary metrics) ---
    maes = [m['mae']     for m in fold_metrics]
    rs   = [m['pearson'] for m in fold_metrics]
    r2s  = [m['r2']      for m in fold_metrics]

    print(f'\n{"="*60}')
    print(f'CV RESULTS  ({N_FOLDS}-fold, stratified by fluid IQ quintile)')
    print(f'{"="*60}')
    print(f'  MAE     : {np.mean(maes):.4f} ± {np.std(maes):.4f}  (primary)')
    print(f'  Pearson : {np.mean(rs):.4f} ± {np.std(rs):.4f}  (secondary)')
    print(f'  R²      : {np.mean(r2s):.4f} ± {np.std(r2s):.4f}')

    # Save the best fold's encoder as the primary checkpoint for Stage 2
    best_fold = int(np.argmin(maes))
    print(f'\n[best fold] {best_fold+1} (MAE={maes[best_fold]:.4f})')
    torch.save(fold_enc_states[best_fold], OUT_WEIGHTS)
    print(f'[save] best encoder → {OUT_WEIGHTS}')

    summary = {
        'n_folds':    N_FOLDS,
        'folds':      all_splits,
        'cv_mae':     {'mean': float(np.mean(maes)), 'std': float(np.std(maes))},
        'cv_pearson': {'mean': float(np.mean(rs)),   'std': float(np.std(rs))},
        'cv_r2':      {'mean': float(np.mean(r2s)),  'std': float(np.std(r2s))},
        'best_fold':  best_fold,
    }
    with open(OUT_LOG, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[save] log    → {OUT_LOG}')

    with open(OUT_SPLITS, 'w') as f:
        json.dump(all_splits, f)
    print(f'[save] splits → {OUT_SPLITS}')


if __name__ == '__main__':
    main()
