"""
TCG-Net Stage 1 — fMRI Connectome Encoder pretraining, Alternative 2 graph.

Alternative 2 (§3, Alt 2): Positive/Negative Dual-Channel Graph
  A_i^+ = max(A_i, 0)    — excitatory FC (positive Fisher-z values)
  A_i^- = |min(A_i, 0)|  — inhibitory FC (absolute of negative Fisher-z values)

Two separate GCN-GAT encoders (SingleChannelEncoder) run in parallel on the positive
and negative FC channels respectively. Their [B, 128] outputs are concatenated to
[B, 256] and projected to z_f ∈ R^{B×128} for Stage 2 compatibility.

The sign split happens on the fly inside DualChannelEncoder.forward: the dataset stores
the full signed FC as edge_attr (identical to Alt 1 format), avoiding PyG batching issues
with heterogeneous per-sample edge structures.

Graph-theoretic node features are identical to Alt 1 (full dense absolute FC graph).
If Alt 1 fold caches already exist (Data/fmri_graph_features_alt1_fold{k}.npz) they are
reused automatically to skip redundant computation. Otherwise features are computed fresh
to Data/fmri_graph_features_alt2_fold{k}.npz.

All other choices (cohort filter, CV, training hyperparameters) match train_fmri_stage1.py.
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
FC_PATH           = '/home/celery/Project/TCG-Net/Data/corr_matrix.mat'
PHENO_CSV         = '/home/celery/uk_BIOBANK_DATA/table/phesantallphenos_04.csv'
CACHE_FEATS       = '/home/celery/Project/TCG-Net/Data/fmri_graph_features_alt2.npz'
CACHE_FEATS_ALT1  = '/home/celery/Project/TCG-Net/Data/fmri_graph_features_alt1.npz'
OUT_WEIGHTS       = '/home/celery/Project/TCG-Net/Data/fmri_stage1_encoder_alt2.pt'
OUT_LOG           = '/home/celery/Project/TCG-Net/Data/fmri_stage1_log_alt2.json'
OUT_SPLITS        = '/home/celery/Project/TCG-Net/Data/fmri_stage1_splits_alt2.json'

# ---------------------------------------------------------------------------
# Cohort filters
# ---------------------------------------------------------------------------
BAD_FC_EIDS  = {1909084, 2364098, 5275169}   # ≥14/66 flat BOLD ROIs
FLUID_IQ_COL = '20016-2.0'                   # imaging visit (instance 2)

# ---------------------------------------------------------------------------
# Hyperparameters  (identical to main + alt1)
# ---------------------------------------------------------------------------
SEED       = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE   = 30
N_FOLDS    = 5
LR         = 5e-4
WD         = 1e-3
DROPOUT    = 0.5
N_ROI      = 66
N_GT_FEAT  = 6
NODE_DIM   = N_ROI + N_GT_FEAT    # 72
EDGE_PCT   = 0.20    # for Louvain community detection only — not the graph topology
Z_DIM      = 128


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 1. Cohort loading  (identical to alt1)
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
# 2. Graph-theoretic node features  (identical to alt1 — full dense graph)
# ===========================================================================

def _thresh_abs(fc_mat: np.ndarray, pct: float = 0.20) -> np.ndarray:
    """Top-pct% absolute FC threshold — used only for Louvain community detection."""
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


def _fold_cache_path(fold: int) -> str:
    """Return the feature cache path for this fold.

    Alt 2 features are identical to Alt 1 (both use the full dense absolute FC
    graph). If an Alt 1 fold cache exists, reuse it to avoid redundant computation.
    """
    alt1_path = CACHE_FEATS_ALT1.replace('.npz', f'_fold{fold}.npz')
    if Path(alt1_path).exists():
        return alt1_path
    return CACHE_FEATS.replace('.npz', f'_fold{fold}.npz')


def compute_graph_features(fc_matrices: np.ndarray, train_idx: np.ndarray,
                            cache_path: str) -> np.ndarray:
    """
    Compute §3.2.2 features for all subjects on the full dense absolute FC graph.

    Identical computation to alt1 — six nodal metrics (strength, clustering,
    betweenness, participation, local efficiency, within-module degree z-score)
    computed on the absolute FC matrix without any thresholding.

    Community partition for participation coefficient and within-module degree is
    derived from the thresholded training-set mean FC (EDGE_PCT) so Louvain finds
    meaningful modules rather than a single fully-connected partition.
    """
    if Path(cache_path).exists():
        cached = np.load(cache_path)['features']
        if cached.shape[0] == len(fc_matrices):
            print(f'[features] loading cached features from {cache_path}')
            return cached
        print(f'[features] cache size mismatch '
              f'({cached.shape[0]} cached vs {len(fc_matrices)} current) — recomputing...')

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

    # All nodes are neighbours of each other in the full dense graph.
    nbr_lists = [np.array([j for j in range(N_ROI) if j != i]) for i in range(N_ROI)]

    N        = len(fc_matrices)
    features = np.zeros((N, N_ROI, N_GT_FEAT), dtype=np.float32)
    CHUNK    = 1000
    eps      = 1e-12
    _s_ne_t  = ~np.eye(N_ROI, dtype=bool)

    print(f'[features] computing for {N} subjects (Alt 2 — full dense graph, same as Alt 1)...')
    for start in tqdm(range(0, N, CHUNK), unit='chunk', dynamic_ncols=True):
        end = min(start + CHUNK, N)

        A = np.abs(fc_matrices[start:end]).astype(np.float32)   # [C, R, R]
        C = A.shape[0]

        # 1. Nodal strength
        strength = A.sum(axis=2)

        # 2. Clustering coefficient (Onnela 2005 weighted)
        Amax = A.max(axis=(1, 2), keepdims=True).clip(min=eps)
        W13  = np.cbrt(A / Amax)
        tmp  = np.matmul(W13, W13)
        cyc3 = (tmp * W13.transpose(0, 2, 1)).sum(axis=2)
        k    = (A > 0).sum(axis=2).astype(np.float64)
        den  = k * (k - 1)
        clust = np.divide(cyc3, den, out=np.zeros_like(cyc3), where=den > 0)

        # 3. Betweenness centrality (Floyd-Warshall)
        dist        = _floyd_warshall_batch(A)
        betweenness = np.zeros((C, N_ROI), dtype=np.float64)
        for v in range(N_ROI):
            via_v = dist[:, :, v:v+1] + dist[:, v:v+1, :]
            with np.errstate(invalid='ignore'):
                on_path = np.abs(via_v - dist) < 1e-9
            s_ne_v = np.arange(N_ROI) != v
            valid  = (s_ne_v[np.newaxis, :, np.newaxis] &
                      s_ne_v[np.newaxis, np.newaxis, :] &
                      _s_ne_t[np.newaxis, :, :] &
                      ~np.isinf(dist))
            betweenness[:, v] = (on_path & valid).sum(axis=(1, 2))
        betweenness /= (N_ROI - 1) * (N_ROI - 2)

        # 4. Participation coefficient
        P = np.zeros((C, N_ROI), dtype=np.float64)
        for m in np.unique(module_labels):
            idx_m = np.where(module_labels == m)[0]
            k_in  = A[:, :, idx_m].sum(axis=2)
            P    += (k_in / (strength + eps)) ** 2
        participation = 1.0 - P

        # 5. Local efficiency (65 neighbours per node in the complete graph)
        local_eff = np.zeros((C, N_ROI), dtype=np.float64)
        for i in range(N_ROI):
            nbrs   = nbr_lists[i]
            n_nbrs = len(nbrs)
            sub_A  = A[:, nbrs[:, None], nbrs[None, :]]
            sub_d  = _floyd_warshall_batch(sub_A)
            inv_d  = np.divide(1.0, sub_d,
                               out=np.zeros_like(sub_d),
                               where=(sub_d > 0) & ~np.isinf(sub_d))
            local_eff[:, i] = inv_d.sum(axis=(1, 2)) / (n_nbrs * (n_nbrs - 1))

        # 6. Within-module degree z-score (Guimerà & Amaral 2005) — z-score of
        # each node's within-module strength against its module peers.
        within_z = np.zeros((C, N_ROI), dtype=np.float64)
        for m in np.unique(module_labels):
            idx_m = module_labels == m
            if idx_m.sum() < 2:
                continue
            k_in_m = A[:, :, idx_m].sum(axis=2)[:, idx_m]       # [C, |m|] within-module strength
            std_m  = k_in_m.std(axis=1, keepdims=True)
            within_z[:, idx_m] = np.where(
                std_m > 0,
                (k_in_m - k_in_m.mean(axis=1, keepdims=True)) / (std_m + eps),
                0.0,
            )

        features[start:end] = np.stack(
            [strength, clust, betweenness, participation, local_eff, within_z],
            axis=2,
        ).astype(np.float32)

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, module_labels=module_labels)
    print(f'[features] saved → {cache_path}')
    return features


# ===========================================================================
# 3. Dataset  — full dense graph, signed FC as edge_attr (identical to Alt 1)
# ===========================================================================

class ConnectomeDataset:
    """
    Alt 2 dataset: same as Alt 1 — edge_index covers all i≠j pairs, edge_attr
    stores signed Fisher-z FC values. DualChannelEncoder splits them by sign
    during the forward pass, which avoids per-sample heterogeneous edge structures
    that would complicate PyG batching.
    """

    def __init__(self, fc_matrices: np.ndarray, graph_feats: np.ndarray,
                 y: np.ndarray):
        self.fc_z        = np.arctanh(np.clip(fc_matrices, -0.999999, 0.999999)).astype(np.float32)
        self.graph_feats = graph_feats.astype(np.float32)
        self.y_raw       = y.astype(np.float32)
        self.y           = y.astype(np.float32)

        # Complete directed graph: all (i, j) pairs where i ≠ j
        rows, cols      = np.where(~np.eye(N_ROI, dtype=bool))
        self.rows       = rows
        self.cols       = cols
        self.edge_index = torch.from_numpy(
            np.stack([rows, cols], axis=0)
        ).long()                                                  # [2, N_ROI*(N_ROI-1)]

        self.y_mean     = 0.0
        self.y_std      = 1.0
        self.fc_mu      = None
        self.fc_sigma   = None
        self.feat_mu    = None
        self.feat_sigma = None

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

        fc_std = np.clip((self.fc_z[idx] - self.fc_mu) / self.fc_sigma,
                         -5.0, 5.0).astype(np.float32)           # [R, R]

        gf_std = np.clip((self.graph_feats[idx] - self.feat_mu) / self.feat_sigma,
                         -5.0, 5.0)                              # [R, 6]

        node_feats = torch.from_numpy(
            np.concatenate([fc_std, gf_std], axis=1).astype(np.float32)   # [R, 72]
        )

        # Signed Fisher-z FC values at all i≠j positions.
        # The sign is preserved so DualChannelEncoder can split by sign on the fly.
        edge_attr = torch.from_numpy(
            self.fc_z[idx][self.rows, self.cols].astype(np.float32)
        ).unsqueeze(-1)                                           # [R*(R-1), 1]

        return Data(
            x          = node_feats,
            edge_index = self.edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor(self.y[idx]),
        )


# ===========================================================================
# 4. Model  — Dual-Channel GCN-GAT Encoder  (§3 Alt 2)
# ===========================================================================

class SingleChannelEncoder(nn.Module):
    """
    One branch of the dual-channel encoder.

    Identical to ConnectomeEncoder in the main and alt1 scripts:
      GCNConv(72→128)  → BN → GELU → Dropout
      GATConv(128→128, heads=4, concat→512) → BN → GELU → Dropout
      GCNConv(512→128) → GELU
      global_mean_pool ∥ global_max_pool → [B, 256]
      Linear(256→128)  → [B, 128]
    """

    def __init__(self, node_feat_dim: int = NODE_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.gcn1 = GCNConv(node_feat_dim, 128, add_self_loops=True, normalize=True)
        self.bn1  = nn.BatchNorm1d(128)
        self.gat  = GATConv(128, 128, heads=4, concat=True, dropout=dropout,
                            edge_dim=1, add_self_loops=True, fill_value=0.0)
        self.bn2  = nn.BatchNorm1d(512)
        self.gcn2 = GCNConv(512, 128, add_self_loops=True, normalize=True)
        self.proj = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout)

        # Cached GAT attention from last forward pass (for interpretability)
        self.last_attn: tuple | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        ew = edge_attr.abs().squeeze(-1)                          # [E] non-negative weights

        h = self.gcn1(x, edge_index, edge_weight=ew)
        h = F.gelu(self.bn1(h))
        h = self.drop(h)

        gat_edge = edge_attr.clamp(-3.0, 3.0)                    # [E, 1] clamped for stability
        h, (attn_ei, attn_w) = self.gat(
            h, edge_index, edge_attr=gat_edge,
            return_attention_weights=True,
        )
        self.last_attn = (attn_ei.detach(), attn_w.detach())
        h = F.gelu(self.bn2(h))
        h = self.drop(h)

        h = F.gelu(self.gcn2(h, edge_index, edge_weight=ew))

        z = torch.cat([global_mean_pool(h, batch),
                        global_max_pool(h, batch)], dim=1)        # [B, 256]
        return self.proj(z)                                        # [B, 128]


class DualChannelEncoder(nn.Module):
    """
    §3 Alt 2 positive/negative dual-channel encoder.

    Takes the same input signature as SingleChannelEncoder (shared node features +
    full signed FC as edge_attr). Splits edge_attr by sign in the forward pass:
      pos channel: FC > 0 (excitatory, weights preserved)
      neg channel: FC < 0 (inhibitory, magnitudes passed as positive weights)

    Both channels share identical node feature matrix x (FC profile + graph metrics).
    The sign information already encoded in x (the FC profile row) complements the
    topological split across the two edge channels.

    Outputs: z_pos ∈ R^{B×128} and z_neg ∈ R^{B×128} concatenated to [B, 256],
    then projected to z_f ∈ R^{B×128} for Stage 2 compatibility.

    Edge safety: add_self_loops=True in each SingleChannelEncoder ensures that
    even if a subject has an empty channel (all-positive or all-negative FC),
    self-loops keep the graph connected and the GNN computes valid outputs.
    """

    def __init__(self, node_feat_dim: int = NODE_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.pos_encoder = SingleChannelEncoder(node_feat_dim, dropout)
        self.neg_encoder = SingleChannelEncoder(node_feat_dim, dropout)
        # Fuse the two 128-dim embeddings back to 128 for Stage 2 API compatibility
        self.fusion_proj = nn.Linear(256, 128)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        weights  = edge_attr.squeeze(-1)                          # [E] signed FC values

        # Positive channel (excitatory FC)
        pos_mask = weights > 0
        pos_ei   = edge_index[:, pos_mask]
        pos_ea   = edge_attr[pos_mask]                            # [E_pos, 1] positive values

        # Negative channel (inhibitory FC — flip sign so GCN receives non-neg weights)
        neg_mask = weights < 0
        neg_ei   = edge_index[:, neg_mask]
        neg_ea   = edge_attr[neg_mask].abs()                      # [E_neg, 1] magnitudes

        z_pos = self.pos_encoder(x, pos_ei, pos_ea, batch)        # [B, 128]
        z_neg = self.neg_encoder(x, neg_ei, neg_ea, batch)        # [B, 128]

        z = torch.cat([z_pos, z_neg], dim=1)                      # [B, 256]
        return self.fusion_proj(z)                                 # [B, 128] = z_f


class AuxHead(nn.Module):
    """§5 prediction head: z_f → ŷ_f (scalar fluid intelligence). Stage 1 only."""

    def __init__(self, z_dim: int = Z_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32),    nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


# ===========================================================================
# 5. Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(encoder, head, loader, device, loss_fn,
             y_mean: float, y_std: float) -> dict:
    """Loss in z-scored space; MAE/MSE/R²/Pearson reported on original 0-13 scale."""
    encoder.eval(); head.eval()
    preds_z, ys_z, losses = [], [], []
    for batch in loader:
        batch = batch.to(device)
        z     = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        yhat  = head(z)
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
# 6. Main
# ===========================================================================

def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Graph: Alt 2 — positive/negative dual-channel dense graph')
    print(f'       pos edges: FC > 0  |  neg edges: |FC| where FC < 0')

    fc, eids, y = load_cohort()

    quintile_labels = pd.qcut(y, q=5, labels=False, duplicates='drop').astype(int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics    = []
    fold_enc_states = []
    all_splits      = []

    for fold, (train_val_idx, test_idx) in enumerate(
            skf.split(np.arange(len(eids)), quintile_labels)):

        print(f'\n{"="*60}')
        print(f'FOLD {fold+1}/{N_FOLDS}  '
              f'(train_val={len(train_val_idx)}, test={len(test_idx)})')
        print(f'{"="*60}')

        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.125, random_state=SEED,
            stratify=quintile_labels[train_val_idx],
        )
        print(f'[split] train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')

        # Feature cache — reuse Alt 1's if it exists (identical computation)
        fold_cache  = _fold_cache_path(fold)
        graph_feats = compute_graph_features(fc, train_idx, fold_cache)

        ds = ConnectomeDataset(fc, graph_feats, y)
        ds.apply_normalization(train_idx)

        loader_kw    = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        train_loader = DataLoader(Subset(ds, train_idx), shuffle=True,  **loader_kw)
        val_loader   = DataLoader(Subset(ds, val_idx),   shuffle=False, **loader_kw)
        test_loader  = DataLoader(Subset(ds, test_idx),  shuffle=False, **loader_kw)

        encoder = DualChannelEncoder(node_feat_dim=NODE_DIM, dropout=DROPOUT).to(device)
        head    = AuxHead(z_dim=Z_DIM, dropout=DROPOUT).to(device)
        if fold == 0:
            enc_params  = sum(p.numel() for p in encoder.parameters())
            head_params = sum(p.numel() for p in head.parameters())
            print(f'[model] encoder {enc_params:,} params '
                  f'(2 × SingleChannelEncoder + fusion_proj)  '
                  f'head {head_params:,} params')

        all_params = list(encoder.parameters()) + list(head.parameters())
        opt     = AdamW(all_params, lr=LR, weight_decay=WD)
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
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
                t_losses.append(loss.item())
            sched.step()

            val_m = evaluate(encoder, head, val_loader, device,
                             loss_fn, ds.y_mean, ds.y_std)
            print(f'  [ep {epoch:3d}] train={np.mean(t_losses):.4f}  '
                  f'val_loss={val_m["loss"]:.4f}  MAE={val_m["mae"]:.4f}  '
                  f'r={val_m["pearson"]:.4f}  lr={opt.param_groups[0]["lr"]:.2e}')

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
            'fold':         fold,
            'train_eids':   eids[train_idx].tolist(),
            'val_eids':     eids[val_idx].tolist(),
            'test_eids':    eids[test_idx].tolist(),
            'y_mean':       ds.y_mean,
            'y_std':        ds.y_std,
            'best_epoch':   best_state['epoch'] if best_state else None,
            'best_val_mae': float(best_val_mae),
            'test':         test_m,
        })

    maes = [m['mae']     for m in fold_metrics]
    rs   = [m['pearson'] for m in fold_metrics]
    r2s  = [m['r2']      for m in fold_metrics]

    print(f'\n{"="*60}')
    print(f'CV RESULTS  ({N_FOLDS}-fold, stratified by fluid IQ quintile)')
    print(f'Graph: Alt 2 — positive/negative dual-channel dense graph')
    print(f'{"="*60}')
    print(f'  MAE     : {np.mean(maes):.4f} ± {np.std(maes):.4f}  (primary)')
    print(f'  Pearson : {np.mean(rs):.4f} ± {np.std(rs):.4f}  (secondary)')
    print(f'  R²      : {np.mean(r2s):.4f} ± {np.std(r2s):.4f}')

    best_fold = int(np.argmin(maes))
    print(f'\n[best fold] {best_fold+1} (MAE={maes[best_fold]:.4f})')
    torch.save(fold_enc_states[best_fold], OUT_WEIGHTS)
    print(f'[save] best encoder → {OUT_WEIGHTS}')

    summary = {
        'graph':      'alt2_dual_channel_pos_neg',
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
