---
tags:
  - research
  - status/in-progress
Related to:
  - "[[TCG-Net-Analysis]]"
  - "[[TCG-Net Research Log]]"
  - "[[Research]]"
Date: "20260519"
Update date:
Source: TCG-Net paper, section 3.4
---

```ad-summary
Implementation plan for TCG-Net Stage 1: single-modality fMRI prediction of fluid intelligence using a GCN-GAT encoder. Proteomics is out of scope for now. This note covers the graph definition choices, data pipeline, and model architecture.
```

## Scope

Goal: predict **fluid intelligence** from fMRI data alone using the GCN-GAT encoder defined in the paper's section 3.4. No proteomics, no Shapley fusion. The checkpoint produced here becomes the fMRI encoder init for Stage 2 later.

---

## Step 1 — fMRI Data Preprocessing

- `FC_corr.py`: compute Pearson correlation from BOLD time-series `[T, R]` per subject, align 68→66 to the sMRI ROI order, zero the diagonal → saves **raw Pearson** FC to `corr_matrix.mat`
- Training scripts: clip to ±0.999999, apply Fisher-z transform (`arctanh`) at dataset construction
- Result: one `[R, R]` FC matrix per subject
- **All preprocessing fit on training indices only — no leakage into val/test**

---

## Step 2 — Graph Definition

The paper defines one main graph and two alternatives as sensitivity analyses. The choice determines what the GCN-GAT receives as `edge_index` and `edge_attr`.

### Main Graph — Group Sparse Mask
- Compute mean absolute FC across **training subjects only**
- Threshold to top 20% edges → binary `[R, R]` mask (shared topology across all subjects)
- Per-subject edge weights = individual FC values at those masked positions
- **Must be recomputed per CV fold** — this is the single most leakage-prone step

### Alternative 1 — Fully Weighted Individual FC Graph
$$A_i = FC_i$$
- No thresholding. Every ROI pair has an edge
- Edge weights = raw Fisher-z FC values per subject
- Each subject has their own full graph topology
- Simplest to implement; no group mask needed

### Alternative 2 — Positive/Negative Dual-Channel Graph
$$A_i^+ = \max(A_i,\ 0)$$
$$A_i^- = |\min(A_i,\ 0)|$$
- Split FC by sign into two separate adjacency matrices
- Two separate GCN-GAT encoders run in parallel (one per channel)
- Outputs concatenated: `[B, 128]` + `[B, 128]` → `[B, 256]`, then a `Linear(256→128)` fusion projection back to 128 for Stage-2 API compatibility
- Most biologically motivated: treats excitatory and inhibitory FC as distinct signals

### Comparison

| | Edge topology | Channels | Per-subject topology |
|---|---|---|---|
| Main | Sparse (top 20%) | 1 | No (shared mask) |
| Alt 1 | Dense (full) | 1 | Yes |
| Alt 2 | Dense (full) | 2 (+/−) | Yes |

**Implementation order:** start with Alt 1 (easiest, no mask risk) → validate MAE → swap to main sparse mask → treat Alt 2 as sensitivity experiment.

---

## Step 3 — Node Feature Construction

For each subject, each ROI node gets a feature vector `[D = R + 6]`:
- FC profile row (R values — connectivity to all other ROIs)
- 6 graph-theoretic metrics: nodal strength, weighted clustering coefficient, betweenness centrality, local efficiency, participation coefficient, within-module degree

Last two require a community partition — compute once from group-mean FC on training subjects via Louvain (networkx, seeded). All six metrics are implemented in batched numpy — no `bctpy` dependency.

---

## Step 4 — BrainGraphDataset (PyG)

Wrap each subject as a `torch_geometric.data.Data` object:
- `x`: node features `[R, D]`
- `edge_index`: from group mask or full graph `[2, E]`
- `edge_attr`: per-subject FC values at those edges `[E]`
- `y`: fluid intelligence score (scalar)

Use `PyGDataLoader` from `torch_geometric.loader` — not standard PyTorch DataLoader.

---

## Step 5 — GCN-GAT Encoder

Architecture: `GCNConv → BN → GELU → GATConv (4 heads) → BN → GELU → GCNConv → global mean+max pool → Linear → [B, 128]`

Key details:
- GAT attention weights cached as `self.last_attn` for interpretability later
- `add_self_loops=True`, clamp `edge_attr` to `[-3, 3]` to prevent attention logit explosion
- For Alt 2: instantiate two encoders, concatenate outputs → projection to 128

---

## Step 6 — Prediction Head

`Linear(128, 64) → GELU → Dropout(0.5) → Linear(64, 32) → GELU → Linear(32, 1)`

For Alt 2 the head input dim stays 128 — the encoder fuses the two channels back to 128 internally.

---

## Step 7 — Training

- Loss: `F.huber_loss` (delta = 1.0)
- Optimizer: `AdamW(lr=5e-4, weight_decay=1e-3)` (tuned down from 1e-3/1e-4 — stronger L2 against overfitting)
- Gradient clipping: 1.0 globally
- Early stopping on val MAE, patience 30
- ~50–100 epochs
- Save best checkpoint: `Data/fmri_stage1_encoder.pt` (best fold), plus per-fold `Data/fmri_stage1_encoder_fold{k}.pt`; Alt 1/2 use `_alt1` / `_alt2` suffixes

---

## Step 8 — Evaluation

- Primary: MAE on held-out test fold
- Secondary: Pearson r between predicted and actual fluid intelligence scores
- Cross-validation: 5-fold or 10-fold, stratified by fluid intelligence quintiles

---

## Known Pitfalls

- Group mask must be recomputed per CV fold using only that fold's training subjects
- GAT can produce NaN attention if a node has one neighbour — use `add_self_loops=True`
- Check `nx.number_connected_components` per subject — if sparse mask disconnects graphs, lower threshold to top 30%
- BatchNorm unstable at very small batch sizes with graph data — consider LayerNorm as fallback
- PyG version must be pinned with matching `torch-scatter` / `torch-sparse` wheels

---

# Reference
- [[TCG-Net-Analysis]] — full architecture analysis, implementation code, pitfalls
- [[TCG-Net Research Log]] — ongoing research log
