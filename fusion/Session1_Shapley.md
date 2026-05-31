# Session 1 — Cooperative Shapley Fusion Module (TCG-Net Stage 2, §4–§6)

## Context
Both Stage 1 encoders are complete:
- fMRI: `ConnectomeEncoder` → `z_f ∈ ℝ^{B×128}`, checkpoint at `Data/fmri_stage1_encoder.pt` (flat state dict)
- Proteomics: `ProteomicsEncoder` → `(z_p ∈ ℝ^{B×128}, s_p ∈ ℝ^{B×1024})`, checkpoint key `"encoder_state_dict"`

Data alignment is being handled by a separate colleague. This session covers only the **model and loss modules** that live in `fusion/` — i.e., everything except the training script and data loader.

Goal: implement the Cooperative Shapley Fusion Module (§4), the final Prediction Head (§5), and the Stage 2 loss function (§6) as standalone, testable PyTorch modules.

---

## File Structure

```
fusion/
├── models/
│   ├── __init__.py               — exports FusionOutput, CooperativeShapleyFusion, PredictionHead
│   ├── shapley_fusion.py         — §4 full: three aux heads + Shapley values + fusion gate
│   └── prediction_head.py       — §5: final regression head z_fusion → ŷ
├── training/
│   ├── __init__.py               — empty
│   └── losses.py                 — §6: Stage2Loss (L_main + L_aux + L_shapley + L_sparse)
└── Session1_Shapley.md           — this file
```

---

## File Details

### `fusion/models/shapley_fusion.py` — §4 (entire section)

**Class:** `CooperativeShapleyFusion(nn.Module)`

**What it implements:**

| Sub-component | PDF section | Description |
|---|---|---|
| `H_f = Linear(128, 1)` | §4.1 | fMRI-only aux head: `z_f → ŷ_f` |
| `H_p = Linear(128, 1)` | §4.1 | Proteomics-only aux head: `z_p → ŷ_p` |
| `H_fp = Linear(512,256)→GELU→Dropout→Linear(256,1)` | §4.1 | Joint aux head: `z_joint → ŷ_fp`, where `z_joint = [z_f, z_p, z_f⊙z_p, \|z_f−z_p\|] ∈ ℝ^{512}` |
| `y_baseline` buffer | §4.2 | `ŷ_0 = mean(y_train)` — scalar, not learned, set via `set_baseline(y_mean)` |
| φ_f, φ_p, φ_int computation | §4.2 | Analytical Shapley values (no approximation, no extra params) |
| `W_f, W_p, W_int` = `Linear(128,128,bias=False)` | §4.3 | Modality-specific projections for the fusion gate |
| Fusion gate | §4.3 | `w = softmax([φ_f, φ_p, φ_int])` → weighted sum of projected embeddings |

**Forward signature:**
```python
def forward(self, z_f: Tensor, z_p: Tensor) -> FusionOutput
```

**FusionOutput dataclass** (returned by forward, carries everything needed for loss):
```python
@dataclass
class FusionOutput:
    z_fusion : Tensor  # [B, 128]  → fed to PredictionHead
    y_hat_f  : Tensor  # [B]       → fMRI-only prediction (§4.1)
    y_hat_p  : Tensor  # [B]       → proteomics-only prediction (§4.1)
    y_hat_fp : Tensor  # [B]       → joint prediction (§4.1)
    phi_f    : Tensor  # [B]       → fMRI Shapley value (§4.2)
    phi_p    : Tensor  # [B]       → proteomics Shapley value (§4.2)
    phi_int  : Tensor  # [B]       → interaction Shapley value (§4.2)
    w        : Tensor  # [B, 3]    → fusion gate weights [w_f, w_p, w_int] (§4.3, needed for L_sparse)
```

**Shapley computation (§4.2):**
```
φ_f   = 0.5 * ((ŷ_f − ŷ_0) + (ŷ_fp − ŷ_p))
φ_p   = 0.5 * ((ŷ_p − ŷ_0) + (ŷ_fp − ŷ_f))
φ_int = ŷ_fp − ŷ_0 − φ_f − φ_p
```

**Fusion gate (§4.3):**
```
w = softmax([φ_f, φ_p, φ_int], dim=1)          [B, 3]
z_fusion = w_f·W_f(z_f) + w_p·W_p(z_p) + w_int·W_int(z_f⊙z_p)   [B, 128]
```

---

### `fusion/models/prediction_head.py` — §5

**Class:** `PredictionHead(nn.Module)`

**Architecture (exactly §5):**
```
Linear(128, 64) → GELU → Dropout(0.2) → Linear(64, 32) → GELU → Linear(32, 1)
```

**Forward signature:**
```python
def forward(self, z_fusion: Tensor) -> Tensor  # [B, 128] → [B]
```

---

### `fusion/training/losses.py` — §6

**Class:** `Stage2Loss(nn.Module)`

**Loss formula (§6):**
```
L = L_main + λ1·L_aux + λ2·L_shapley + λ3·L_sparse
```

| Term | Formula | Default λ |
|---|---|---|
| `L_main` | `Huber(y, ŷ)` | — (always 1.0) |
| `L_aux` | `MSE(y, ŷ_f) + MSE(y, ŷ_p) + MSE(y, ŷ_fp)` | λ1 = 0.3 |
| `L_shapley` | `‖ŷ_fp − (ŷ_0 + φ_f + φ_p + φ_int)‖²` | λ2 = 0.1 |
| `L_sparse` | `‖w‖₁ + ‖s_p‖₁ + ‖α_f‖₁` | λ3 = 1e-4 |

**Forward signature:**
```python
def forward(
    self,
    y_true:  Tensor,        # [B]        ground truth (z-scored)
    y_hat:   Tensor,        # [B]        final prediction from PredictionHead
    fusion:  FusionOutput,  # dataclass from CooperativeShapleyFusion
    s_p:     Tensor,        # [B, 1024]  SE gate from ProteomicsEncoder
    alpha_f: Tensor,        # [E, heads] GAT attention weights from ConnectomeEncoder.last_attn
) -> dict[str, Tensor]
```

**Returns dict** (matching proteomics convention):
```python
{
    "total":     L,       # call .backward() on this
    "l_main":    L_main,
    "l_aux":     L_aux,
    "l_shapley": L_shapley,
    "l_sparse":  L_sparse,
    "w_l1":      ...,     # per-sample mean ‖w‖₁, for logging
    "sp_l1":     ...,     # per-sample mean ‖s_p‖₁, for logging
    "af_l1":     ...,     # per-sample mean ‖α_f‖₁, for logging
}
```

---

## Key Design Decisions

1. **H_f and H_p are `Linear(128, 1)`** — single layer, no hidden, matching `ProteomicsAuxHead`. These are aux heads for Shapley computation, not the final predictor.
2. **H_fp is deeper** — must capture cross-modal interactions from a 512-d input.
3. **`y_baseline` is a buffer, not a parameter** — set once from `mean(y_train)` per fold; saved/loaded with state dict but never updated by optimizer.
4. **`W_f, W_p, W_int` have no bias** — Shapley weights already provide per-sample scaling; a bias would interfere with the game-theoretic interpretation.
5. **`w` is included in `FusionOutput`** — needed for `‖w‖₁` in `L_sparse` and for per-subject interpretability (§11: which modality dominated).
6. **Loss follows proteomics dict convention** — `"total"` key for `.backward()`, individual terms for logging. Matches `ProteomicsStage1Loss` in `proteomics/training/losses.py`.

---

## Verification (shape check — no real data needed)

```python
import torch
from fusion.models import CooperativeShapleyFusion, PredictionHead

B = 8
z_f = torch.randn(B, 128)
z_p = torch.randn(B, 128)

fusion = CooperativeShapleyFusion()
fusion.set_baseline(5.5)
head = PredictionHead()

out = fusion(z_f, z_p)
assert out.z_fusion.shape == (B, 128)
assert out.y_hat_f.shape  == (B,)
assert out.y_hat_p.shape  == (B,)
assert out.y_hat_fp.shape == (B,)
assert out.phi_f.shape    == (B,)
assert out.w.shape        == (B, 3)

y_hat = head(out.z_fusion)
assert y_hat.shape == (B,)
```

---

## Session 1 Completion Status — 2026-05-31

**Status: COMPLETE ✅ — all shapes verified, `All shapes OK — Session 1 complete` printed**

### What was built

All three files implemented and shape-checked with random tensors (no real data needed):

#### `fusion/models/shapley_fusion.py`
- `FusionOutput` dataclass — holds all forward pass outputs needed by the loss
- `CooperativeShapleyFusion(nn.Module)`:
  - `__init__`: H_f, H_p, H_fp, y_baseline buffer, W_f, W_p, W_int
  - `set_baseline(y_mean)` — sets ŷ_0 from training set mean via `.fill_()`
  - `forward(z_f, z_p)` — §4.1 three pathways, §4.2 Shapley values, §4.3 fusion gate → returns `FusionOutput`

#### `fusion/models/prediction_head.py`
- `PredictionHead(nn.Module)`:
  - `__init__`: `nn.Sequential(Linear(128,64)→GELU→Dropout→Linear(64,32)→GELU→Linear(32,out_dim))`
  - `forward(z_fusion)` → `ŷ [B]`
  - `out_dim` parameter supports both single-task (1) and multi-task (K) prediction

#### `fusion/training/losses.py`
- `Stage2Loss(nn.Module)`:
  - `__init__`: stores lambda_aux=0.3, lambda_shapley=0.1, lambda_sparse=1e-4, HuberLoss
  - `forward(y_true, y_hat, fusion, s_p, alpha_f, fusion_model)` — computes all four loss terms
  - Returns dict with keys: `total`, `l_main`, `l_aux`, `l_shapley`, `l_sparse`, `w_l1`, `sp_l1`, `af_l1`

#### `fusion/models/__init__.py`
- Exports `FusionOutput`, `CooperativeShapleyFusion`, `PredictionHead`

### Key design decisions made during session

- `y_baseline` lives in `CooperativeShapleyFusion` as a registered buffer (not in `FusionOutput`) — keeps the dataclass clean and reads from the model in `Stage2Loss` via `fusion_model.y_baseline`
- `L_aux` uses `F.mse_loss` (not Huber) — PDF §6.2 specifies MSE for auxiliary terms
- `L_sparse` uses `.abs().mean()` per term — size-invariant L1 norm so w, s_p, alpha_f terms are comparable despite different shapes
- `w_l1`, `sp_l1`, `af_l1` stored separately in return dict — needed individually for training logs

### What's next (Session 2)

1. **Data alignment pipeline** — colleague handling; produces aligned (fMRI, proteomics, y) dataset with consistent train/val/test splits across both modalities
2. **`fusion/training/train_stage2.py`** — wires `ConnectomeEncoder` + `ProteomicsEncoder` + `CooperativeShapleyFusion` + `PredictionHead` + `Stage2Loss` together with §7.2 freeze/unfreeze protocol:
   - Load pretrained encoder weights
   - First 10 epochs: freeze both encoders, train fusion + heads only (lr=5e-4)
   - After epoch 10: unfreeze all, encoder lr=1e-5, fusion/head lr=1e-4
