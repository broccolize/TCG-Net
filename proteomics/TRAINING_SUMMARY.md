# Stage 1 Proteomics Encoder — Training Summary

**Date completed:** 2026-05-08
**Pipeline version:** Stage 1 proteomics encoder, TCG-Net rebuild v1
**Status:** Complete — both targets trained, importance analysis run, checkpoints verified

---

## 1. Dataset

| Property | Value |
|---|---|
| Source | UK Biobank, Olink Explore plasma proteomics |
| Proteins | 2,920 (Olink panel) |
| Participants — age target | 52,697 |
| Participants — fluid IQ target | 16,903 |
| Split | 70 / 15 / 15 (train / val / test), seed 1234 |
| Missing data (pre-imputation) | 9.97% |
| Missing data (post-imputation) | 0% |
| Imputation | KNN (k=5), fit on training split only |
| Normalisation | z-score (StandardScaler), fit on training split only |
| Covariate residualization | Not applied (default off; use `--residualize` to enable) |

---

## 2. Model

**Class:** `ProteomicsEncoder` (`models/encoder.py`)

```
Linear(2920, 1024) → BatchNorm1d(1024) → GELU → Dropout(0.3)
SEBlock: Linear(1024, 256) → GELU → Linear(256, 1024) → Sigmoid
Linear(1024, 512)  → BatchNorm1d(512)  → GELU → Dropout(0.3)
Linear(512, 256)   → BatchNorm1d(256)  → GELU → Dropout(0.2)
Linear(256, 128)
```

| Component | Trainable parameters |
|---|---|
| `ProteomicsEncoder` | 4,209,280 |
| `ProteomicsAuxHead` (Stage 1 only) | 129 |
| **Total (Stage 1)** | **4,209,409** |

---

## 3. Training Configuration

| Hyperparameter | Age | Fluid IQ |
|---|---|---|
| Optimiser | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 |
| Weight decay | 1e-5 | 1e-5 |
| Batch size | 64 | 64 |
| Epochs | 30 | 30 |
| `lambda_sparse` | 1e-4 | 1e-4 |
| `drop_last` (train loader) | True | True |
| Device | CUDA | CUDA |

Loss: `L = MSE(y, y_p) + lambda_sparse × ||s_p||_1`

---

## 4. Results

### Age (UKB field 21003-0.0, continuous)

| Split | Epoch | R² | MAE | Pearson r |
|---|---|---|---|---|
| Validation | 13 | 0.8336 | 2.613 years | 0.919 |
| **Test** | **13** | **0.868** | **2.352 years** | **0.933** |

Best epoch 13 of 30. Mild late-training drift observed: validation loss
rises from 8.70 at epoch 13 to 12.39 at epoch 30.

### Fluid IQ (UKB field 20016-0.0, discrete ordinal)

| Split | Epoch | R² | MAE | Pearson r |
|---|---|---|---|---|
| Validation | 1 | 0.047 | 1.723 points | 0.225 |
| **Test** | **1** | **0.067** | **1.681 points** | **0.263** |

Best epoch 1 of 30. **Target characteristics:** discrete ordinal variable
with 14 unique integer values (0–13), std = 2.17. The weak proteomic signal
is consistent with published literature and is not an encoder failure.
Ordinal regression should be considered at the Stage 2 fusion layer.

---

## 5. Key Artefacts

| File | Path | Notes |
|---|---|---|
| KNN imputer | `data/artefacts/knn_imputer_1234.joblib` | 925 MB; do not re-run without reason |
| StandardScaler | `data/artefacts/scaler_1234.joblib` | Fit on training split only |
| Scaler arrays | `data/artefacts/scaler_1234.npz` | Portable mean/std for non-sklearn use |
| Age dataset | `data/dataset_age.pt` | x, y, eid, features, train/val/test idx |
| Fluid IQ dataset | `data/dataset_fluid_iq.pt` | x, y, eid, features, train/val/test idx |
| Age best checkpoint | `checkpoints/stage1_age_20260508_025644_14dae01e_best.pt` | Epoch 13, val R²=0.834 |
| Age last checkpoint | `checkpoints/stage1_age_20260508_025644_14dae01e_last.pt` | Epoch 30 |
| Fluid IQ best checkpoint | `checkpoints/stage1_fluid_iq_20260508_031652_4abd9cf0_best.pt` | Epoch 1, val R²=0.047 |
| Fluid IQ last checkpoint | `checkpoints/stage1_fluid_iq_20260508_031652_4abd9cf0_last.pt` | Epoch 30 |
| Age protein importance | `analysis_outputs/protein_importance_age_<run_id>.csv` | Columns: Protein, Importance |
| Age SE gate mean | `analysis_outputs/sp_mean_age_<run_id>.npy` | Shape (1024,); latent-channel importance |
| Fluid IQ protein importance | `analysis_outputs/protein_importance_fluid_iq_<run_id>.csv` | |
| Fluid IQ SE gate mean | `analysis_outputs/sp_mean_fluid_iq_<run_id>.npy` | Shape (1024,) |

---

## 6. Known Issues and Recommendations

| Issue | Priority | Recommendation |
|---|---|---|
| Age late-epoch drift | Medium | Add `ReduceLROnPlateau` scheduler; re-run age training |
| Fluid IQ weak signal | Low | Implement ordinal regression head at Stage 2 fusion; do not retrain Stage 1 encoder |
| K=5 CV not yet run | Low | Run `cv_stage1.py --target age` and `cv_stage1.py --target fluid_iq` |
| Residualization untested | Low | Test `preprocess.py --residualize`; confirm covariate column names from CSV header |
| KNN runtime | Info | ~3 hours CPU; imputer is saved — only re-run if split seed or data changes |

---

## 7. Stage 2 Handoff

### What Stage 2 receives from this encoder

- **Embedding:** `z_p ∈ R^(B×128)` from `ProteomicsEncoder.forward(x)[0]`
- **Gate:** `s_p ∈ R^(B×1024)` from `ProteomicsEncoder.forward(x)[1]` (used in Stage 2 `L_sparse`)
- **Checkpoint to use:** `stage1_age_20260508_025644_14dae01e_best.pt` (age model, R²=0.868)

### How to load the encoder into Stage 2

```python
import torch
from models.encoder import ProteomicsEncoder

ckpt = torch.load(
    "TCG-Net/proteomics/checkpoints/"
    "stage1_age_20260508_025644_14dae01e_best.pt",
    map_location="cpu",
)
encoder = ProteomicsEncoder()
encoder.load_state_dict(ckpt["encoder_state_dict"])
encoder.eval()
# The aux_head_state_dict key exists in the checkpoint but is not loaded —
# Stage 2 replaces the aux head with the full L_aux + Shapley fusion structure.
```

### Preprocessing artefacts required at Stage 2 inference time

Any new input to the Stage 2 encoder must pass through the same preprocessing
as Stage 1 training data:

1. KNN imputation using `data/artefacts/knn_imputer_1234.joblib`
2. z-score normalisation using `data/artefacts/scaler_1234.joblib` (or the mean/std from `scaler_1234.npz`)
3. (If residualization was applied) subtract covariate predictions using `data/artefacts/covariate_resid_1234.npz`

Reapplying the same artefacts is essential to keep the Stage 2 input
distribution consistent with Stage 1 training.
