# Proteomics Encoder — TCG-Net

The proteomics branch of TCG-Net converts a participant's Olink Explore
plasma protein profile (2,920 proteins) into a compact 128-dimensional
embedding `z_p` that the Stage 2 Cooperative Shapley Fusion Module
combines with the fMRI embedding `z_f`. The encoder also exposes a
1,024-dimensional SE gate vector `s_p` used both for sparsity
regularisation during Stage 1 pretraining and as a per-latent-channel
importance readout for downstream interpretability.

---

## I/O Contract

| Tensor | Shape | Description |
|---|---|---|
| Input `X_p` | `(N, 2920)` | z-scored (or covariate-residualized + z-scored) plasma protein matrix, imputed and scaled using statistics fit on the training split only |
| Output `z_p` | `(B, 128)` | Final proteomics embedding; consumed by `ProteomicsAuxHead` in Stage 1 and by the Shapley Fusion Module in Stage 2 |
| Output `s_p` | `(B, 1024)` | SE gate vector (values in (0, 1)); used for the `||s_p||_1` sparsity term in `L_stage1` and for latent-channel importance analysis |

---

## Architecture — SE-MLP

```
x  (B, 2920)
  → Linear(2920, 1024) → BatchNorm1d(1024) → GELU → Dropout(0.3)    # stage 1
  → SEBlock: Linear(1024, 256) → GELU → Linear(256, 1024) → Sigmoid  # squeeze-and-excite
               produces gate s_p ∈ (0,1)^1024; applies h̃ = h ⊙ s_p
  → Linear(1024, 512)  → BatchNorm1d(512)  → GELU → Dropout(0.3)    # stage 2
  → Linear(512, 256)   → BatchNorm1d(256)  → GELU → Dropout(0.2)    # stage 3
  → Linear(256, 128)                                                  # projection
z_p (B, 128),  s_p (B, 1024)
```

Stage 1 additionally instantiates `ProteomicsAuxHead = Linear(128, 1)` which
produces the auxiliary scalar prediction `y_p`. The aux head is **not part of
the encoder** and is discarded at the Stage 1 → Stage 2 handoff.

**Parameter counts (confirmed on real data):**
- `ProteomicsEncoder`: 4,209,280 trainable parameters
- `ProteomicsAuxHead`: 129 trainable parameters

---

## Running the Full Pipeline

All scripts must be run from the repository root (`TCG-Net/`'s parent directory)
using the `tcg_net` conda environment.

### Step 1 — Preprocessing

KNN imputation and z-score normalisation, split-aware (fit on train only).
**Warning: KNN imputation on 52,697 participants × 2,920 proteins takes
approximately 3 hours on CPU.** The fitted imputer (925 MB) is saved and
reused by all subsequent runs — you only need to run this once.

```bash
# Without covariate residualization (default):
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/data/preprocess.py

# With covariate residualization (batch, plate, fasting time; opt-in):
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/data/preprocess.py --residualize
```

### Step 2 — Build target datasets

```bash
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/data/build_dataset.py --target age

conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/data/build_dataset.py --target fluid_iq
```

### Step 3 — Stage 1 training

```bash
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/training/train_stage1.py --target age

conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/training/train_stage1.py --target fluid_iq
```

### Step 4 — Protein importance analysis

Replace `<run_id>` with the actual run ID from the checkpoint filename.

```bash
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/analysis/protein_importance.py \
  --target age \
  --checkpoint TCG-Net/proteomics/checkpoints/stage1_age_<run_id>_best.pt

conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/analysis/protein_importance.py \
  --target fluid_iq \
  --checkpoint TCG-Net/proteomics/checkpoints/stage1_fluid_iq_<run_id>_best.pt
```

### Step 5 — K=5 cross-validation (optional)

```bash
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/training/cv_stage1.py --target age

conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/training/cv_stage1.py --target fluid_iq
```

---

## Confirmed Training Results (Stage 1, real UK Biobank data)

### Age (UKB field 21003-0.0)

| Split | R² | MAE | Pearson r |
|---|---|---|---|
| Validation (epoch 13) | 0.8336 | 2.613 years | 0.919 |
| **Test (best epoch 13 / 30)** | **0.868** | **2.352 years** | **0.933** |

The age model exhibits mild late-training drift: validation loss rises from
its best of 8.70 at epoch 13 back to 12.39 at epoch 30. A learning rate
scheduler (e.g. `ReduceLROnPlateau`) is recommended for future runs.

Checkpoint: `stage1_age_20260508_025644_14dae01e_best.pt`

### Fluid IQ (UKB field 20016-0.0)

| Split | R² | MAE | Pearson r |
|---|---|---|---|
| Validation (epoch 1) | 0.047 | 1.723 points | 0.225 |
| **Test (best epoch 1 / 30)** | **0.067** | **1.681 points** | **0.263** |

**Note on fluid IQ:** The weak result is not an encoder error. UK Biobank
field 20016-0.0 is a discrete ordinal variable with only 14 unique integer
values (0–13, std = 2.17). The proteomic signal for this target is weak and
consistent with published literature — proteomics alone does not explain much
fluid intelligence variance. Ordinal regression should be considered at the
fusion stage (Stage 2), not addressed by changing the Stage 1 encoder.

Checkpoint: `stage1_fluid_iq_20260508_031652_4abd9cf0_best.pt`

---

## Known Limitations and Recommendations

| Issue | Recommendation |
|---|---|
| Age model late-epoch drift | Add `torch.optim.lr_scheduler.ReduceLROnPlateau` in Stage 1 |
| Fluid IQ weak signal | Revisit with ordinal regression head at the Stage 2 fusion layer |
| KNN imputation runtime ~3 h on CPU | Imputer is saved at `data/artefacts/knn_imputer_1234.joblib`; do not re-run unless the raw data or split changes |
| Residualization covariate set | `--residualize` currently uses batch, plate, and fasting time from `RAW_PROTEIN_CSV`; age and sex are pulled from `phesantallphenos` files automatically. Full covariate set pending verification. |
| K=5 CV pending | CV runs have not yet been executed on real data |

---

## Stage 1 Checkpoint Structure

Two checkpoints are saved per training run:

- `checkpoints/stage1_<target>_<run_id>_best.pt` — weights at the best validation epoch.
- `checkpoints/stage1_<target>_<run_id>_last.pt` — weights at the final epoch (for resumption).

Each checkpoint is a `torch.save` dict with the following keys:

```python
{
  "stage":                "stage1",
  "run_id":               "<YYYYmmdd_HHMMSS_hex>",
  "epoch":                13,               # epoch at which this was saved
  "encoder_state_dict":   { ... },          # ProteomicsEncoder weights only
  "aux_head_state_dict":  { ... },          # ProteomicsAuxHead weights (Stage 1 only)
  "optimizer_state_dict": { ... },          # Adam state (for resumption from last.pt)
  "config": {
      "lr", "batch_size", "epochs", "weight_decay",
      "lambda_sparse", "lambda_aux", "target",
      "global_seed", "split_seed"
  },
  "val_metrics": {
      "val_loss", "val_r2", "val_mae", "val_pearson_r", "sp_l1_norm", "epoch"
  },
}
```

---

## Loading the Encoder for Stage 2

Stage 2 needs only the encoder weights. The aux head is discarded.

```python
import torch
from models.encoder import ProteomicsEncoder

checkpoint_path = (
    "TCG-Net/proteomics/checkpoints/"
    "stage1_age_20260508_025644_14dae01e_best.pt"
)
ckpt = torch.load(checkpoint_path, map_location="cpu")
encoder = ProteomicsEncoder()
encoder.load_state_dict(ckpt["encoder_state_dict"])
encoder.eval()
# encoder is now ready for the Stage 2 fusion module.
```

The fitted scaler, imputer, and (if used) covariate-residualizer artefacts
live under `data/artefacts/` and **must be reapplied** to any new input
to keep the Stage 2 input distribution consistent with Stage 1.

---

## Log File Locations

| Type | Path pattern | Producer | Format |
|---|---|---|---|
| Process | `logs/process/process_<script>_<timestamp>.log` | every entry script | Plain text `[YYYY-MM-DD HH:MM:SS] [LEVEL] message` |
| Experiment | `logs/experiments/experiment_<run_id>.json` | `train_stage1.py`, `cv_stage1.py` | JSON; written once before training starts |
| Training CSV | `logs/training/train_log_<target>_<run_id>.csv` | `train_stage1.py`, `cv_stage1.py` | CSV; one row per epoch, atomic append |
| Test metrics | `logs/experiments/test_metrics_<target>_<run_id>.json` | `train_stage1.py` | JSON; written once after training |
| CV summary CSV | `logs/training/cv_summary_<target>_<run_id>.csv` | `cv_stage1.py` | CSV; one row per fold + mean±std row |
| CV summary JSON | `logs/experiments/cv_summary_<target>_<run_id>.json` | `cv_stage1.py` | JSON; written after all folds |
| Preprocessing | `logs/preprocessing/preprocess_log_<timestamp>.json` | `preprocess.py` | JSON; written once at end |
| Dataset build | `logs/preprocessing/dataset_build_<target>_<timestamp>.json` | `build_dataset.py` | JSON; written once at end |
| Importance | `logs/experiments/importance_<target>_<run_id>.json` | `protein_importance.py` | JSON; written once at end |

All timestamps are UTC. Run IDs follow `YYYYmmdd_HHMMSS_<8-char hex>`.

---

## Artefact Inventory

A complete pipeline run produces the following files:

| File | Path | Producer |
|---|---|---|
| Processed proteomics tensor | `data/proteomics_tensor.pt` | `preprocess.py` |
| Global split indices | `data/splits/split_1234.npz` | `preprocess.py` |
| KNN imputer | `data/artefacts/knn_imputer_1234.joblib` | `preprocess.py` |
| StandardScaler (joblib) | `data/artefacts/scaler_1234.joblib` | `preprocess.py` |
| StandardScaler (npz) | `data/artefacts/scaler_1234.npz` | `preprocess.py` |
| Covariate residualizer | `data/artefacts/covariate_resid_1234.npz` | `preprocess.py` (if `--residualize`) |
| Age dataset bundle | `data/dataset_age.pt` | `build_dataset.py --target age` |
| Fluid IQ dataset bundle | `data/dataset_fluid_iq.pt` | `build_dataset.py --target fluid_iq` |
| Age split indices | `data/splits/split_age_1234.npz` | `build_dataset.py --target age` |
| Fluid IQ split indices | `data/splits/split_fluid_iq_1234.npz` | `build_dataset.py --target fluid_iq` |
| Best checkpoint | `checkpoints/stage1_<target>_<run_id>_best.pt` | `train_stage1.py` |
| Last checkpoint | `checkpoints/stage1_<target>_<run_id>_last.pt` | `train_stage1.py` |
| Protein importance CSV | `analysis_outputs/protein_importance_<target>_<run_id>.csv` | `protein_importance.py` |
| SE gate mean | `analysis_outputs/sp_mean_<target>_<run_id>.npy` | `protein_importance.py` |
| CV fold indices | `data/splits/cv_<target>_5fold_seed7_<run_id>.npz` | `cv_stage1.py` |
