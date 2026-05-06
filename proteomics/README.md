# Proteomics encoder

The proteomics branch of TCG-Net. This module implements the plasma-protein
encoder that produces the modality-specific embedding `z_p` consumed by the
Stage 2 Cooperative Shapley Fusion Module, and exposes the per-hidden-unit
gate vector `s_p` used both for the sparsity regulariser and for downstream
interpretability.

## Role in TCG-Net

In TCG-Net's two-branch design, the proteomics encoder converts a
participant's plasma protein profile into a low-dimensional embedding that
the fusion stage can combine with the fMRI embedding `z_f`. During Stage 1
the encoder is pretrained on a single cognitive target (`age` or
`fluid_iq`) with an auxiliary regression head; during Stage 2 the head is
discarded and only the encoder weights are loaded into the joint TCG-Net.

## I/O contract

- **Input**: `X_p ∈ R^(N×2920)` — z-scored or covariate-residualized
  plasma protein measurements. Imputed and scaled using statistics fit on
  the training split only (no leakage).
- **Output**: `z_p ∈ R^(B×128)` — the 128-d embedding consumed by the
  fusion stage.
- **Auxiliary output**: `s_p ∈ R^(B×1024)` — the SE-block gate over the
  1024-d hidden representation. Used as the `||s_p||_1` term in
  `L_sparse` and as a per-hidden-channel importance vector.

## Architecture summary (SE-MLP)

```
x  (B, 2920)
  → Linear(2920, 1024) → BN → GELU → Dropout(0.3)        # stage 1
  → SE-Block(1024, 256)        gate s_p, multiply       # squeeze-and-excite
  → Linear(1024, 512)  → BN → GELU → Dropout(0.3)        # stage 2
  → Linear(512, 256)   → BN → GELU → Dropout(0.2)        # stage 3
  → Linear(256, 128)                                     # projection
z_p (B, 128),  s_p (B, 1024)
```

Stage 1 additionally instantiates a `ProteomicsAuxHead = Linear(128, 1)`
that produces the auxiliary scalar prediction `y_p`. The aux head is **not
part of the encoder** and is dropped at the Stage 1 → Stage 2 handoff.

## Log artefacts

The pipeline emits four log types, all rooted under `proteomics/logs/`:

| Type           | Path                                                              | Producer                              | Format |
|----------------|-------------------------------------------------------------------|---------------------------------------|--------|
| Process        | `logs/process/process_<script>_<timestamp>.log`                   | every entry script                    | plain text, `[YYYY-MM-DD HH:MM:SS] [LEVEL] message` |
| Experiment     | `logs/experiments/experiment_<run_id>_<timestamp>.json`           | `train_stage1.py` / `cv_stage1.py`    | JSON, written once before training |
| Training       | `logs/training/train_log_<target>_<timestamp>.csv`                | `train_stage1.py` / `cv_stage1.py`    | CSV, one row per epoch (atomic append) |
| Preprocessing  | `logs/preprocessing/preprocess_log_<timestamp>.json`              | `preprocess.py` / `build_dataset.py`  | JSON, written once at end of run |

All timestamps are UTC. Run IDs follow the format `YYYYmmdd_HHMMSS_<8-char hex>`.

## Running the pipeline

> **Note:** the data scripts, model modules, training loops and analysis
> tools listed below have not yet been implemented in this monorepo. The
> commands are forward-looking placeholders; they will become live in
> subsequent sessions as Steps 2.4 onward of the rebuild plan are landed.

```bash
# 1. Preprocessing — split-aware imputation + scaling.
python -m proteomics.data.preprocess [--residualize] [--device cpu]

# 2. Build a target-specific (x, y, splits) dataset.
python -m proteomics.data.build_dataset --target age
python -m proteomics.data.build_dataset --target fluid_iq

# 3. Stage 1 pretraining (with mandatory K=5 cross-validation).
python -m proteomics.training.train_stage1 --target age
python -m proteomics.training.train_stage1 --target fluid_iq

# 4. Per-protein importance analysis.
python -m proteomics.analysis.protein_importance --target age \
    --checkpoint proteomics/checkpoints/stage1_age_<run_id>.pt
```

## Stage 1 checkpoint structure

Two checkpoints are saved per training run:

- `proteomics/checkpoints/stage1_<target>_<run_id>_best.pt` — best validation epoch.
- `proteomics/checkpoints/stage1_<target>_<run_id>_last.pt` — last epoch (for resumption).

Each checkpoint is a single torch.save dict with keys:

```python
{
  "stage": "stage1",
  "run_id": "<run_id>",
  "epoch": <int>,
  "encoder_state_dict": ...,    # ProteomicsEncoder weights
  "aux_head_state_dict": ...,   # ProteomicsAuxHead weights
  "optimizer_state_dict": ...,  # for resumption from last.pt
  "config": { ... },            # full hyperparameter snapshot
  "val_metrics": { val_loss, val_r2, val_mae, val_pearson_r, sp_l1_norm },
}
```

## Loading the encoder for Stage 2

Stage 2 needs only the encoder weights; the aux head is discarded.

```python
import torch
from proteomics.models.encoder import ProteomicsEncoder

ckpt = torch.load("proteomics/checkpoints/stage1_age_<run_id>_best.pt", map_location="cpu")
encoder = ProteomicsEncoder()
encoder.load_state_dict(ckpt["encoder_state_dict"])
encoder.eval()
```

The persisted scaler / imputer / (optional) covariate-residualizer artefacts
live under `proteomics/data/artefacts/` and must be reapplied to any new
inputs to keep the Stage 2 input distribution consistent with Stage 1.
