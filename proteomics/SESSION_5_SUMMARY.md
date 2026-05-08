# Session 5 Summary
**Date:** 2026-05-07 (UTC)
**Scope:** Steps 2.11, 2.12, 2.13 of the proteomics encoder rebuild plan

---

## What Was Done

### Files Read at Session Start (per plan requirement)
All eight required files were read in full before any implementation began:

| File | Purpose |
|---|---|
| `proteomics/proteomics_encoder_rebuild_plan.md` | Rebuild plan and spec |
| `proteomics/config.py` | All path and hyperparameter constants |
| `proteomics/utils/seed.py` | `set_global_seeds`, `make_split_generator`, `seed_worker` |
| `proteomics/utils/logging_utils.py` | `make_process_logger`, `make_run_id`, `dump_json`, `append_csv_row` |
| `proteomics/models/se_block.py` | SEBlock architecture |
| `proteomics/models/encoder.py` | ProteomicsEncoder |
| `proteomics/models/aux_head.py` | ProteomicsAuxHead |
| `proteomics/training/losses.py` | ProteomicsStage1Loss |

---

## Files Created

### Step 2.11 — `TCG-Net/proteomics/training/train_stage1.py`
Stage 1 single-modality pretraining loop.

**Key behaviours (in execution order):**
1. `set_global_seeds(args.seed)` — first line of `main()`
2. Process logger via `make_process_logger("train_stage1", LOG_DIR_PROCESS)`
3. `run_id = make_run_id()`
4. Dataset loaded from `TARGET_MAP[args.target]`; keys validated
5. `TensorDataset` + `Subset` built from persisted `train_idx / val_idx / test_idx`
6. Three `DataLoader`s: train (shuffle=True, `seed_worker`, seeded generator), val and test (shuffle=False); all `num_workers=4, pin_memory=True`
7. Device auto-detected (`cuda` if available, else `cpu`)
8. `ProteomicsEncoder` and `ProteomicsAuxHead` instantiated and moved to device
9. `Adam` optimiser over both models with `WEIGHT_DECAY`
10. `ProteomicsStage1Loss(lambda_sparse=args.lambda_sparse)`
11. Experiment JSON written to `logs/experiments/experiment_{run_id}.json` **before** training begins; contains `run_id`, timestamp (UTC), target, device, architecture, hyperparameters, dataset sizes, seeds, param counts, checkpoint path, CSV path, log path
12. CSV created with header only (`epoch, train_loss, val_loss, val_r2, val_mae, val_pearson_r, sp_l1_norm`) — written using `csv.writer` directly so no stub data row is produced
13. Training loop (epoch 1 … N):
    - Train phase: accumulates `l_yp` and `sp_l1_norm` per batch, averages
    - Validation phase: **pooled** (all predictions concatenated before any metric computed) — fixes biased R² issue T6
    - Metrics: `val_loss` (MSE), `val_r2` (1 − SSE/SST), `val_mae`, `val_pearson_r`, `sp_l1_norm`
    - `append_csv_row` called once per epoch
    - Best-validation checkpoint: `stage1_{target}_{run_id}_best.pt`
    - Last checkpoint: `stage1_{target}_{run_id}_last.pt` (overwritten every epoch)
14. Post-training: best checkpoint re-loaded; test set evaluated (pooled); test metrics saved to `logs/experiments/test_metrics_{target}_{run_id}.json`; skipped if `--no-test-eval`
15. Final INFO log; logger closed in `finally` block
16. Full `try/except` with `logger.error(traceback.format_exc())` and re-raise

**CLI arguments:**
```
--target       {age, fluid_iq}   required
--epochs       INT               default EPOCHS (30)
--batch-size   INT               default BATCH_SIZE (64)
--lr           FLOAT             default LR_STAGE1 (1e-3)
--lambda-sparse FLOAT            default LAMBDA_SPARSE (1e-4)
--seed         INT               default GLOBAL_SEED (42)
--no-test-eval FLAG              skip test eval if passed
```

**Checkpoint structure:**
```json
{
  "stage": "stage1",
  "run_id": "...",
  "epoch": N,
  "encoder_state_dict": {...},
  "aux_head_state_dict": {...},
  "optimizer_state_dict": {...},
  "config": { "lr", "batch_size", "epochs", "weight_decay",
               "lambda_sparse", "lambda_aux", "target",
               "global_seed", "split_seed" },
  "val_metrics": { "val_loss", "val_r2", "val_mae",
                   "val_pearson_r", "sp_l1_norm", "epoch" }
}
```

---

### Step 2.12 — `TCG-Net/proteomics/training/cv_stage1.py`
K=5 cross-validation runner operating on the 85% train+val pool only.

**Key behaviours (in execution order):**
1. `set_global_seeds(args.seed)` — first line of `main()`
2. Process logger via `make_process_logger("cv_stage1", LOG_DIR_PROCESS)`
3. `parent_run_id = make_run_id()` — shared across all folds
4. Dataset loaded; `train_idx` and `val_idx` concatenated into `trainval_idx`; `test_idx` extracted but never used during training
5. `KFold(n_splits=args.folds, shuffle=True, random_state=CV_SEED)` applied to `trainval_idx`
6. All fold index arrays saved to `data/splits/cv_{target}_{K}fold_seed{CV_SEED}_{parent_run_id}.npz` before any fold trains
7. For each fold k:
    - `fold_run_id = f"{parent_run_id}_fold{k}"`
    - `set_global_seeds(args.seed + k)` for deterministic per-fold weight init
    - Fresh `ProteomicsEncoder`, `ProteomicsAuxHead`, `Adam`, `ProteomicsStage1Loss`
    - Fold-level experiment JSON written to `logs/experiments/experiment_{fold_run_id}.json` (includes `cv_parent_run_id`, `fold_index`, `fold_train_size`, `fold_val_size`)
    - Fold-level CSV with same header as train_stage1
    - Same per-epoch train/val loop as train_stage1 (pooled validation)
    - Best checkpoint: `stage1_{target}_{fold_run_id}_best.pt`
    - Last checkpoint: `stage1_{target}_{fold_run_id}_last.pt`
    - Best val_metrics collected for summary
8. Summary CSV: `logs/training/cv_summary_{target}_{parent_run_id}.csv` — one row per fold + `mean±std` row
9. Summary JSON: `logs/experiments/cv_summary_{target}_{parent_run_id}.json` — contains `per_fold_metrics`, `summary_mean`, `summary_std`
10. Final INFO log; logger closed in `finally` block

**Import note:** `cv_stage1.py` imports `evaluate`, `save_checkpoint`, `CSV_HEADER`, and `TARGET_MAP` from `train_stage1.py` to avoid full duplication. The per-fold epoch loop (`run_one_fold()`) is implemented locally because the fold-specific DataLoader and model setup cannot be cleanly abstracted away; a comment marks the intentional duplication.

**CLI arguments:**
```
--target        {age, fluid_iq}   required
--folds         INT               default CV_FOLDS (5)
--epochs        INT               default EPOCHS (30)
--batch-size    INT               default BATCH_SIZE (64)
--lr            FLOAT             default LR_STAGE1 (1e-3)
--lambda-sparse FLOAT             default LAMBDA_SPARSE (1e-4)
--seed          INT               default GLOBAL_SEED (42)
```

---

### Step 2.13 — `TCG-Net/proteomics/analysis/protein_importance.py`
Per-protein importance via gradient × input attribution.

**Key behaviours (in execution order):**
1. `set_global_seeds(args.seed)` — first line of `main()`
2. Process logger via `make_process_logger("protein_importance", LOG_DIR_PROCESS)`
3. Checkpoint loaded from `--checkpoint` path; keys `encoder_state_dict`, `aux_head_state_dict`, `run_id`, `config`, `val_metrics` verified
4. `ProteomicsEncoder` and `ProteomicsAuxHead` instantiated; state dicts loaded; both set to `eval()` — eval mode maintained throughout (this is inference, not training)
5. Full dataset loaded (`train + val + test` combined) for post-hoc analysis
6. Device auto-detected; models moved to device
7. Attribution computation — batched loop over all participants:
    - `x_batch.requires_grad_(True)`
    - Forward: `z_p, s_p = encoder(x_batch)` → `y_p = aux_head(z_p)`
    - `y_p.sum().backward()`
    - `grad_input = (x_batch.grad * x_batch).abs()` — shape `(B, 2920)`
    - Accumulate `attribution_sum` (shape 2920) and `sp_sum` (shape 1024)
    - After all batches: `attribution_mean = attribution_sum / n_participants`, `sp_mean = sp_sum / n_participants`
8. DataFrame `[Protein, Importance]` built from feature names + attribution; sorted descending; saved to `analysis_outputs/protein_importance_{target}_{run_id}.csv`
9. Top 5 protein names (not values) logged to process log
10. `sp_mean` saved as numpy array to `analysis_outputs/sp_mean_{target}_{run_id}.npy`; shape `(1024,)` confirmed
11. Process summary JSON saved to `logs/experiments/importance_{target}_{run_id}.json`; contains `top_10_proteins` by name only (no importance values — data protection)
12. Final INFO log; logger closed in `finally` block

**CLI arguments:**
```
--target      {age, fluid_iq}   required
--checkpoint  PATH              required; path to Stage 1 best .pt file
--batch-size  INT               default BATCH_SIZE (64)
--seed        INT               default GLOBAL_SEED (42)
```

Also created: `TCG-Net/proteomics/analysis/__init__.py` (empty package marker, required for imports to work).

---

## Smoke Run Status

The smoke run was **not executed** in this session. The dataset files (`proteomics_tensor.pt` and `dataset_age.pt`) do not yet exist on disk — they require running `preprocess.py` and `build_dataset.py` against the real UKB data, which takes significant time due to KNN imputation.

All three scripts **compiled cleanly** under the `tcg_net` conda environment via `py_compile`:
```
train_stage1.py    OK
cv_stage1.py       OK
protein_importance.py OK
```

### To run the smoke run manually (in order):

```bash
# Step 1 — Preprocess (slow: KNN imputation on ~52 k participants × 2920 proteins)
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/data/preprocess.py

# Step 2 — Build age dataset
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/data/build_dataset.py --target age

# Step 3 — Smoke run (2 epochs, skip test eval)
conda run --no-capture-output -n tcg_net \
  python TCG-Net/proteomics/training/train_stage1.py \
  --target age --epochs 2 --no-test-eval
```

### Expected artefacts after smoke run:
| Artefact | Expected content |
|---|---|
| `logs/process/process_train_stage1_*.log` | Timestamped INFO entries for full run |
| `logs/experiments/experiment_*.json` | All required fields present |
| `logs/training/train_log_age_*.csv` | Header + exactly 2 data rows |
| `checkpoints/stage1_age_*_best.pt` | Contains `encoder_state_dict`, `aux_head_state_dict`, `config`, `val_metrics` |
| `checkpoints/stage1_age_*_last.pt` | Same structure |

---

## Design Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| CSV header write | `csv.writer` directly before loop | `append_csv_row` always writes a data row; using it for header-only would produce a stub empty row |
| Code sharing between cv/train | Import `evaluate`, `save_checkpoint`, `CSV_HEADER` from `train_stage1.py` | Reduces duplication for pure helpers; fold loop itself stays local for clarity |
| Pooled validation metrics | All predictions concatenated before computing R², MAE, Pearson r | Fixes biased-R² issue T6 identified in rebuild plan |
| `eval()` during attribution | Maintained throughout despite `.backward()` | This is inference; eval mode disables dropout/BN training behaviour, which is correct for attribution |
| Raw importance values | Never logged or printed | Data protection; only protein names appear in logs and JSONs |
| `set_global_seeds(args.seed + k)` per fold | Different but deterministic seed per fold | Ensures each fold has distinct but reproducible weight initialisation |

---

## Data Protection Compliance

- No file inside `uk_BIOBANK_DATA/` was opened, read, or modified during this session.
- No raw data values from `uk_BIOBANK_DATA/` were logged, printed, or exposed.
- All paths to `uk_BIOBANK_DATA/` are accessed only via `config.py` constants (`RAW_PROTEIN_CSV`, `HOME_ROOT`); nothing is hardcoded in the new scripts.
- `protein_importance.py` logs only protein identifiers (column names), never importance score values.

---

## Next Steps

1. Run the three-command sequence above to complete the smoke run.
2. Verify the five expected artefacts (content check listed in table above).
3. Run `protein_importance.py` on the best checkpoint from the smoke run:
   ```bash
   conda run --no-capture-output -n tcg_net \
     python TCG-Net/proteomics/analysis/protein_importance.py \
     --target age \
     --checkpoint TCG-Net/proteomics/checkpoints/stage1_age_<run_id>_best.pt
   ```
4. Optionally run CV (Step 2.12) on age and fluid_iq.
5. Proceed to rebuild plan Step 2.14 (README and docstring additions) in the next session.
