# Proteomics Encoder — Rebuild Plan

**Scope:** Proteomics encoder only. The fMRI and sMRI branches are out of scope.
**Basis:** Findings from `proteomics_encoder_report.md` and subsequent review (Additional Issues 1–4).
**Purpose:** This document is a planning reference for a future implementation session. No code changes are made in this session.

---

## 1. Consolidated Issue List

### 1.1 Architecture Issues

- **A1 — Missing 1024-d Stage 1 block.** Spec requires `Linear(2920, 1024) → BN1d(1024) → GELU → Dropout(0.3)` as the first block; current model jumps straight to `Linear(2920, 512)`.
- **A2 — SE-Block on the wrong tensor.** Spec applies SE to the 1024-d hidden representation; current code applies SE to the raw 2920-d input.
- **A3 — SE-Block dimensions wrong.** Spec is `W1: 1024 → 256` and `W2: 256 → 1024`; current code is `2920 → 182 → 2920` (default `reduction=16`).
- **A4 — SE-Block inner activation wrong.** Spec uses GELU between W1 and W2; current code uses `ReLU(inplace=True)`.
- **A5 — Encoder body activations wrong.** Spec uses GELU at every stage; current code uses ReLU at every stage.
- **A6 — One Linear stage collapsed.** Spec defines four Linear stages (2920→1024→512→256→128); current code defines only three (2920→512→256→128).
- **A7 — Missing Dropout(0.3) after stage 2.** Spec places Dropout(0.3) after the `Linear(1024, 512) → BN(512) → GELU` block; absent in current code.
- **A8 — Missing Dropout(0.2) before final Linear.** Spec places Dropout(0.2) after the `Linear(512, 256) → BN(256) → GELU` block; absent in current code.
- **A9 — `s_p` not exposed via `forward`.** Interpretability code reaches into `model.se_block.fc(x)` directly. There is no contract for retrieving `s_p` cleanly.
- **A10 — Regression head hardwired into encoder forward.** `regression_head = Linear(128, 1)` is inside `ProteomicsBranch`. The encoder cannot be reused independently for downstream Shapley fusion or `y_fp` heads while this coupling stands.

### 1.2 Data Pipeline Issues

- **D1 — Z-score leakage.** `StandardScaler.fit_transform(X)` is fit on the entire dataset before the train/val split.
- **D2 — KNN imputation leakage.** Both `process_proteomics.py` (CPU `sklearn.KNNImputer`) and `process_proteomics_gpu.py` (custom batched search) impute over all participants before splitting.
- **D3 — Covariate residualization not implemented.** Source CSV is `UKB_Protein_BL_withCov.csv` (covariates available), but only raw z-score is implemented. Spec permits `z-score OR covariate-residualized + standardised`.
- **D4 — Inconsistent output paths.** `process_proteomics.py` writes to `~/uk_BIOBANK_DATA/protein/proteomics_tensor.pt`; `process_proteomics_gpu.py` writes to `~/research/proteomics_tensor.pt`. Downstream scripts expect the latter.
- **D5 — No persisted preprocessing artefacts.** Fitted scaler mean/std and imputer state are not saved, so the same transform cannot be reapplied to a held-out test split or to Stage 2 input.
- **D6 — No saved split indices.** Train/val splits are produced ad hoc inside training scripts via `random_split` with no seed and never persisted.
- **D7 — Two redundant preprocessing entry points.** CPU and GPU variants of the same pipeline coexist with diverging behaviour and output locations.
- **D8 — No held-out test set.** The pipeline produces only `dataset_age.pt` / `dataset_final_fluid_iq.pt`; there is no separate test split.

### 1.3 Training Protocol Issues

- **T1 — Stage 1 learning rate wrong (Additional Issue 1).** Spec assigns `lr = 1e-3` to Stage 1 single-modality pretraining; current scripts use `lr = 5e-4` (the Stage 2 value).
- **T2 — No Stage 1 / Stage 2 separation (Additional Issue 1).** Checkpoints save the entire `state_dict` (encoder + SE block + regression head) under one key. There is no clean encoder-only export that can be loaded into TCG-Net Stage 2.
- **T3 — No held-out test set, no CV.** Only a single 80/20 random split exists; no test split, no k-fold CV.
- **T4 — `training_loop_final.py` mislabelled.** Loads `dataset_age.pt` but prints "Fluid Intelligence" messages; saves to `best_model_iq.pth` while training the age model.
- **T5 — `anaylze_iq_training_loop.py` overfits by design.** Trains on the full IQ dataset with no validation split, by explicit author intent.
- **T6 — Biased R² computation.** `calculate_r2` is computed per batch and averaged, biasing the result; should be computed over pooled validation predictions.
- **T7 — No early stopping.** Fixed 30-epoch run with best-model checkpoint by validation MSE only.
- **T8 — Best-model selection contaminated.** Validation loss is computed on data that was already seen by the global StandardScaler/KNN — see D1, D2.
- **T9 — Filename typo: `anaylze_iq_training_loop.py`.** Should be `analyze_iq_training_loop.py`, and per the rebuild this script's purpose is replaced by a proper IQ training loop.

### 1.4 Loss Function Issues

- **L1 — Auxiliary loss scaffolding missing (Additional Issue 2).** TCG-Net total loss includes `L_aux = MSE(y, y_p) + MSE(y, y_f) + MSE(y, y_fp)` with `λ_aux = 0.3`. The current `nn.MSELoss()` is conceptually `MSE(y, y_p)` but is not labelled or structured as the `y_p` component, and there are no hooks for `y_f` / `y_fp`.
- **L2 — Sparsity regularisation on `s_p` missing (Additional Issue 3).** Spec defines `L_sparse = ||w||_1 + ||s_p||_1 + ||α_f||_1` with `λ_sparse = 1e-4`. The proteomics encoder must contribute the `||s_p||_1` term; currently `s_p` is not exposed and no L1 is applied.
- **L3 — `||s_p||_1` not logged.** No diagnostic of SE gate sparsity is recorded during training.
- **L4 — No combined Stage 1 loss object.** Loss is constructed inline as a single `criterion(prediction, batch_Y)` call; there is no `ProteomicsStage1Loss` (or similar) that bundles `y_p` MSE, `λ_sparse · ||s_p||_1`, and produces logging-ready scalars.

### 1.5 Logging and Metric Persistence Issues

- **LG1 — No per-epoch training log.** Metrics are only printed; no CSV is produced.
- **LG2 — No experiment configuration JSON.** The exact hyperparameters, architecture, dataset sizes, split seed, and checkpoint path of any past run cannot be recovered from the repository.
- **LG3 — No preprocessing log.** Imputation counts, fit-split, normalisation method, and output statistics are not persisted.
- **LG4 — No process audit log.** No timestamped human-readable record of what each script did and when.
- **LG5 — No held-out test metrics.** A direct consequence of D8.
- **LG6 — Metric set incomplete.** Spec requires `train_loss, val_loss, val_r2, val_mae, val_pearson_r, sp_l1_norm` per epoch; current code computes only train MSE, val MSE, biased val R².
- **LG7 — Best-checkpoint metadata not embedded.** Saved `.pth` files contain only the `state_dict`; no mapping back to a training run, configuration, or epoch.

### 1.6 Reproducibility Issues

- **R1 — No global random seed.** No call to `torch.manual_seed`, `numpy.random.seed`, `random.seed`, or `torch.cuda.manual_seed_all`.
- **R2 — `random_split` not seeded.** Used without `generator=torch.Generator().manual_seed(...)`, so the train/val partition changes every run.
- **R3 — DataLoader workers not seeded.** No `worker_init_fn` is provided.
- **R4 — Split indices not persisted.** Even with a seed fixed, the actual indices are not written to disk for downstream reuse (e.g., Stage 2).
- **R5 — Determinism flags not set.** `torch.backends.cudnn.deterministic` and `torch.backends.cudnn.benchmark` are not configured.
- **R6 — No K-fold scaffolding.** When CV is added it must use a fixed fold seed and persist fold indices; currently nothing exists to extend.

### 1.7 Code Quality and Structure Issues

- **C1 — Hardcoded path constants.** Every script defines its own `os.path.expanduser` constants. There is no single config source.
- **C2 — Filename typo.** `anaylze_iq_training_loop.py` should be removed/renamed.
- **C3 — Empty `Pre_Processing/` directory.** Dead artefact.
- **C4 — `__pycache__/` committed.** Should not live next to source.
- **C5 — Duplicate preprocessing scripts.** CPU and GPU variants without a unified entry point with a `--device` toggle.
- **C6 — No README / no docstrings on the encoder contract.** Inputs, outputs, and downstream integration points are undocumented.
- **C7 — `find_targets.py` and `find_age.py` are one-off discovery scripts.** Once the field IDs are known they should be retired or moved to a `scripts/discovery/` folder.
- **C8 — No shape / smoke tests.** There is nothing that asserts forward-pass shapes match the spec (`(B, 2920) → z_p (B, 128), s_p (B, 1024)`).
- **C9 — Monolithic training scripts.** Data loading, splitting, model construction, logging, and the train loop all live in one `main()`. Hard to swap pieces (e.g., target, fold, loss term).
- **C10 — Inconsistent label/print messaging in `training_loop_final.py`.** Behaviourally innocuous but actively misleading for any reader.

---

## 2. Detailed Rebuild Plan

The plan is ordered so that earlier entries do not depend on later ones. Build infrastructure first (config, seeds, logging helpers), then preprocessing, then model, then loss, then training, then evaluation/analysis, then quality.

---

### 2.1 — Add a configuration module
**Problem:** Every script hardcodes its own paths. There is no single source of truth for input files, output files, model hyperparameters, λ values, or seeds.
**Fix:** Create `research/proteomics/config.py` exposing:
- Path constants: `DATA_DIR`, `RAW_PROTEIN_CSV`, `PROCESSED_TENSOR_PATH`, `DATASET_AGE_PATH`, `DATASET_IQ_PATH`, `CHECKPOINT_DIR`, `LOG_DIR_TRAINING`, `LOG_DIR_EXPERIMENT`, `LOG_DIR_PREPROCESS`, `LOG_DIR_PROCESS`.
- Default hyperparameters (Stage 1): `LR_STAGE1 = 1e-3`, `BATCH_SIZE = 64`, `EPOCHS = 30`, `WEIGHT_DECAY = 1e-5`, `LAMBDA_SPARSE = 1e-4`, `LAMBDA_AUX = 0.3` (kept for forward-compat reference even though Stage 1 uses only the `y_p` component), `DROPOUT_STAGE1 = 0.3`, `DROPOUT_STAGE2 = 0.3`, `DROPOUT_STAGE3 = 0.2`.
- Reproducibility: `GLOBAL_SEED = 42`, `SPLIT_SEED = 1234`, `CV_SEED = 7`.
- Split fractions: `TRAIN_FRAC = 0.7`, `VAL_FRAC = 0.15`, `TEST_FRAC = 0.15`.
**Files affected:** New `research/proteomics/config.py`. (No modifications to other files in this entry; they will read from it in subsequent entries.)
**Dependencies:** None.
**Expected outcome:** A single, importable configuration object that downstream scripts can read.

---

### 2.2 — Add a reproducibility utility (Additional Issue 4)
**Problem:** No fixed seeds anywhere. Splits, weight init, DataLoader workers, and CV folds are all non-deterministic.
**Fix:** Create `research/proteomics/utils/seed.py` with:
- `set_global_seeds(seed: int) -> None` calling `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and setting `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- `make_split_generator(seed: int) -> torch.Generator` returning a seeded `torch.Generator()` to pass into `random_split`.
- `seed_worker(worker_id: int) -> None` to be passed to `DataLoader(worker_init_fn=...)`, setting numpy/random seeds derived from `torch.initial_seed()`.
Every entry script (preprocessing, dataset construction, training, analysis) must call `set_global_seeds(GLOBAL_SEED)` as its first line of `main()`.
**Files affected:** New `research/proteomics/utils/seed.py`; later integrated into all entry scripts.
**Dependencies:** 2.1.
**Expected outcome:** Re-running any pipeline step from a clean state produces bitwise-identical (or numerically deterministic, modulo cuDNN reductions) outputs.

---

### 2.3 — Add a logging utility for process `.log` files
**Problem:** No human-readable audit trail exists. All scripts only `print()`.
**Fix:** Create `research/proteomics/utils/logging_utils.py` with:
- `make_process_logger(script_name: str, log_dir: Path) -> logging.Logger` that:
  - Builds the path `logs/process/process_[script_name]_[YYYYmmdd_HHMMSS].log`.
  - Configures a `logging.Logger` with a `FileHandler` and a `StreamHandler` (stdout), both using formatter `"[%(asctime)s] [%(levelname)s] %(message)s"` with `datefmt="%Y-%m-%d %H:%M:%S"`.
  - Returns the logger and the resolved file path.
- `make_run_id() -> str` returning `f"{YYYYmmdd_HHMMSS}_{uuid4().hex[:8]}"`.
- `dump_json(path: Path, payload: dict) -> None` writing pretty-printed JSON with `indent=2, sort_keys=False`.
- `append_csv_row(path: Path, row: dict, header: list[str]) -> None` opening in append mode, writing the header only if the file is being created.
**Files affected:** New `research/proteomics/utils/logging_utils.py`.
**Dependencies:** 2.1.
**Expected outcome:** A reusable helper layer for the four log types specified in Section 4.

---

### 2.4 — Rebuild preprocessing into a single split-aware script
**Problem:** Two redundant scripts exist (`process_proteomics.py`, `process_proteomics_gpu.py`); both fit `KNNImputer` and `StandardScaler` on the full dataset (train/val/test leakage); paths diverge; no preprocessing log is written; covariate residualization is not implemented (D1, D2, D3, D4, D5, D7).
**Fix:** Replace both scripts with `research/proteomics/data/preprocess.py`. Behaviour:
1. Load `UKB_Protein_BL_withCov.csv`, drop the seven metadata columns, keep `eid`.
2. Compute split indices first: shuffle EIDs with `SPLIT_SEED`, then 70/15/15 train/val/test. Save the index arrays to `research/data/splits/split_[SPLIT_SEED].npz` (keys `train_idx`, `val_idx`, `test_idx`, plus the `eid` order they index into).
3. Fit `KNNImputer(n_neighbors=5)` on training rows only (`X[train_idx]`); transform train, val, test separately. Optionally retain GPU batched imputation as a `--device gpu` flag, but with the same fit-on-train constraint. Persist the imputer (`joblib.dump`) to `research/data/artefacts/knn_imputer_[SPLIT_SEED].joblib`.
4. Optional covariate residualization (`--residualize` flag): fit a `LinearRegression` per-protein on the train split using covariates {age (21003-0.0), sex, batch one-hot, plate one-hot, fasting time, assay-warning flag}; subtract predicted values to obtain residuals; persist the regressor coefficients per protein to `research/data/artefacts/covariate_resid_[SPLIT_SEED].npz`. (If `--residualize` is off, skip this step.)
5. Fit `StandardScaler()` on training rows only; transform all three splits. Persist scaler mean/std arrays to `research/data/artefacts/scaler_[SPLIT_SEED].npz` and the sklearn object to `research/data/artefacts/scaler_[SPLIT_SEED].joblib`.
6. Save the final tensor as a dict `{x, eid, features, train_idx, val_idx, test_idx}` to `research/data/proteomics_tensor.pt`.
7. Write `logs/preprocessing/preprocess_log_[timestamp].json` (see §4.3) and `logs/process/process_preprocess_[timestamp].log` (see §4.4).
**Files affected:** New `research/proteomics/data/preprocess.py`. Delete: `research/process_proteomics.py`, `research/process_proteomics_gpu.py`.
**Dependencies:** 2.1, 2.2, 2.3.
**Expected outcome:** A single canonical preprocessing entry point, no fit-on-full-data leakage, all artefacts persisted, audit logs written.

---

### 2.5 — Rebuild target-linking scripts as one parameterised module
**Problem:** `create_age_dataset.py` and `create_final_dataset.py` are near-duplicates that only differ in target column and CSV path (D6 supporting structure missing).
**Fix:** Replace both with `research/proteomics/data/build_dataset.py`, taking `--target {age, fluid_iq}` as a CLI argument:
- `age` → `phesantallphenos_05.csv`, column `21003-0.0`, output `research/data/dataset_age.pt`.
- `fluid_iq` → `phesantallphenos_04.csv`, column `20016-0.0`, output `research/data/dataset_fluid_iq.pt`.
- Load the processed tensor produced by 2.4 (which now carries `train_idx / val_idx / test_idx`).
- Intersect `eid`s with the target CSV (preserving the global order from the processed tensor).
- Re-emit the split indices on the *intersected* subset (so downstream code works with positional indices into `dataset_age.pt`/`dataset_fluid_iq.pt`, not the parent tensor). Persist to `research/data/splits/split_[target]_[SPLIT_SEED].npz`.
- Write `{x, y, eid, features, target_name, train_idx, val_idx, test_idx}` to the output `.pt`.
- Write a process `.log` and a small JSON summary `logs/preprocessing/dataset_build_[target]_[timestamp].json` capturing the row count before/after intersection, target column mean/std, and split sizes.
**Files affected:** New `research/proteomics/data/build_dataset.py`. Delete: `research/create_age_dataset.py`, `research/create_final_dataset.py`.
**Dependencies:** 2.4.
**Expected outcome:** One target-linking script, two `(x, y, splits)` artefacts.

---

### 2.6 — Rebuild the SE-Block to spec
**Problem:** Current `SEBlock(channel, reduction=16)` uses `2920 → 182 → 2920` with ReLU; spec is `1024 → 256 → 1024` with GELU + sigmoid (A2, A3, A4).
**Fix:** Reimplement `SEBlock` in `research/proteomics/models/se_block.py`:
```
class SEBlock(nn.Module):
    def __init__(self, channel=1024, hidden=256):
        super().__init__()
        self.W1 = nn.Linear(channel, hidden, bias=False)
        self.act = nn.GELU()
        self.W2 = nn.Linear(hidden, channel, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, h_p):
        s_p = self.gate(self.W2(self.act(self.W1(h_p))))   # (B, 1024)
        return h_p * s_p, s_p
```
The forward must return both the gated tensor and `s_p` (so the encoder can pass `s_p` upward for the sparsity loss and interpretability). Default args match the spec (`channel=1024, hidden=256`); no `reduction` parameter.
**Files affected:** New `research/proteomics/models/se_block.py`. (Old `research/model_proteomics.py` to be deleted in 2.7.)
**Dependencies:** 2.1.
**Expected outcome:** Spec-compliant SE-Block that explicitly returns `s_p`.

---

### 2.7 — Rebuild the proteomics encoder to spec
**Problem:** Current `ProteomicsBranch` is a 3-Linear MLP with SE on the raw input; missing 1024-d Stage 1; SE in wrong place; ReLU instead of GELU; missing dropouts; head fused into encoder forward (A1, A2, A5, A6, A7, A8, A9, A10).
**Fix:** Implement `ProteomicsEncoder` in `research/proteomics/models/encoder.py`:
```
class ProteomicsEncoder(nn.Module):
    def __init__(self, input_dim=2920, hidden_dim=1024, embed_dim=128,
                 dropout1=0.3, dropout2=0.3, dropout3=0.2):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout1),
        )
        self.se_block = SEBlock(channel=hidden_dim, hidden=hidden_dim // 4)  # 1024 -> 256
        self.stage2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),     # 1024 -> 512
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout2),
        )
        self.stage3 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 512 -> 256
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout3),
        )
        self.proj = nn.Linear(hidden_dim // 4, embed_dim)  # 256 -> 128

    def forward(self, x):
        h = self.stage1(x)
        h_gated, s_p = self.se_block(h)
        h = self.stage2(h_gated)
        h = self.stage3(h)
        z_p = self.proj(h)
        return z_p, s_p
```
Notes: the encoder no longer carries a regression head. Forward returns `(z_p, s_p)`. The expected shapes are `x: (B, 2920)`, `z_p: (B, 128)`, `s_p: (B, 1024)`.
**Files affected:** New `research/proteomics/models/encoder.py`. Delete `research/model_proteomics.py`.
**Dependencies:** 2.6.
**Expected outcome:** Encoder matches the TCG-Net spec exactly and exposes `s_p` cleanly.

---

### 2.8 — Add a separate auxiliary head module
**Problem:** Current `regression_head` is fused inside the encoder forward, blocking decoupled use in TCG-Net Stage 2 (A10).
**Fix:** Implement `ProteomicsAuxHead` in `research/proteomics/models/aux_head.py`:
```
class ProteomicsAuxHead(nn.Module):
    def __init__(self, embed_dim=128, out_dim=1):
        super().__init__()
        self.fc = nn.Linear(embed_dim, out_dim)
    def forward(self, z_p):
        return self.fc(z_p)
```
The Stage 1 training script will instantiate `encoder` and `aux_head` separately and feed `z_p → aux_head → y_p`.
**Files affected:** New `research/proteomics/models/aux_head.py`.
**Dependencies:** 2.7.
**Expected outcome:** Encoder and head are independently constructable; checkpoints can save them under separate keys.

---

### 2.9 — Add a forward-shape smoke test
**Problem:** Nothing asserts that the rebuilt model produces the expected shapes (C8).
**Fix:** Create `research/proteomics/tests/test_shapes.py` (a plain pytest-style file; can also run as `python -m research.proteomics.tests.test_shapes`):
- Build `ProteomicsEncoder()` with spec defaults.
- Pass a random tensor of shape `(8, 2920)` through `forward`.
- Assert `z_p.shape == (8, 128)`, `s_p.shape == (8, 1024)`.
- Build `ProteomicsAuxHead()` and assert `aux_head(z_p).shape == (8, 1)`.
- Verify a backward pass runs without error.
**Files affected:** New `research/proteomics/tests/test_shapes.py`.
**Dependencies:** 2.7, 2.8.
**Expected outcome:** A one-line invocation confirms the architecture is wired correctly before any training is launched.

---

### 2.10 — Build the Stage 1 loss module (Additional Issues 2 and 3)
**Problem:** Loss is a single inline `nn.MSELoss()`; no `y_p` labelling, no scaffold for `y_f` / `y_fp`, no L1 on `s_p`, no diagnostics (L1, L2, L3, L4).
**Fix:** Create `research/proteomics/training/losses.py` with `ProteomicsStage1Loss`:
```
class ProteomicsStage1Loss(nn.Module):
    """
    Stage 1 of TCG-Net training. Implements the y_p component of L_aux,
    plus the ||s_p||_1 component of L_sparse. y_f and y_fp are deferred
    to Stage 2; this module exposes the slots so the Stage 2 loss can
    extend it without re-plumbing.
    """
    def __init__(self, lambda_aux=0.3, lambda_sparse=1e-4):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_sparse = lambda_sparse
        self.mse = nn.MSELoss()

    def forward(self, y_true, y_p, s_p):
        l_yp = self.mse(y_p, y_true)              # y_p component of L_aux
        l_sparse_sp = s_p.abs().mean(dim=1).mean()  # mean over batch of L1-per-sample
        # Stage 1 total = y_p MSE + λ_sparse * ||s_p||_1.
        # λ_aux is logged but not multiplied here, since y_f and y_fp are absent.
        # In Stage 2, the joint loss object will multiply L_aux (= l_yp + l_yf + l_yfp)
        # by λ_aux instead.
        total = l_yp + self.lambda_sparse * l_sparse_sp
        components = {
            "loss_total": total.detach(),
            "loss_yp": l_yp.detach(),
            "loss_sparse_sp": l_sparse_sp.detach(),
            "sp_l1_norm": l_sparse_sp.detach(),  # logged diagnostic
        }
        return total, components
```
Notes:
- `||s_p||_1` is reported as the per-sample L1 mean averaged across the batch. Document this convention in the docstring so future work matches.
- In Stage 2 the joint loss should reuse `l_yp` and `l_sparse_sp` directly; the Stage 1 module's API (`y_true, y_p, s_p`) is a strict subset of the Stage 2 API (`y_true, y_p, y_f, y_fp, s_p, alpha_f, w`), so refactoring later is additive.
**Files affected:** New `research/proteomics/training/losses.py`.
**Dependencies:** 2.7.
**Expected outcome:** A loss module that explicitly labels its components, exposes diagnostics, and is forward-compatible with Stage 2.

---

### 2.11 — Build the Stage 1 training script (Additional Issue 1)
**Problem:** Current `training_loop_final.py` uses `lr=5e-4`, mislabels its target, has no test set, no fixed seed, no proper logging, no L1 on `s_p`, computes biased R², and saves a monolithic checkpoint that cannot be loaded into Stage 2 (T1, T2, T3, T4, T6, T7, T8, R1–R5, LG1, LG2, LG6, LG7).
**Fix:** Implement `research/proteomics/training/train_stage1.py` with the following structure:

CLI: `--target {age, fluid_iq}` (required), `--epochs N` (default from config), `--batch-size N`, `--lr` (default `1e-3`), `--lambda-sparse` (default `1e-4`), `--seed N`, `--cv-fold N` (optional), `--no-test-eval` (default false).

Pseudocode:
```
1.  set_global_seeds(args.seed)
2.  logger, log_path = make_process_logger("train_stage1", LOG_DIR_PROCESS)
3.  run_id = make_run_id()
4.  Load dataset_[target].pt → x, y, train_idx, val_idx, test_idx, features
5.  Build TensorDataset; build train/val/test Subsets using the persisted indices
6.  Build DataLoaders (worker_init_fn=seed_worker, generator=make_split_generator)
7.  encoder = ProteomicsEncoder(...)
8.  aux_head = ProteomicsAuxHead(embed_dim=128, out_dim=1)
9.  optimizer = Adam([{params: encoder.parameters()}, {params: aux_head.parameters()}],
                     lr=1e-3, weight_decay=1e-5)
10. loss_fn = ProteomicsStage1Loss(lambda_aux=0.3, lambda_sparse=1e-4)
11. Write experiment JSON (see §4.2) BEFORE training begins
12. Open training CSV and write header
13. For epoch in 1..EPOCHS:
        train_loss = train_one_epoch(...)
        val_metrics = evaluate(val_loader)  # MSE, R2 (pooled), MAE, Pearson r,
                                            # mean sp_l1_norm
        append CSV row {epoch, train_loss, val_loss, val_r2, val_mae,
                        val_pearson_r, sp_l1_norm}
        logger.info(...)
        If val_loss is best so far:
            Save checkpoint (see structure below)
14. Optionally evaluate test_loader once at end; log to process .log; do NOT touch
    training CSV (the CSV is per-epoch only).
```

Checkpoint structure (Additional Issue 1):
```
torch.save({
    "stage": "stage1",
    "run_id": run_id,
    "epoch": best_epoch,
    "encoder_state_dict": encoder.state_dict(),
    "aux_head_state_dict": aux_head.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": { all hyperparameters, seeds, target, lambda_sparse, lambda_aux },
    "val_metrics": { val_loss, val_r2, val_mae, val_pearson_r, sp_l1_norm },
}, CHECKPOINT_DIR / f"stage1_{target}_{run_id}.pt")
```
This separation lets Stage 2 load `encoder_state_dict` directly into the TCG-Net joint model without dragging the aux head along.

Validation metrics must be **pooled** (concatenate predictions and targets across batches, then compute R², MAE, Pearson r once on the full validation set), per Recommendation 5 of the report and to fix T6.

**Files affected:** New `research/proteomics/training/train_stage1.py`. Delete: `research/training_loop_final.py`, `research/training_loop_random_targets.py`, `research/anaylze_iq_training_loop.py`.
**Dependencies:** 2.1–2.10.
**Expected outcome:** A single, properly seeded, properly logged Stage 1 pretraining loop that is target-agnostic, produces CSV + JSON + .log artefacts, and saves an encoder checkpoint loadable into TCG-Net Stage 2.

---

### 2.12 — (Optional but recommended) Add a K-fold cross-validation runner
**Problem:** No CV scaffolding exists (T3, R6).
**Fix:** Implement `research/proteomics/training/cv_stage1.py` that:
- Loads `dataset_[target].pt`.
- Re-splits the **train + val** indices (keeping `test_idx` untouched as a true hold-out) into K folds with `KFold(n_splits=K, shuffle=True, random_state=CV_SEED)`.
- For each fold, persists fold indices to `research/data/splits/cv_[target]_[K]_seed[CV_SEED].npz` and runs the Stage 1 loop (calling into the same training entry point internally).
- Writes one experiment JSON per fold (`run_id` includes fold index) and one CSV per fold.
- After all folds complete, writes `logs/training/cv_summary_[target]_[timestamp].csv` with one row per fold (mean & std of the final-epoch val metrics).
**Files affected:** New `research/proteomics/training/cv_stage1.py`.
**Dependencies:** 2.11.
**Expected outcome:** Reproducible CV with fold-level CSVs and a summary file.

---

### 2.13 — Rebuild the importance analysis script
**Problem:** `analyze_importance.py` and `anaylze_iq_training_loop.py` both reach into `model.se_block.fc(x)` directly and rely on a coupled forward (T5, T9, A9 dependent).
**Fix:** Implement `research/proteomics/analysis/protein_importance.py` that:
- Accepts `--target {age, fluid_iq}` and `--checkpoint PATH`.
- Loads the Stage 1 checkpoint; instantiates `ProteomicsEncoder`; loads `encoder_state_dict`.
- Iterates over the dataset in batches, calling `encoder(x)` to get `(z_p, s_p)`. Note: since the spec moves SE to the 1024-d hidden layer, `s_p` is per-hidden-unit, **not per-protein**. Per-protein importance must be aggregated through the gradient of the loss (or a chosen output) w.r.t. the input — implement a `gradient × input` attribution as the per-protein importance:
  ```
  x.requires_grad_(True)
  z_p, s_p = encoder(x)
  y_p = aux_head(z_p)
  attribution = (x.grad).abs().mean(dim=0)  # (2920,)
  ```
  computed by running `y_p.sum().backward()` and collecting `x.grad`.
- Also compute and save the **mean s_p across the dataset** (a 1024-d vector) as a separate artefact, since this is interpretable as latent-channel importance.
- Save:
  - `research/analysis/protein_importance_[target]_[run_id].csv` (columns: `Protein`, `Importance`).
  - `research/analysis/sp_mean_[target]_[run_id].npy` (the 1024-d gate vector).
  - A process `.log`.
**Files affected:** New `research/proteomics/analysis/protein_importance.py`. Delete: `research/analyze_importance.py`, `research/anaylze_iq_training_loop.py`.
**Dependencies:** 2.7, 2.11.
**Expected outcome:** A single analysis script that uses the public encoder API, no internal-attribute access; per-protein importance is well-defined under the spec architecture.

---

### 2.14 — Document the encoder contract
**Problem:** No README, no docstrings explaining inputs, outputs, or the Stage 1 → Stage 2 handoff (C6).
**Fix:** Add `research/proteomics/README.md` covering:
- Input shape: `(B, 2920)`, z-scored or covariate-residualized + standardised.
- Output shape: `z_p ∈ (B, 128)`, `s_p ∈ (B, 1024)`.
- The Stage 1 checkpoint structure (keys: `stage`, `encoder_state_dict`, `aux_head_state_dict`, `config`, `val_metrics`) and how to load only the encoder for Stage 2: `encoder.load_state_dict(torch.load(path)["encoder_state_dict"])`.
- Locations of saved scaler / imputer artefacts so Stage 2 inference can reapply the same transform.
- The four log types and where they land.
- Where split indices live and why they must be reused across stages.
Also add module-level docstrings on `encoder.py`, `se_block.py`, `aux_head.py`, and `losses.py` summarising the same.
**Files affected:** New `research/proteomics/README.md`; docstring additions to the four module files above.
**Dependencies:** 2.6, 2.7, 2.8, 2.10, 2.11.
**Expected outcome:** Any future collaborator (TCG-Net Stage 2 author, fMRI/sMRI branch author, fusion author) can integrate without reading the internals.

---

### 2.15 — Retire one-off discovery scripts
**Problem:** `find_targets.py` and `find_age.py` are header-scan utilities that have served their purpose; their results are baked into `build_dataset.py` (C7).
**Fix:** Move both to `research/proteomics/scripts/discovery/` and add a one-line header note in each that they are archived, not part of the production pipeline.
**Files affected:** Move `research/find_targets.py` → `research/proteomics/scripts/discovery/find_targets.py`; move `research/find_age.py` → `research/proteomics/scripts/discovery/find_age.py`.
**Dependencies:** None (independent of model/data changes).
**Expected outcome:** Production code paths contain no dead one-off scripts.

---

### 2.16 — Clean up dead artefacts
**Problem:** Empty `Pre_Processing/` directory, committed `__pycache__/`, mislabelled / stale checkpoints (`best_model_age_clock.pth`, `best_model_iq.pth`, `model_fluid_iq_analysis.pth`) that are not state-dict-compatible with the new architecture, and the legacy CSVs (`protein_importance_age.csv`, `protein_importance_iq.csv`) that were produced from the wrong architecture (C3, C4).
**Fix:**
- Remove the empty `research/Pre_Processing/` directory.
- Remove `research/__pycache__/`. Add `__pycache__/` and `*.pyc` to `.gitignore` (create `.gitignore` if not present).
- Move legacy artefacts to `research/_archive/legacy_v0/`: `best_model_age_clock.pth`, `best_model_iq.pth` (if present), `model_fluid_iq_analysis.pth`, `protein_importance_age.csv`, `protein_importance_iq.csv`, `proteomics_tensor.pt` (the unsplit one), `dataset_age.pt`, `dataset_final_fluid_iq.pt`. They are not deleted (they may be useful for sanity comparison) but they are clearly fenced off from the rebuilt pipeline.
**Files affected:** As listed above.
**Dependencies:** 2.4, 2.5, 2.7, 2.11, 2.13 (so we are not deleting artefacts the new pipeline still needs).
**Expected outcome:** The active code tree contains only spec-aligned code and artefacts.

---

## 3. Recommended File and Script Structure

After the rebuild, the layout under `research/` should be:

```
research/
├── proteomics_encoder_report.md             # untouched (analysis report)
├── proteomics_encoder_rebuild_plan.md       # this document
├── proteomics/
│   ├── README.md                            # encoder contract (2.14)
│   ├── config.py                            # paths + hyperparameters + seeds (2.1)
│   ├── data/
│   │   ├── preprocess.py                    # split-aware preprocessing (2.4)
│   │   └── build_dataset.py                 # target linking, --target {age, fluid_iq} (2.5)
│   ├── models/
│   │   ├── se_block.py                      # SE-Block 1024 → 256 → 1024, GELU (2.6)
│   │   ├── encoder.py                       # ProteomicsEncoder (2.7)
│   │   └── aux_head.py                      # ProteomicsAuxHead(128 → 1) (2.8)
│   ├── training/
│   │   ├── losses.py                        # ProteomicsStage1Loss (2.10)
│   │   ├── train_stage1.py                  # Stage 1 pretraining loop (2.11)
│   │   └── cv_stage1.py                     # K-fold runner (2.12, optional)
│   ├── analysis/
│   │   └── protein_importance.py            # importance analysis (2.13)
│   ├── tests/
│   │   └── test_shapes.py                   # forward-pass shape smoke test (2.9)
│   ├── utils/
│   │   ├── seed.py                          # seed control (2.2)
│   │   └── logging_utils.py                 # process logger, JSON, CSV helpers (2.3)
│   └── scripts/
│       └── discovery/
│           ├── find_targets.py              # archived one-off (2.15)
│           └── find_age.py                  # archived one-off (2.15)
├── data/
│   ├── proteomics_tensor.pt                 # output of preprocess.py
│   ├── dataset_age.pt                       # output of build_dataset.py --target age
│   ├── dataset_fluid_iq.pt                  # output of build_dataset.py --target fluid_iq
│   ├── splits/
│   │   ├── split_[seed].npz                 # global EID-level split
│   │   ├── split_age_[seed].npz             # post-intersection positional indices
│   │   ├── split_fluid_iq_[seed].npz
│   │   └── cv_[target]_[K]_seed[seed].npz   # K-fold indices (if 2.12)
│   └── artefacts/
│       ├── knn_imputer_[seed].joblib
│       ├── scaler_[seed].joblib
│       ├── scaler_[seed].npz                # mean/std arrays for portable load
│       └── covariate_resid_[seed].npz       # only if --residualize was on
├── checkpoints/
│   └── stage1_[target]_[run_id].pt          # encoder + aux_head + config + metrics
├── analysis/
│   ├── protein_importance_[target]_[run_id].csv
│   └── sp_mean_[target]_[run_id].npy
├── logs/
│   ├── preprocessing/
│   │   ├── preprocess_log_[timestamp].json
│   │   └── dataset_build_[target]_[timestamp].json
│   ├── experiments/
│   │   └── experiment_[run_id]_[timestamp].json
│   ├── training/
│   │   ├── train_log_[target]_[timestamp].csv
│   │   └── cv_summary_[target]_[timestamp].csv     # only if 2.12 used
│   └── process/
│       └── process_[script_name]_[timestamp].log
└── _archive/
    └── legacy_v0/                                  # populated by 2.16
        ├── model_proteomics.py
        ├── process_proteomics.py
        ├── process_proteomics_gpu.py
        ├── training_loop_final.py
        ├── training_loop_random_targets.py
        ├── anaylze_iq_training_loop.py
        ├── analyze_importance.py
        ├── create_age_dataset.py
        ├── create_final_dataset.py
        ├── best_model_age_clock.pth
        ├── model_fluid_iq_analysis.pth
        ├── protein_importance_age.csv
        ├── protein_importance_iq.csv
        ├── proteomics_tensor.pt
        ├── dataset_age.pt
        └── dataset_final_fluid_iq.pt
```

### Replacement / consolidation map

| Old file | New location / role |
|---|---|
| `research/model_proteomics.py` | Replaced by `proteomics/models/se_block.py`, `models/encoder.py`, `models/aux_head.py` (2.6, 2.7, 2.8). |
| `research/process_proteomics.py` | Replaced by `proteomics/data/preprocess.py` (2.4). |
| `research/process_proteomics_gpu.py` | Replaced by `proteomics/data/preprocess.py --device gpu` (2.4). |
| `research/create_age_dataset.py` | Replaced by `proteomics/data/build_dataset.py --target age` (2.5). |
| `research/create_final_dataset.py` | Replaced by `proteomics/data/build_dataset.py --target fluid_iq` (2.5). |
| `research/training_loop_final.py` | Replaced by `proteomics/training/train_stage1.py --target age` (2.11). |
| `research/training_loop_random_targets.py` | Removed. Superseded by `proteomics/tests/test_shapes.py` (2.9) for pipeline sanity. |
| `research/anaylze_iq_training_loop.py` | Removed (overfits by design); IQ training is now `proteomics/training/train_stage1.py --target fluid_iq`; importance analysis is `proteomics/analysis/protein_importance.py` (2.11, 2.13). |
| `research/analyze_importance.py` | Replaced by `proteomics/analysis/protein_importance.py` (2.13). |
| `research/find_targets.py`, `find_age.py` | Archived to `proteomics/scripts/discovery/` (2.15). |
| `research/Pre_Processing/` | Removed (empty) (2.16). |
| `research/proteomics_tensor.pt`, `dataset_age.pt`, `dataset_final_fluid_iq.pt` | Replaced by new outputs under `research/data/`; old files moved to `_archive/legacy_v0/` (2.16). |
| `research/best_model_age_clock.pth`, `model_fluid_iq_analysis.pth` | Not state-dict-compatible with the rebuilt architecture; archived to `_archive/legacy_v0/`. New checkpoints land in `research/checkpoints/` under `stage1_[target]_[run_id].pt`. |
| `research/protein_importance_age.csv`, `protein_importance_iq.csv` | Archived; new outputs land in `research/analysis/protein_importance_[target]_[run_id].csv`. |

### Naming conventions

- `[timestamp]` = `YYYYmmdd_HHMMSS` (UTC or local; pick one in `config.py` and stick with it).
- `[run_id]` = `[timestamp]_[8-char hex]` produced by `make_run_id()`.
- `[target]` ∈ `{age, fluid_iq}`.
- `[seed]` = the integer split seed actually used (so multiple split seeds can coexist on disk).
- `[script_name]` = the basename of the entry script without the `.py` extension.
- All checkpoints use the `stage1_[target]_[run_id].pt` convention so that Stage 2 wiring can `glob("stage1_*_*.pt")` deterministically.

---

## 4. Logging Implementation Plan

Each log type is fixed by the requirements. This section specifies which script writes each, when, and how partial runs are handled.

### 4.1 Training Logs (CSV)

- **Responsible scripts:** `proteomics/training/train_stage1.py`, `proteomics/training/cv_stage1.py`.
- **File path:** `research/logs/training/train_log_[target]_[timestamp].csv` (for CV runs, `cv_fold[K]_train_log_[target]_[timestamp].csv` and an aggregated `cv_summary_[target]_[timestamp].csv`).
- **Header (created on first write):** `epoch,train_loss,val_loss,val_r2,val_mae,val_pearson_r,sp_l1_norm`.
- **When opened:** before the training loop starts, the file is *created* (header written) but no per-epoch rows yet exist. The path is recorded in the experiment JSON as `training_log_save_path`.
- **When written:** at the end of each epoch, the script appends one row using `append_csv_row(...)` (open in `"a"` mode, write the row, flush, close). The flush-and-close on every epoch is what guarantees a partially completed run still produces a usable log.
- **`val_loss` semantics:** the validation MSE on **pooled** predictions (not the per-batch mean), to fix issue T6.
- **`val_r2` semantics:** `1 - SSE/SST` over pooled validation predictions and targets.
- **`val_mae` and `val_pearson_r`:** computed once per epoch over pooled validation predictions.
- **`sp_l1_norm`:** the per-sample L1 mean of `s_p` averaged across the validation set, as defined in 2.10.
- **Closing:** the file is implicitly closed after each row (since `append_csv_row` opens, writes, and closes).
- **Partial-run handling:** because every row is committed atomically and the file is opened in append mode, any kill (SIGINT, OOM, system reboot) leaves a CSV that contains exactly the epochs that completed.

### 4.2 Experiment Logs (JSON)

- **Responsible script:** `proteomics/training/train_stage1.py` (and `cv_stage1.py`, one JSON per fold).
- **File path:** `research/logs/experiments/experiment_[run_id]_[timestamp].json`.
- **When written:** immediately after seeds are set and the dataset is loaded but **before the first training epoch begins**. Single write, single close. The file is not updated again during the run; per-epoch state belongs in the CSV.
- **Required fields (all populated from `config.py` and the live dataset):**
  - `run_id` — from `make_run_id()`.
  - `timestamp` — from the same call.
  - `target` — `"age"` or `"fluid_iq"`.
  - `model_architecture` — a structured layer summary, e.g.:
    ```
    {
      "stage1": "Linear(2920, 1024) -> BN1d(1024) -> GELU -> Dropout(0.3)",
      "se_block": "Linear(1024, 256) -> GELU -> Linear(256, 1024) -> Sigmoid",
      "stage2": "Linear(1024, 512) -> BN1d(512) -> GELU -> Dropout(0.3)",
      "stage3": "Linear(512, 256) -> BN1d(256) -> GELU -> Dropout(0.2)",
      "proj":   "Linear(256, 128)",
      "aux_head": "Linear(128, 1)"
    }
    ```
  - `hyperparameters` — `{lr, batch_size, epochs, weight_decay, lambda_sparse, lambda_aux, dropout1, dropout2, dropout3}`.
  - `dataset_sizes` — `{train, val, test}`.
  - `split_seed` — the seed used for the persisted split.
  - `preprocessing_steps_applied` — read from the most recent preprocessing JSON (path passed via `--preprocess-log` CLI arg, or located by convention under `logs/preprocessing/`); copy `imputer_fit_split`, `normalisation_method`, `scaler_fit_split`, `normalisation_statistics_file`.
  - `checkpoint_save_path` — the resolved path under `research/checkpoints/`.
  - `training_log_save_path` — the path of the CSV created in 4.1.
- **Partial-run handling:** because the JSON is written before training, even a run that crashes after epoch 0 has a complete experiment record on disk. The CSV (which may be partial) still cross-references `run_id` via the filename.

### 4.3 Preprocessing Logs (JSON)

- **Responsible scripts:** `proteomics/data/preprocess.py` (primary) and `proteomics/data/build_dataset.py` (target-link summary).
- **File path:** `research/logs/preprocessing/preprocess_log_[timestamp].json` (for `preprocess.py`) and `research/logs/preprocessing/dataset_build_[target]_[timestamp].json` (for `build_dataset.py`).
- **When written:** at the very end of the script's main routine, after all artefacts (`proteomics_tensor.pt`, scaler `.joblib`, imputer `.joblib`, optional residualization `.npz`) have been saved successfully. Single write.
- **Required fields for `preprocess.py`:**
  - `timestamp` — produced at script start, embedded in the file name and the JSON.
  - `input_file` — absolute path of `UKB_Protein_BL_withCov.csv`.
  - `input_shape` — `[N, D]` after metadata-column drop (expected `[52697, 2920]`).
  - `missing_values_before_imputation` — total NaN count.
  - `missing_values_after_imputation` — should be 0; if not, the script must `WARNING`-log this before writing.
  - `imputer_fit_split` — fixed string `"train"`.
  - `normalisation_method` — `"z-score"` or `"covariate-residualized"`.
  - `scaler_fit_split` — fixed string `"train"`.
  - `output_shape` — same as `input_shape` (rows × cols are unchanged by imputation/normalisation).
  - `output_file` — `research/data/proteomics_tensor.pt`.
  - `normalisation_statistics_file` — `research/data/artefacts/scaler_[seed].npz`.
  - Plus, recommended additions (allowed because the spec says "Required fields" and does not forbid extras): `split_seed`, `train_size`, `val_size`, `test_size`, `imputer_artefact_path`, `covariate_residualization_applied` (boolean), `residualization_artefact_path` (or `null`).
- **Required fields for `build_dataset.py`:** `timestamp`, `input_proteomics_file`, `input_target_csv`, `target_column`, `rows_in_target_csv_before_dropna`, `rows_after_dropna`, `intersection_size`, `output_file`, `target_mean`, `target_std`, `train_size`, `val_size`, `test_size`, `split_indices_file`.
- **Partial-run handling:** preprocessing is single-shot. If the script crashes before this JSON is written, the absence of a completion JSON for that timestamp signals the run did not finish; the corresponding `process_preprocess_*.log` file (4.4) records exactly where it stopped.

### 4.4 Process Logs (Plain Text `.log`)

- **Responsible scripts:** every entry script — `preprocess.py`, `build_dataset.py`, `train_stage1.py`, `cv_stage1.py`, `protein_importance.py`. Each call produces exactly one `.log` file.
- **File path:** `research/logs/process/process_[script_name]_[timestamp].log` where `[script_name]` is the entry script's basename without `.py` (e.g., `process_train_stage1_20260506_143201.log`).
- **When opened:** as the very first action in `main()`, immediately after `set_global_seeds(...)`. The logger writes an `INFO` line stating the script name, run ID (if applicable), CLI args, target, seed, and resolved input/output paths.
- **When written:** throughout execution, using the `make_process_logger` helper from 2.3. Required entries:
  - `INFO` at script start (configuration summary).
  - `INFO` after each significant phase boundary (data loaded, split applied, imputer fit, scaler fit, model built, training started, every epoch completed with summary metrics, checkpoint saved, evaluation done, artefacts written).
  - `WARNING` whenever a non-fatal anomaly is detected (e.g., residual NaNs after imputation, validation R² unusually low, epoch took unexpectedly long).
  - `ERROR` on any exception caught by a top-level `try/except`; the script then re-raises so the process exits non-zero.
- **Format:** exactly `[YYYY-MM-DD HH:MM:SS] [LEVEL] Message`, enforced by the formatter `"[%(asctime)s] [%(levelname)s] %(message)s"` with `datefmt="%Y-%m-%d %H:%M:%S"`.
- **When closed:** the `FileHandler` is closed in a top-level `finally:` block in `main()` after writing a final `INFO` line `"Run [run_id] complete."` (or `"Run [run_id] aborted."` on exception).
- **Partial-run handling:** because `logging.FileHandler` flushes on each emit (Python default for non-buffered handlers), a killed process leaves a `.log` file that ends at the last successfully emitted line. The file is not rotated; one log per script invocation, indexed by timestamp.

---

## 5. Implementation Order Summary

Work through these in order. Each step assumes the previous ones are merged and tested.

1. **2.1** — Create `research/proteomics/config.py` (paths, hyperparameters, seeds).
2. **2.2** — Create `research/proteomics/utils/seed.py` (`set_global_seeds`, `make_split_generator`, `seed_worker`).
3. **2.3** — Create `research/proteomics/utils/logging_utils.py` (process logger, JSON writer, CSV appender, `make_run_id`).
4. **2.4** — Create `research/proteomics/data/preprocess.py` (split-aware KNN + scaler, optional covariate residualization, persisted artefacts, JSON + .log).
5. **2.5** — Create `research/proteomics/data/build_dataset.py` (target linking with `--target {age, fluid_iq}`, post-intersection split indices, JSON + .log).
6. **2.6** — Create `research/proteomics/models/se_block.py` (1024 → 256 → 1024, GELU, returns `s_p`).
7. **2.7** — Create `research/proteomics/models/encoder.py` (`ProteomicsEncoder`, returns `(z_p, s_p)`, no head).
8. **2.8** — Create `research/proteomics/models/aux_head.py` (`ProteomicsAuxHead(128 → 1)`).
9. **2.9** — Create `research/proteomics/tests/test_shapes.py`; run it to confirm shapes before going further.
10. **2.10** — Create `research/proteomics/training/losses.py` (`ProteomicsStage1Loss` with labelled `y_p` MSE and `||s_p||_1` term).
11. **2.11** — Create `research/proteomics/training/train_stage1.py` (Stage 1 loop with `lr=1e-3`, pooled metrics, checkpoint with separate `encoder_state_dict` / `aux_head_state_dict`, full logging).
12. Run Stage 1 once on `--target age` end-to-end as a smoke run; verify all four log artefacts land in their expected paths.
13. **2.13** — Create `research/proteomics/analysis/protein_importance.py` (uses `(z_p, s_p)` from forward; computes input gradient × input attribution; saves CSV and `s_p` mean).
14. **2.12** — (Optional) Create `research/proteomics/training/cv_stage1.py`.
15. Run Stage 1 on `--target fluid_iq` and run the importance analysis on both targets.
16. **2.14** — Write `research/proteomics/README.md` and add module-level docstrings to encoder, SE-block, aux head, and loss modules.
17. **2.15** — Move `find_targets.py` and `find_age.py` to `proteomics/scripts/discovery/`.
18. **2.16** — Archive legacy artefacts to `research/_archive/legacy_v0/`; remove `Pre_Processing/`; clean `__pycache__/` and add `.gitignore`.
19. Final verification: re-run the shape test, re-run Stage 1 from scratch with the published `GLOBAL_SEED`, confirm the resulting CSV / JSON / .log triplet matches a previously logged run bit-for-bit (modulo timestamps and `run_id`).

---

## 6. Open Questions

The following decisions are outside what the existing files and the prior report alone can resolve. They should be confirmed by the researcher before implementation begins.

1. **Covariate residualization — which covariates and which form?**
   The spec permits either raw z-score or covariate-residualized + standardised input. The source CSV is `UKB_Protein_BL_withCov.csv`, which strongly implies covariates are present. The minimum viable residualization set is plausibly {age, sex, batch, plate, fasting time, assay-warning flag}, but the exact column names in the CSV have not been confirmed in this session (the CSV was not opened, per the read-only constraint on `uk_BIOBANK_DATA/`). **Decision needed:** which covariate columns to residualize against, and whether residualization is on by default or opt-in via `--residualize`.

2. **Per-protein importance under the corrected architecture.**
   With SE relocated to the 1024-d hidden layer, `s_p ∈ R^1024` no longer maps one-to-one onto proteins. The plan proposes input-gradient × input attribution (mean over the dataset) as the per-protein importance, plus a separate `sp_mean` artefact for hidden-channel importance. **Decision needed:** is gradient × input the right attribution method, or should the proteomics encoder additionally include an input-level attention/gate (e.g., a learnable 2920-d gate at the input) so that "per-protein importance" remains a direct readout? The spec does not prescribe an answer and §6 of the prior report flags this for clarification.

3. **Train/Val/Test ratio.**
   The prior report's recommendation 4 suggests 70/15/15. The plan codifies that. **Decision needed:** confirm 70/15/15, or specify a different ratio (e.g., 80/10/10, or stratified by some covariate).

4. **K-fold parameters.**
   If 2.12 is implemented: number of folds (default proposal: K=5), and whether the held-out 15% test split is preserved across folds (the plan assumes yes — folds are within the train+val 85%, leaving the 15% test as a one-shot held-out set). **Decision needed:** confirm K and whether to nest CV within the test split or replace the split altogether.

5. **`λ_sparse` interpretation of `||s_p||_1`.**
   The TCG-Net spec gives the sparsity loss as `L_sparse = ||w||_1 + ||s_p||_1 + ||α_f||_1` with `λ_sparse = 1e-4`. In Stage 1, only the `s_p` term is computable. The plan defines `||s_p||_1` as the per-sample L1 mean averaged across the batch, so it is invariant to batch size. **Decision needed:** confirm this convention, or specify an alternative (e.g., sum across hidden dim without batch averaging).

6. **`λ_aux` in Stage 1.**
   In the joint loss (Stage 2), `λ_aux = 0.3` multiplies the *sum* `MSE(y, y_p) + MSE(y, y_f) + MSE(y, y_fp)`. In Stage 1 only `MSE(y, y_p)` exists, so `λ_aux` does not naturally appear in the total loss; the plan logs it for traceability and uses the bare `MSE(y, y_p)` as the Stage 1 reconstruction term. **Decision needed:** confirm that Stage 1 should optimise plain `MSE(y, y_p) + λ_sparse · ||s_p||_1` (and not, for example, `λ_aux · MSE(y, y_p) + λ_sparse · ||s_p||_1`).

7. **GPU vs CPU preprocessing parity.**
   The plan unifies preprocessing under `preprocess.py --device {cpu, gpu}`. The current GPU implementation seeds the search database with column means, which is a different KNN behaviour from sklearn's row-wise neighbour search. **Decision needed:** is exact parity with sklearn required (in which case the GPU path needs a more faithful implementation), or is the existing GPU heuristic acceptable as a faster approximation that we explicitly document as such?

8. **Checkpoint pruning policy.**
   The plan saves `stage1_[target]_[run_id].pt` for the best validation epoch only. **Decision needed:** also save a `last.pt` for resumption? Cap the number of checkpoints retained?

9. **Whether to retain `_archive/legacy_v0/` in git.**
   The legacy `.pt` files are large (~600 MB each). Even moved into `_archive/`, committing them costs storage. **Decision needed:** archive but git-ignore (so they exist on disk for sanity comparisons but are not tracked), or delete outright?

10. **Timestamp timezone.**
    All filenames and JSON `timestamp` fields share the same convention. **Decision needed:** UTC or local time? The plan defaults to local but expects this to be set centrally in `config.py`.
