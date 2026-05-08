"""
Central configuration for the TCG-Net proteomics encoder branch.

This module is the single source of truth for:
    - Filesystem paths (raw data, processed tensors, checkpoints, logs).
    - Default hyperparameters for the Stage 1 (single-modality) pretraining
      run and Stage 2 (joint TCG-Net) fine-tuning learning rates.
    - Network architecture dimensions (input, hidden, embedding).
    - Reproducibility seeds (global, split, cross-validation).
    - Train / val / test split fractions.
    - Preprocessing defaults (KNN imputer, optional covariate residualization).
    - Timezone for all timestamps emitted by the pipeline.

All paths are constructed from ``Path(__file__).resolve().parent`` so they
resolve correctly no matter the current working directory of the caller.
Every entry script in the proteomics pipeline must import from this module
rather than redefining its own constants.
"""

# ---------------------------------------------------------------------------
# QUICK-REFERENCE: full pipeline run order (all commands from repo root)
#
#   1. Preprocessing — split-aware KNN imputation + z-score (~3 h on CPU):
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/data/preprocess.py
#      # Optional covariate residualization:
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/data/preprocess.py --residualize
#
#   2. Build target datasets:
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/data/build_dataset.py --target age
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/data/build_dataset.py --target fluid_iq
#
#   3. Stage 1 training:
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/training/train_stage1.py --target age
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/training/train_stage1.py --target fluid_iq
#
#   4. Protein importance analysis (replace <run_id> with actual checkpoint name):
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/analysis/protein_importance.py \
#        --target age \
#        --checkpoint TCG-Net/proteomics/checkpoints/stage1_age_<run_id>_best.pt
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/analysis/protein_importance.py \
#        --target fluid_iq \
#        --checkpoint TCG-Net/proteomics/checkpoints/stage1_fluid_iq_<run_id>_best.pt
#
#   5. K=5 cross-validation (optional):
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/training/cv_stage1.py --target age
#      conda run --no-capture-output -n tcg_net \
#        python TCG-Net/proteomics/training/cv_stage1.py --target fluid_iq
# ---------------------------------------------------------------------------

from pathlib import Path

# ---------------------------------------------------------------------------
# Path anchors
# ---------------------------------------------------------------------------
# HERE       = TCG-Net/proteomics/      (directory containing this config.py)
# REPO_ROOT  = TCG-Net/                  (the monorepo root)
# HOME_ROOT  = parent of TCG-Net/        (where uk_BIOBANK_DATA/ lives)
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
HOME_ROOT = REPO_ROOT.parent


# ---------------------------------------------------------------------------
# PATHS
# Filesystem locations for raw inputs, processed tensors, persisted artefacts,
# checkpoints, logs and analysis outputs. All paths are pathlib.Path objects.
# ---------------------------------------------------------------------------

# Root of all proteomics-side data (processed tensors, splits, artefacts).
DATA_DIR = HERE / "data"

# Raw UKB plasma proteomics CSV (with covariates). Lives outside the monorepo
# under uk_BIOBANK_DATA/ at the parent of TCG-Net/.
RAW_PROTEIN_CSV = HOME_ROOT / "uk_BIOBANK_DATA" / "protein" / "UKB_Protein_BL_withCov.csv"

# Output of preprocess.py: imputed + standardised tensor with split indices.
PROCESSED_TENSOR_PATH = DATA_DIR / "proteomics_tensor.pt"

# Outputs of build_dataset.py: target-specific (x, y, splits) bundles.
DATASET_AGE_PATH = DATA_DIR / "dataset_age.pt"
DATASET_IQ_PATH = DATA_DIR / "dataset_fluid_iq.pt"

# Checkpoints (best-validation and last.pt for resumption).
CHECKPOINT_DIR = HERE / "checkpoints"

# Per-log-type directories. The four log types are documented in
# logging_utils.py and in the proteomics/README.md.
LOG_DIR_TRAINING = HERE / "logs" / "training"
LOG_DIR_EXPERIMENT = HERE / "logs" / "experiments"
LOG_DIR_PREPROCESS = HERE / "logs" / "preprocessing"
LOG_DIR_PROCESS = HERE / "logs" / "process"

# Persisted preprocessing artefacts (KNN imputer, scaler, optional residualizer).
ARTEFACTS_DIR = DATA_DIR / "artefacts"

# Persisted split index arrays (train/val/test and CV folds).
SPLITS_DIR = DATA_DIR / "splits"

# Outputs of analysis scripts (per-protein importance CSVs, sp_mean .npy, etc.).
ANALYSIS_DIR = HERE / "analysis_outputs"


# ---------------------------------------------------------------------------
# HYPERPARAMETERS (Stage 1)
# Default optimisation, regularisation and dropout values for the Stage 1
# single-modality pretraining loop. Stage 2 uses a different set of learning
# rates (also stored here) for the encoder, fusion head and frozen segments.
# ---------------------------------------------------------------------------
LR_STAGE1 = 1e-3              # Stage 1 base learning rate (encoder + aux head).
LR_STAGE2_FROZEN = 5e-4       # Stage 2 LR while the encoder is frozen.
LR_STAGE2_ENCODER = 1e-5      # Stage 2 LR for the proteomics encoder once unfrozen.
LR_STAGE2_FUSION_HEAD = 1e-4  # Stage 2 LR for the Shapley fusion head.

BATCH_SIZE = 64
EPOCHS = 30
WEIGHT_DECAY = 1e-5

LAMBDA_SPARSE = 1e-4   # Coefficient on ||s_p||_1 in Stage 1.
LAMBDA_AUX = 0.3       # Stored for forward-compat (Stage 2 L_aux); not applied in Stage 1.
LAMBDA_SHAPLEY = 0.1   # Stored for forward-compat with the Stage 2 Shapley alignment term.

# Dropout rates after the three encoder stages (1024-d, 512-d, 256-d blocks).
DROPOUT_1 = 0.3
DROPOUT_2 = 0.3
DROPOUT_3 = 0.2


# ---------------------------------------------------------------------------
# ARCHITECTURE
# Layer dimensions for the SE-MLP proteomics encoder. The encoder maps
# X_p ∈ R^(B×2920) → z_p ∈ R^(B×128) and exposes s_p ∈ R^(B×1024).
# ---------------------------------------------------------------------------
INPUT_DIM = 2920          # Number of plasma proteins (Olink panel).
HIDDEN_DIM_1 = 1024       # First hidden block (where SE gate is applied).
HIDDEN_DIM_2 = 512        # Second hidden block.
HIDDEN_DIM_3 = 256        # Third hidden block.
EMBED_DIM = 128           # Final embedding dimension (z_p).
SE_REDUCTION_DIM = 256    # Bottleneck width inside the SE block (1024 → 256 → 1024).
AUX_HEAD_OUT_DIM = 1      # Stage 1 auxiliary regression head output (scalar y_p).


# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# Seeds used across the pipeline. GLOBAL_SEED is set at the start of every
# entry script. SPLIT_SEED determines the train/val/test partition.
# CV_SEED determines the K-fold partition of the train+val pool.
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
SPLIT_SEED = 1234
CV_SEED = 7
CV_FOLDS = 5  # K=5 cross-validation, mandatory on every training run.


# ---------------------------------------------------------------------------
# SPLIT FRACTIONS
# Fixed 70/15/15 train/val/test partition (must sum to 1.0).
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


# ---------------------------------------------------------------------------
# PREPROCESSING
# Defaults for the split-aware preprocessing pipeline (preprocess.py).
# ---------------------------------------------------------------------------
KNN_NEIGHBORS = 5            # n_neighbors for sklearn.KNNImputer.
RESIDUALIZE_DEFAULT = False  # Covariate residualization is opt-in via --residualize.


# ---------------------------------------------------------------------------
# TIMESTAMPS
# All filenames and JSON timestamp fields use this timezone. Set centrally
# here so the entire pipeline is internally consistent.
# ---------------------------------------------------------------------------
TIMEZONE = "UTC"
