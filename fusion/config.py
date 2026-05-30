"""
Central configuration for the TCG-Net fusion branch (Stage 2).

This module is the single source of truth for:
    - Shared data directory (proteomics tensor, joint datasets, split file).
    - Network architecture dimensions specific to the fusion stage.
    - Stage 2 learning rates and training hyperparameters.
    - Loss coefficients for the auxiliary, Shapley alignment, and sparsity terms.

All entry scripts in the fusion pipeline must import from this module
rather than redefining their own constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Shared data directory
# All cross-modality inputs and joint dataset outputs live here.
# ---------------------------------------------------------------------------
SHARED_DATA_DIR = Path("/shared/tcgnet_data")

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
R = 66            # Number of fMRI ROIs (from colleague's correlation matrix).
NODE_DIM = 72     # Per-node feature dimension after GNN embedding.
EMBED_DIM = 128   # Proteomics encoder output dimension (z_p).
JOINT_DIM = 512   # Fusion head input dimension.

# ---------------------------------------------------------------------------
# Stage 2 learning rates
# Three-group schedule: frozen encoder, unfrozen encoder, fusion head.
# ---------------------------------------------------------------------------
LR_STAGE2_FROZEN = 5e-4       # LR while proteomics encoder is frozen.
LR_STAGE2_ENCODER = 1e-5      # LR for proteomics encoder once unfrozen.
LR_STAGE2_FUSION_HEAD = 1e-4  # LR for the Shapley fusion head.

# ---------------------------------------------------------------------------
# Stage 2 training schedule
# ---------------------------------------------------------------------------
FREEZE_EPOCHS = 10  # Epochs to train with encoder frozen before unfreezing.

# ---------------------------------------------------------------------------
# Loss coefficients
# ---------------------------------------------------------------------------
LAMBDA_AUX = 0.3      # Coefficient on the auxiliary per-modality loss terms.
LAMBDA_SHAPLEY = 0.1  # Coefficient on the Shapley alignment regularisation term.
LAMBDA_SPARSE = 1e-4  # Coefficient on the L1 sparsity term for Shapley weights.
