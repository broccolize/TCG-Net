"""
Proteomics encoder (SE-MLP) for the TCG-Net proteomics branch.

Input / Output contract
-----------------------
    Input:  x   ∈ R^(B × 2920)   — z-scored (or residualised + z-scored)
                                    plasma proteomics feature matrix.
    Output: (z_p, s_p)
            z_p ∈ R^(B × 128)    — final embedding; fed to the auxiliary
                                    head for Stage 1 training, and into
                                    the TCG-Net fusion module in Stage 2.
            s_p ∈ R^(B × 1024)   — SE gate vector; used by the Stage 1
                                    loss (sparsity term) and by the
                                    importance analysis script.

The encoder **never** returns a prediction.  A separate
``ProteomicsAuxHead`` (``models/aux_head.py``) maps ``z_p → y_p`` for
Stage 1 training.  Keeping the head separate means the encoder's
``state_dict`` can be loaded cleanly into TCG-Net Stage 2:

    ckpt = torch.load("stage1_age_<run_id>.pt", map_location="cpu")
    encoder = ProteomicsEncoder()
    encoder.load_state_dict(ckpt["encoder_state_dict"])

Architecture (in order)
-----------------------
    Stage 1:    Linear(2920, 1024) → BatchNorm1d(1024) → GELU → Dropout(0.3)
    SE Block:   SEBlock(in_dim=1024, reduction_dim=256)
                → returns (h̃_p, s_p)
    Stage 2:    Linear(1024, 512)  → BatchNorm1d(512)  → GELU → Dropout(0.3)
    Stage 3:    Linear(512, 256)   → BatchNorm1d(256)  → GELU → Dropout(0.2)
    Projection: Linear(256, 128)
    Output:     z_p ∈ R^(B × 128)

All dimension constants are imported from ``config.py`` so no numbers
are hardcoded in this file.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make the sibling ``models`` package importable when this file is run
# directly (e.g., during shape tests executed from the repo root).
# ---------------------------------------------------------------------------
_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

# Relative import of SEBlock from the same package.
from models.se_block import SEBlock  # noqa: E402

from config import (  # noqa: E402
    DROPOUT_1,
    DROPOUT_2,
    DROPOUT_3,
    EMBED_DIM,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    HIDDEN_DIM_3,
    INPUT_DIM,
    SE_REDUCTION_DIM,
)


class ProteomicsEncoder(nn.Module):
    """
    SE-MLP proteomics encoder for TCG-Net Stage 1 pretraining.

    Transforms a batch of imputed, standardised proteomics profiles into
    a compact embedding ``z_p`` and exposes the SE gate ``s_p`` for
    sparsity regularisation and interpretability analysis.

    Parameters
    ----------
    input_dim : int     Feature count — ``INPUT_DIM`` (2920) by default.
    hidden_dim_1 : int  First hidden block width — ``HIDDEN_DIM_1`` (1024).
    hidden_dim_2 : int  Second hidden block width — ``HIDDEN_DIM_2`` (512).
    hidden_dim_3 : int  Third hidden block width — ``HIDDEN_DIM_3`` (256).
    embed_dim : int     Final embedding width — ``EMBED_DIM`` (128).
    se_reduction_dim : int  SE bottleneck — ``SE_REDUCTION_DIM`` (256).
    dropout_1 : float   Dropout rate after Stage 1 — ``DROPOUT_1`` (0.3).
    dropout_2 : float   Dropout rate after Stage 2 — ``DROPOUT_2`` (0.3).
    dropout_3 : float   Dropout rate after Stage 3 — ``DROPOUT_3`` (0.2).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim_1: int = HIDDEN_DIM_1,
        hidden_dim_2: int = HIDDEN_DIM_2,
        hidden_dim_3: int = HIDDEN_DIM_3,
        embed_dim: int = EMBED_DIM,
        se_reduction_dim: int = SE_REDUCTION_DIM,
        dropout_1: float = DROPOUT_1,
        dropout_2: float = DROPOUT_2,
        dropout_3: float = DROPOUT_3,
    ) -> None:
        super().__init__()

        # Stage 1: project raw proteins to the 1024-d hidden space.
        # BatchNorm1d normalises each hidden unit across the batch before
        # GELU so that gradients are well-conditioned at the start of training.
        self.stage1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.GELU(),
            nn.Dropout(dropout_1),
        )

        # SE Block: learns which hidden units to amplify or suppress.
        # Returns (h̃_p, s_p); s_p is forwarded to the caller for the
        # sparsity loss and interpretability analysis.
        self.se_block = SEBlock(in_dim=hidden_dim_1, reduction_dim=se_reduction_dim)

        # Stage 2: compress 1024-d gated representation to 512-d.
        self.stage2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.GELU(),
            nn.Dropout(dropout_2),
        )

        # Stage 3: compress 512-d to 256-d.
        self.stage3 = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.BatchNorm1d(hidden_dim_3),
            nn.GELU(),
            nn.Dropout(dropout_3),
        )

        # Projection: map 256-d to the final 128-d embedding z_p.
        # No activation or BN here — z_p is fed directly to the aux head
        # (Stage 1) or to the fusion module (Stage 2).
        self.proj = nn.Linear(hidden_dim_3, embed_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of proteomics profiles.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 2920)
            Imputed and z-scored (or residualised + z-scored) protein
            feature matrix.

        Returns
        -------
        z_p : torch.Tensor, shape (B, 128)
            Final proteomics embedding.  Used by ProteomicsAuxHead in
            Stage 1 and by the fusion module in Stage 2.
        s_p : torch.Tensor, shape (B, 1024)
            SE gate vector.  Values are in (0, 1) because SEBlock applies
            Sigmoid.  Passed to the Stage 1 loss for sparsity
            regularisation; also saved for interpretability analysis.
        """
        # Stage 1: project input to 1024-d hidden space.
        h = self.stage1(x)                  # (B, 1024)

        # SE Block: compute and apply channel-wise gates.
        h_tilde, s_p = self.se_block(h)    # h_tilde: (B, 1024), s_p: (B, 1024)

        # Stage 2: compress gated representation.
        h = self.stage2(h_tilde)            # (B, 512)

        # Stage 3: further compression.
        h = self.stage3(h)                  # (B, 256)

        # Projection: produce the final embedding.
        z_p = self.proj(h)                  # (B, 128)

        return z_p, s_p
