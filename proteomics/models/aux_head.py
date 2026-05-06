"""
Auxiliary prediction head for Stage 1 proteomics pretraining.

Role in TCG-Net Stage 1
------------------------
``ProteomicsAuxHead`` takes the 128-d embedding ``z_p`` produced by
``ProteomicsEncoder`` and maps it to a scalar prediction ``y_p``:

    z_p  ∈ R^(B × 128)  →  y_p  ∈ R^(B × 1)

During Stage 1 training the loss is:

    L_stage1 = MSE(y, y_p) + lambda_sparse * ||s_p||_1

``y_p`` is the ``y_p`` component of the auxiliary loss ``L_aux`` in the
full TCG-Net objective (Stage 2).

Stage 1 → Stage 2 handoff
--------------------------
The encoder and auxiliary head are saved under **separate keys** in the
Stage 1 checkpoint:

    {
      "encoder_state_dict":  encoder.state_dict(),
      "aux_head_state_dict": aux_head.state_dict(),
      ...
    }

In Stage 2, **only** ``encoder_state_dict`` is loaded; the auxiliary head
is discarded and replaced by the full auxiliary loss structure
(y_p, y_f, y_fp) and the Shapley Fusion Module.  Loading just the
encoder is therefore a single line:

    encoder.load_state_dict(ckpt["encoder_state_dict"])

This clean separation is why the head must never be embedded inside
``ProteomicsEncoder``.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make the sibling proteomics package importable when run as a script.
# ---------------------------------------------------------------------------
_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from config import AUX_HEAD_OUT_DIM, EMBED_DIM  # noqa: E402


class ProteomicsAuxHead(nn.Module):
    """
    Single-layer linear head that maps the proteomics embedding to a
    scalar target prediction.

    Parameters
    ----------
    embed_dim : int
        Input dimension — must match ``ProteomicsEncoder`` output width.
        Defaults to ``EMBED_DIM`` (128) from ``config.py``.
    out_dim : int
        Output dimension — 1 for scalar regression (age or fluid IQ).
        Defaults to ``AUX_HEAD_OUT_DIM`` (1) from ``config.py``.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        out_dim: int = AUX_HEAD_OUT_DIM,
    ) -> None:
        super().__init__()
        # One linear layer: z_p (B, 128) → y_p (B, 1).
        # No activation — this is a regression head, not a classifier.
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, z_p: torch.Tensor) -> torch.Tensor:
        """
        Predict the target from the proteomics embedding.

        Parameters
        ----------
        z_p : torch.Tensor, shape (B, embed_dim)
            Final proteomics embedding from ``ProteomicsEncoder``.

        Returns
        -------
        y_p : torch.Tensor, shape (B, out_dim)
            Scalar target prediction (e.g., predicted age or fluid IQ).
        """
        return self.fc(z_p)  # (B, out_dim)
