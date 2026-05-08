"""
Stage 1 loss function for the TCG-Net proteomics encoder branch.

Loss formula
------------
The Stage 1 total loss is:

    L_stage1 = MSE(y, y_p) + lambda_sparse * ||s_p||_1

where:

    MSE(y, y_p)      — mean squared error between the target label y and
                       the auxiliary prediction y_p.  In TCG-Net Stage 2,
                       this term becomes the y_p component of L_aux:
                           L_aux = MSE(y, y_p) + MSE(y, y_f) + MSE(y, y_fp)
                       multiplied by lambda_aux = 0.3.  In Stage 1 the
                       other two MSE terms do not yet exist, and
                       lambda_aux is NOT applied here (see note below).

    ||s_p||_1        — per-sample L1 mean of the SE gate vector, averaged
                       across the batch.  Concretely:
                           sp_l1 = mean_B( mean_dim( |s_p| ) )
                       where mean_dim averages over the 1024 hidden units
                       and mean_B averages over the B samples.  This
                       convention makes the term invariant to batch size.

    lambda_sparse    — coefficient from config.LAMBDA_SPARSE (default 1e-4).

Note on lambda_aux
------------------
In TCG-Net Stage 2, the joint loss is:
    L = L_main
        + lambda_aux  * (l_yp + l_yf + l_yfp)
        + lambda_shapley * L_shapley
        + lambda_sparse * (||w||_1 + ||s_p||_1 + ||alpha_f||_1)
The Stage 1 module is a strict subset of this objective.  The keys in
the returned dict (``l_yp``, ``sp_l1_norm``) map directly onto the
corresponding Stage 2 terms, so the Stage 1 → Stage 2 extension is
additive and requires no re-plumbing of the Stage 1 code.
lambda_aux is stored in config.py for forward compatibility but is NOT
applied in Stage 1 (where y_f and y_fp do not yet exist).

Usage
-----
Training code calls::

    loss_fn  = ProteomicsStage1Loss()
    loss_dict = loss_fn(y_true, y_p, s_p)
    loss_dict["total"].backward()          # backward through full loss
    train_loss   = loss_dict["l_yp"].item()      # logged to CSV
    sp_l1_norm   = loss_dict["sp_l1_norm"].item()  # logged to CSV
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make the proteomics package root importable when run as a script.
# ---------------------------------------------------------------------------
_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from config import LAMBDA_SPARSE  # noqa: E402


class ProteomicsStage1Loss(nn.Module):
    """
    Stage 1 loss combining the y_p MSE term with SE-gate sparsity.

    Parameters
    ----------
    lambda_sparse : float
        Coefficient on the ||s_p||_1 regularisation term.
        Defaults to ``LAMBDA_SPARSE`` (1e-4) from ``config.py``.
    """

    def __init__(self, lambda_sparse: float = LAMBDA_SPARSE) -> None:
        """
        Initialise the Stage 1 loss with a configurable sparsity coefficient.

        Parameters
        ----------
        lambda_sparse : float
            Multiplier applied to the ``||s_p||_1`` term before adding it
            to the MSE.  Defaults to ``LAMBDA_SPARSE`` (1e-4) from
            ``config.py``.  Setting this to 0 disables sparsity entirely,
            reducing the loss to plain MSE(y, y_p).
        """
        super().__init__()
        self.lambda_sparse = lambda_sparse
        # reduction="mean" averages over all elements in the batch, which
        # is the standard for regression losses and matches the Stage 2
        # specification (y_p MSE is the per-sample mean).
        self.mse_criterion = nn.MSELoss(reduction="mean")

    def forward(
        self,
        y_true: torch.Tensor,
        y_p: torch.Tensor,
        s_p: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute Stage 1 loss and return all components as a dict.

        Parameters
        ----------
        y_true : torch.Tensor, shape (B,) or (B, 1)
            Ground-truth target labels (age or fluid IQ).
        y_p : torch.Tensor, shape (B,) or (B, 1)
            Auxiliary predictions from ``ProteomicsAuxHead``.
        s_p : torch.Tensor, shape (B, 1024)
            SE gate vector from ``ProteomicsEncoder``.  Values are in
            (0, 1) due to the Sigmoid gate.

        Returns
        -------
        dict with keys:
            "total"       — L_stage1 = l_yp + l_sparse.  Call
                            ``.backward()`` on this key.
            "l_yp"        — MSE(y_true, y_p).  Log to training CSV as
                            ``train_loss``.
            "l_sparse"    — lambda_sparse * sp_l1.  Diagnostic.
            "sp_l1_norm"  — raw ||s_p||_1 (before lambda scaling).
                            Log to training CSV as ``sp_l1_norm``.
        """
        # Squeeze both tensors to 1-D so MSELoss sees matching shapes
        # regardless of whether y_true is (B,) or (B, 1).
        y_true_flat = y_true.squeeze(-1)   # (B,)
        y_p_flat = y_p.squeeze(-1)         # (B,)

        # y_p MSE: the primary reconstruction objective.
        l_yp = self.mse_criterion(y_p_flat, y_true_flat)

        # ||s_p||_1: per-sample L1 mean, then batch-averaged.
        # Step 1 — mean over hidden dim (dim=1) gives one scalar per sample.
        # Step 2 — mean over batch dim (dim=0) makes it batch-size-invariant.
        sp_l1 = s_p.abs().mean(dim=1).mean(dim=0)   # scalar

        # Weighted sparsity term added to the total loss.
        l_sparse = self.lambda_sparse * sp_l1        # scalar

        # Total Stage 1 loss: MSE + sparsity regularisation.
        l_total = l_yp + l_sparse

        # Return all components so training code can log each separately
        # without re-computing them.
        return {
            "total": l_total,       # call .backward() on this
            "l_yp": l_yp,           # logged as train_loss in CSV
            "l_sparse": l_sparse,   # diagnostic (weighted)
            "sp_l1_norm": sp_l1,    # logged as sp_l1_norm in CSV (unweighted)
        }
