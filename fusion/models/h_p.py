"""
H_p — proteomics auxiliary head for TCG-Net Stage 2.

``ProteomicsAuxHead`` is reused unchanged as H_p because it already satisfies
every Stage 2 requirement:

    * Input:  z_p ∈ R^(B × 128)  — proteomics embedding from ProteomicsEncoder
    * Output: y_p ∈ R^(B × 1)   — scalar auxiliary prediction (age or fluid IQ)
    * Single nn.Linear(128 → 1), no activation (regression head)
    * No Stage-1-specific code: no loss hooks, no hardcoded stage assumptions.
      The checkpoint format already separates "aux_head_state_dict" from
      "encoder_state_dict", and the Stage 2 training loop discards the former —
      so H_p starts from random weights in Stage 2, which is the intended
      behaviour per the TCG-Net paper (§7.2).

Usage in Stage 2:

    from fusion.models.h_p import H_p
    h_p = H_p()                     # Linear(128 → 1)
    y_p = h_p(z_p)                  # (B, 1)
"""

import sys
from pathlib import Path

# Make proteomics package importable regardless of working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROTEOMICS_PKG = _REPO_ROOT / "proteomics"
if str(_PROTEOMICS_PKG) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG))

from models.aux_head import ProteomicsAuxHead  # noqa: E402

# Canonical Stage 2 alias.  No subclassing or wrapping — the class is used
# directly so that model.state_dict() keys are identical to Stage 1 artefacts
# if a pre-trained aux head is ever loaded for warm-starting.
H_p = ProteomicsAuxHead

__all__ = ["H_p"]
