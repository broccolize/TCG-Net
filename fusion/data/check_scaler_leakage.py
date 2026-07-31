"""
Quantify the proteomics normalisation-statistic leakage (FIX_PLAN item 6).

The proteomics imputer and StandardScaler are fit on the Stage-1 *training*
split. Roughly 70% of the Stage-2 joint val/test participants land in that
split, so their protein values contribute to the mean/std that every Stage-2
input is normalised by. This is a feature-only, label-free leakage channel —
expected to be small, but "expected" is not "measured", and the honest fix
(re-running preprocess.py --exclude-eids) costs a full KNN imputation plus a
Stage-1 retrain plus a Stage-2 rerun.

This script measures the channel first, in seconds, so that decision is
evidence-based.

Method. proteomics_tensor.pt already holds z-scored values, so the scaler that
*would* have been fit without the leaking rows can be recovered without
touching the raw CSV: in the current z coordinates, refitting on the clean fit
set means subtracting that set's per-feature mean ``m`` and dividing by its
per-feature std ``s``. If the leakage were irrelevant, ``m ≈ 0`` and ``s ≈ 1``.
The reported shift is

    dz = (z - m) / s - z

evaluated on the rows that would be Stage-2 evaluation subjects, in units of a
standard deviation of the feature.

What this does NOT measure: the KNNImputer's donor pool, which was also fit on
the full training split. Measuring that requires re-imputing. Missingness is
under 5% of cells, so the scaler channel is the larger of the two by cell
count.

Usage (server, from ~/TCG-Net, env tcg_net):

    python -u fusion/data/check_scaler_leakage.py

Prints aggregate statistics only — no individual participant values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PROTEOMICS_PKG = _REPO_ROOT / "proteomics"
if str(_PROTEOMICS_PKG) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG))

from config import SHARED_DATA_DIR, SPLIT_SEED  # noqa: E402

PROCESSED_TENSOR_PATH = SHARED_DATA_DIR / "proteomics_tensor.pt"
STAGE1_EXCLUDE_EIDS_PATH = SHARED_DATA_DIR / f"stage1_exclude_eids_{SPLIT_SEED}.npy"

PERCENTILES = (50.0, 90.0, 99.0, 99.9)


def summarise(name: str, values: np.ndarray, unit: str = "SD") -> None:
    """Print percentiles and max of ``|values|`` — aggregates only."""
    v = np.abs(np.asarray(values, dtype=np.float64).ravel())
    pcts = np.percentile(v, PERCENTILES)
    body = "  ".join(
        f"p{p:g}={q:.3e}" for p, q in zip(PERCENTILES, pcts)
    )
    print(f"{name:<34} n={v.size:>10,d}  {body}  max={v.max():.3e}  [{unit}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor", type=Path, default=PROCESSED_TENSOR_PATH,
        help="Path to proteomics_tensor.pt.",
    )
    parser.add_argument(
        "--exclude-eids", type=Path, default=STAGE1_EXCLUDE_EIDS_PATH,
        dest="exclude_eids",
        help="Path to the joint val/test EID list written by build_joint_dataset.py.",
    )
    args = parser.parse_args()

    bundle = torch.load(args.tensor, map_location="cpu", weights_only=False)
    x = bundle["x"].numpy().astype(np.float64)          # already z-scored
    eid = bundle["eid"].numpy().astype(np.int64)
    train_idx = bundle["train_idx"].numpy().astype(np.int64)

    exclude_eids = np.load(args.exclude_eids).astype(np.int64)

    banned_rows = np.flatnonzero(np.isin(eid, exclude_eids))
    banned_train = np.isin(eid[train_idx], exclude_eids)
    fit_idx = train_idx[~banned_train]

    print(f"tensor:          {args.tensor}")
    print(f"exclusion list:  {args.exclude_eids}")
    print(f"rows:            {x.shape[0]:,d} x {x.shape[1]:,d} features")
    print(f"train rows:      {train_idx.size:,d}")
    print(
        f"joint eval EIDs: {exclude_eids.size:,d} listed, "
        f"{banned_rows.size:,d} present in the proteomics cohort, "
        f"{int(banned_train.sum()):,d} of those in the train split "
        f"({100.0 * banned_train.mean():.2f}% of train)"
    )

    already_clean = bundle.get("fit_excluded_eids")
    if already_clean is not None and int(np.asarray(already_clean).size) > 0:
        print(
            f"NOTE: this tensor already records {int(np.asarray(already_clean).size):,d} "
            "EIDs held out of the fit — the shift below should be ~0."
        )
    if int(banned_train.sum()) == 0:
        print("Nothing to measure: no joint eval participant is in the train split.")
        return

    # The scaler that would have been fit on the clean subset, expressed in the
    # current z coordinates.
    m = x[fit_idx].mean(axis=0)
    s = x[fit_idx].std(axis=0, ddof=0)
    if not np.all(s > 0):
        raise RuntimeError("A feature has zero variance on the clean fit set.")

    print()
    print("Per-feature change in the normalisation statistics:")
    summarise("  |mean shift|", m)
    summarise("  |std ratio - 1|", s - 1.0, unit="rel")

    # Effect on the inputs the frozen encoder would see for Stage-2 eval rows.
    z_eval = x[banned_rows]
    dz = (z_eval - m) / s - z_eval
    print()
    print("Change in encoder input for Stage-2 evaluation participants:")
    summarise("  |dz|", dz)
    frac = float((np.abs(dz) > 0.01).mean())
    print(f"  cells shifted by more than 0.01 SD: {100.0 * frac:.4f}%")

    worst = float(np.abs(dz).max())
    print()
    if worst < 0.01:
        print(
            "VERDICT: negligible. The largest input change anywhere is "
            f"{worst:.2e} SD. Re-running preprocess.py --exclude-eids would not "
            "move a Stage-2 number; record the measurement and close item 6."
        )
    elif worst < 0.05:
        print(
            f"VERDICT: small ({worst:.2e} SD worst case). Fix it on the next "
            "preprocessing run rather than triggering one for this alone."
        )
    else:
        print(
            f"VERDICT: material ({worst:.2e} SD worst case). Re-run "
            "preprocess.py --exclude-eids, then Stage 1 and Stage 2."
        )


if __name__ == "__main__":
    main()
