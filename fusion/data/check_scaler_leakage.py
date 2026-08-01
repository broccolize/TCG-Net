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

PERCENTILES = (50.0, 90.0, 99.0, 99.9, 99.99)


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
    #
    #     dz = (z - m)/s - z = -m/s + z * (1/s - 1)
    #
    # The first term is the systematic offset every cell of a feature receives.
    # The second is proportional to the cell's own z, so the extreme tail of
    # |dz| is set by extreme protein values (|z| in the tens) meeting a small
    # relative std change — it measures outliers, not leakage. Judge the channel
    # on the offset and on a high quantile, never on the max.
    z_eval = x[banned_rows]
    dz = (z_eval - m) / s - z_eval
    print()
    print("Change in encoder input for Stage-2 evaluation participants:")
    summarise("  |systematic offset| (-m/s, per feature)", m / s)
    summarise("  |dz| (all cells)", dz)
    for thresh in (0.01, 0.05, 0.10):
        frac = float((np.abs(dz) > thresh).mean())
        print(f"  cells shifted by more than {thresh:.2f} SD: {100.0 * frac:.4f}%")

    offset = float(np.abs(m / s).max())
    tail = float(np.percentile(np.abs(dz), 99.9))
    worst = float(np.abs(dz).max())
    print()
    print(
        f"(worst single cell: {worst:.2e} SD — reported for completeness only; "
        "see the note above on why it is not the deciding statistic)"
    )
    if offset < 0.02 and tail < 0.05:
        print(
            f"VERDICT: negligible. Largest systematic offset {offset:.2e} SD, "
            f"p99.9 input change {tail:.2e} SD — both well under the seed noise "
            "of the Stage-1/Stage-2 metrics. Keep the --exclude-eids fix for the "
            "next preprocessing run, but do not trigger one for this alone."
        )
    elif offset < 0.05 and tail < 0.15:
        print(
            f"VERDICT: small (offset {offset:.2e} SD, p99.9 {tail:.2e} SD). Fold "
            "the fix into the next preprocessing run rather than triggering one."
        )
    else:
        print(
            f"VERDICT: material (offset {offset:.2e} SD, p99.9 {tail:.2e} SD). "
            "Re-run preprocess.py --exclude-eids, then Stage 1 and Stage 2."
        )


if __name__ == "__main__":
    main()
