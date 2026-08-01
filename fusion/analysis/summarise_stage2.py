"""
Compact table over the Stage-2 test-metric JSONs, plus paired branch contrasts.

`cat`-ing these files by hand is how the 2026-08-01 confusion happened: the
truncated and converged runs sit in the same directory, differ only by
`best_epoch`, and a `tail` that clips one record silently drops a seed from the
comparison. This reads every file, groups the runs, and does the arithmetic.

The contrasts are paired within a seed — same test subjects, same split, only
the initialisation differs — which is the only comparison these runs support.
Across-seed spread is initialisation noise, NOT sampling error on the test set;
at n=747 the sampling uncertainty on any single R² is far wider. Quote the
contrasts, never the raw R² to three decimals.

Usage (server, from ~/TCG-Net, env tcg_net):

    python -u fusion/analysis/summarise_stage2.py
    python -u fusion/analysis/summarise_stage2.py --min-epoch 30   # converged only
    python -u fusion/analysis/summarise_stage2.py --target age
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PROTEOMICS_PKG = _REPO_ROOT / "proteomics"
if str(_PROTEOMICS_PKG) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG))

CHECKPOINT_DIR = _REPO_ROOT / "fusion" / "training" / "checkpoints"

BRANCHES = [
    ("test_r2", "fused"),
    ("test_r2_fmri_only", "fMRI-only"),
    ("test_r2_prot_only", "prot-only"),
    ("test_r2_joint_head", "joint-head"),
]
CONTRASTS = [
    ("test_r2", "test_r2_fmri_only", "fused - fMRI-only"),
    ("test_r2_joint_head", "test_r2", "joint-head - fused"),
    ("test_r2_joint_head", "test_r2_fmri_only", "joint-head - fMRI-only"),
]


def load(directory: Path, target: str) -> list[dict]:
    runs = []
    for path in sorted(directory.glob(f"test_metrics_{target}_*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            print(f"  ! skipping unreadable {path.name}: {exc}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--target", default="fluid_iq")
    parser.add_argument(
        "--min-epoch", type=int, default=0, dest="min_epoch",
        help="Keep only runs whose best_epoch is at least this. Use it to "
             "separate converged runs from ones truncated by --phase2-epoch.",
    )
    args = parser.parse_args()

    runs = load(args.dir, args.target)
    if not runs:
        print(f"No test_metrics_{args.target}_*.json under {args.dir}")
        return

    kept = [r for r in runs if int(r.get("best_epoch", 0)) >= args.min_epoch]
    print(f"{len(runs)} run(s) found, {len(kept)} kept (best_epoch >= {args.min_epoch})\n")

    hdr = (
        f"{'run_id':<26}{'seed':>5}{'ep':>5}{'wd':>8}"
        f"{'fused':>9}{'fMRI':>9}{'prot':>9}{'jhead':>9}{'r':>8}{'pstd':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in kept:
        print(
            f"{r.get('run_id','?'):<26}{r.get('seed','?'):>5}"
            f"{r.get('best_epoch','?'):>5}{r.get('weight_decay',float('nan')):>8g}"
            f"{r.get('test_r2', float('nan')):>+9.4f}"
            f"{r.get('test_r2_fmri_only', float('nan')):>+9.4f}"
            f"{r.get('test_r2_prot_only', float('nan')):>+9.4f}"
            f"{r.get('test_r2_joint_head', float('nan')):>+9.4f}"
            f"{r.get('test_pearson_r', float('nan')):>+8.4f}"
            f"{r.get('test_pred_std', float('nan')):>7.3f}"
        )

    have = [r for r in kept if "test_r2_joint_head" in r]
    if len(have) < 2:
        print("\nFewer than 2 runs carry the branch decomposition — no contrasts.")
        return

    print(f"\nbranch R2 across {len(have)} run(s)")
    print(f"  {'branch':<12}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}")
    for key, label in BRANCHES:
        v = np.array([r[key] for r in have], dtype=float)
        print(
            f"  {label:<12}{v.mean():>+10.4f}{v.std(ddof=1):>10.4f}"
            f"{v.min():>+10.4f}{v.max():>+10.4f}"
        )

    print("\npaired contrasts (per seed)")
    for a, b, label in CONTRASTS:
        d = np.array([r[a] - r[b] for r in have], dtype=float)
        pos = int((d > 0).sum())
        line = f"  {label:<22}{d.mean():>+9.4f} +/- {d.std(ddof=1):<8.4f} signs {pos}/{d.size}"
        if d.size >= 3:
            se = d.std(ddof=1) / np.sqrt(d.size)
            line += f"   t={d.mean()/se:+.2f} (df={d.size - 1})"
        print(line)
        print(f"    {'  '.join(f'{x:+.4f}' for x in d)}")

    # Val-fitted affine recalibration, if the runs carry it. This is the
    # reportable version of the r^2 ceiling printed below.
    cal = [r for r in have if "test_r2_cal" in r]
    if cal:
        print(f"\nval-calibrated R2 across {len(cal)} run(s)")
        print(f"  {'branch':<12}{'raw':>10}{'calibrated':>12}{'delta':>10}{'scale':>9}")
        for raw_k, cal_k, sc_k, label in (
            ("test_r2", "test_r2_cal", "cal_scale", "fused"),
            ("test_r2_fmri_only", "test_r2_cal_fmri_only", "cal_scale_fmri_only", "fMRI-only"),
            ("test_r2_prot_only", "test_r2_cal_prot_only", "cal_scale_prot_only", "prot-only"),
            ("test_r2_joint_head", "test_r2_cal_joint_head", "cal_scale_joint_head", "joint-head"),
        ):
            raw = np.array([r[raw_k] for r in cal], dtype=float)
            new = np.array([r[cal_k] for r in cal], dtype=float)
            sc = np.array([r[sc_k] for r in cal], dtype=float)
            print(
                f"  {label:<12}{raw.mean():>+10.4f}{new.mean():>+12.4f}"
                f"{(new - raw).mean():>+10.4f}{sc.mean():>9.3f}"
            )
        d = np.array(
            [r["test_r2_cal"] - r["test_r2_cal_fmri_only"] for r in cal], dtype=float
        )
        print(
            f"\n  calibrated fused - fMRI-only: {d.mean():+.4f} "
            f"+/- {d.std(ddof=1) if d.size > 1 else float('nan'):.4f}   "
            f"signs {int((d > 0).sum())}/{d.size}"
        )
        print(f"    {'  '.join(f'{x:+.4f}' for x in d)}")

    # Shrinkage diagnostic: r^2 is what an optimally rescaled predictor would
    # score. A large r^2 - R^2 gap means the ranking is there but the amplitude
    # is wrong. NB: r^2 is fitted on the test set, so it is an optimistic
    # ceiling, not a reportable number — refit the scale on val to claim it.
    print("\nshrinkage: r^2 - R^2  (optimistic ceiling — see note in source)")
    print(f"  {'seed':<6}{'branch':<12}{'r':>9}{'r^2':>9}{'R^2':>9}{'gap':>9}")
    pairs = [
        ("test_pearson_r", "test_r2", "fused"),
        ("test_pearson_fmri_only", "test_r2_fmri_only", "fMRI-only"),
        ("test_pearson_joint_head", "test_r2_joint_head", "joint-head"),
    ]
    for r in have:
        for rk, r2k, label in pairs:
            if rk not in r:
                continue
            rr, r2 = float(r[rk]), float(r[r2k])
            print(
                f"  {r.get('seed','?'):<6}{label:<12}{rr:>9.4f}"
                f"{rr ** 2:>9.4f}{r2:>9.4f}{rr ** 2 - r2:>+9.4f}"
            )


if __name__ == "__main__":
    main()
