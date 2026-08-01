"""
Aggregate the per-participant Shapley contributions across Stage-2 runs.

This is the analysis for the model's interpretability output — spec sections 4.2
(individual-level contributions), 4.3 (the gate they drive), 11 (interpretable
quantities) and 12 (mechanistic subtyping).

It reads the `shapley_<target>_<run_id>.npz` files written by train_stage2.py at
test time, so every number here comes from the held-out fold of a converged model
and can be aggregated over seeds rather than read off a single run.

Three things it checks that are easy to get wrong
-------------------------------------------------
1. **phi_int must not be degenerate.** For two players the Shapley efficiency
   axiom gives phi_f + phi_p = y_fp - y_0 identically, so defining phi_int as the
   residual of that identity yields exactly zero for every participant. The model
   uses the 2-way interaction index instead. This script reports the residual
   alongside phi_int so the distinction is visible rather than assumed.

2. **The gate weights are not contribution measures.** w comes from a softmax over
   the phi values; a branch that predicts nothing can still draw a high weight.
   Reported next to each branch's actual correlation with the target so the two
   are never conflated.

3. **The subtyping score simplifies.** Section 12 defines D = phi_f - phi_p, which
   substituting 4.2 reduces exactly to y_hat_f - y_hat_p. Verified numerically —
   if it holds, the subtyping needs no Shapley machinery and the interpretability
   claim has to be worded accordingly.

Usage (server, from ~/TCG-Net, env tcg_net):

    python -u fusion/analysis/interpret_shapley.py
    python -u fusion/analysis/interpret_shapley.py --target age

Prints aggregate statistics only — no individual participant values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
CHECKPOINT_DIR = _REPO_ROOT / "fusion" / "training" / "checkpoints"

PCT = (5, 25, 50, 75, 95)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dist_row(name: str, v: np.ndarray) -> str:
    q = np.percentile(v, PCT)
    return (f"  {name:<10}{v.mean():>+9.4f}{v.std(ddof=1):>9.4f}"
            + "".join(f"{x:>+9.4f}" for x in q))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--target", default="fluid_iq")
    args = p.parse_args()

    files = sorted(args.dir.glob(f"shapley_{args.target}_*.npz"))
    if not files:
        print(f"No shapley_{args.target}_*.npz under {args.dir}.\n"
              "Re-run train_stage2.py — the dump was added 2026-08-01, so runs "
              "from before then have no Shapley output saved.")
        return

    runs = [np.load(f) for f in files]
    print(f"{len(runs)} run(s) — seeds {[int(r['seed']) for r in runs]}, "
          f"n_test = {len(runs[0]['y'])}\n")

    # ---- 1. contribution distributions, pooled over runs -------------------
    print("Individual-level Shapley contributions (pooled over runs)")
    print(f"  {'':<10}{'mean':>9}{'sd':>9}" + "".join(f"{'p'+str(q):>9}" for q in PCT))
    for key in ("phi_f", "phi_p", "phi_int"):
        print(dist_row(key, np.concatenate([r[key] for r in runs])))

    # phi_int against the degenerate definition it replaced
    resid = np.concatenate([
        r["y_hat_fp"] - r["y_baseline"] - r["phi_f"] - r["phi_p"] for r in runs
    ])
    pi = np.concatenate([r["phi_int"] for r in runs])
    print(f"\n  efficiency residual  y_fp - y_0 - phi_f - phi_p:  "
          f"max |.| = {np.abs(resid).max():.3e}   <- zero by construction")
    print(f"  interaction index    y_fp - y_f - y_p + y_0:      "
          f"sd = {pi.std(ddof=1):.4f}, range {pi.min():+.3f} to {pi.max():+.3f}")
    if np.abs(pi).max() < 1e-6:
        print("  ! phi_int is degenerate — it has been rewritten as the residual.")

    # ---- 2. gate weights vs actual branch skill ----------------------------
    print("\nGate weights (softmax over the contributions) vs branch skill")
    print(f"  {'branch':<12}{'mean w':>9}{'sd w':>9}{'min w':>9}{'max w':>9}"
          f"{'branch r':>11}")
    w = np.concatenate([r["w"] for r in runs], axis=0)
    y = np.concatenate([r["y"] for r in runs])
    skills = {
        "fMRI": corr(np.concatenate([r["y_hat_f"] for r in runs]), y),
        "proteomics": corr(np.concatenate([r["y_hat_p"] for r in runs]), y),
        "interaction": float("nan"),
    }
    for i, (label, sk) in enumerate(skills.items()):
        sk_s = "—" if np.isnan(sk) else f"{sk:+.4f}"
        print(f"  {label:<12}{w[:, i].mean():>9.3f}{w[:, i].std(ddof=1):>9.3f}"
              f"{w[:, i].min():>9.3f}{w[:, i].max():>9.3f}{sk_s:>11}")
    print("  Note: w is a softmax over phi, not a measure of predictive skill. A "
          "branch\n  can hold a high weight while correlating near zero with the "
          "target.")
    if w.std(ddof=1) < 0.02:
        print("  ! gate is effectively frozen — weights barely vary across participants.")

    # ---- 3. section-12 subtyping ------------------------------------------
    print("\nSection-12 mechanistic subtyping   D = phi_f - phi_p")
    d_alg = np.concatenate([r["phi_f"] - r["phi_p"] for r in runs])
    d_sim = np.concatenate([r["y_hat_f"] - r["y_hat_p"] for r in runs])
    gap = float(np.abs(d_alg - d_sim).max())
    print(f"  max |(phi_f - phi_p) - (y_hat_f - y_hat_p)| = {gap:.3e}")
    print("  -> the subtyping score reduces to the difference of the two "
          "single-modality\n     predictions; the Shapley terms cancel algebraically."
          if gap < 1e-4 else
          "  -> the two differ; the subtyping is not a plain branch difference.")

    # Is the subtype-target association mechanical? D is dominated by y_hat_f,
    # which is itself the stronger predictor of y, so selecting a high D partly
    # selects a high y_hat_f and therefore a high y. If corr(D, y) is comparable
    # to corr(y_hat_f, y), the group differences below are that selection effect
    # and not a mechanistic distinction between participants.
    d_all = np.concatenate([r["phi_f"] - r["phi_p"] for r in runs])
    yf_all = np.concatenate([r["y_hat_f"] for r in runs])
    yp_all = np.concatenate([r["y_hat_p"] for r in runs])
    print("\n  Is the subtype-target link mechanical?")
    print(f"    corr(D, target)      = {corr(d_all, y):+.4f}")
    print(f"    corr(y_hat_f, target) = {corr(yf_all, y):+.4f}   "
          f"corr(D, y_hat_f) = {corr(d_all, yf_all):+.4f}")
    print(f"    corr(y_hat_p, target) = {corr(yp_all, y):+.4f}   "
          f"corr(D, y_hat_p) = {corr(d_all, yp_all):+.4f}")
    if abs(corr(d_all, y)) > 0.5 * abs(corr(yf_all, y)):
        print("    -> D tracks the target largely because it tracks the stronger "
              "branch;\n       report the group differences as a selection effect, "
              "not a mechanism.")

    print(f"\n  {'subtype':<18}{'n':>8}{'share':>9}{'mean target':>13}")
    for r in runs[:1]:  # per-run so the counts are a real cohort, not a pool
        d = r["phi_f"] - r["phi_p"]
        dz = (d - d.mean()) / d.std(ddof=0)
        groups = [("brain-dominant", dz > 0.5), ("balanced", np.abs(dz) <= 0.5),
                  ("blood-dominant", dz < -0.5)]
        for name, mask in groups:
            n = int(mask.sum())
            mt = float(r["y"][mask].mean()) if n else float("nan")
            print(f"  {name:<18}{n:>8,d}{100.0 * n / dz.size:>8.1f}%{mt:>13.4f}")
        print(f"  (seed {int(r['seed'])}; thresholding a z-score at ±0.5 gives "
              "≈31/38/31% for a normal\n   variable, so near-thirds is the "
              "expected split, not evidence of structure)")


if __name__ == "__main__":
    main()
