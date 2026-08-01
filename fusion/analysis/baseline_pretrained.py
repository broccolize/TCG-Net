"""
Fair linear baselines: ridge with the SAME pretraining advantage as the encoders.

The problem this fixes
----------------------
`baseline_joint.py` fits ridge on the 3,471–3,730 joint training subjects, while
the Stage-2 encoders were pretrained on tens of thousands. On age, the same
modality scores 0.774 with 3,730 training rows and 0.868 with 36,888 — a ~0.09 R²
swing from data volume alone. So "the deep model is 1.79× the best ridge" mixes
two effects: architecture, and a 10× larger pretraining cohort. Only ridge was
denied the second one.

This script gives ridge the same deal:

    1. Fit ridge on the FULL single-modality cohort, with every joint val/test
       participant removed from both the train and the alpha-selection folds, so
       the joint test set stays untouched.
    2. Apply it to the joint participants.
    3. Refit an affine map (a·p + b) on the joint TRAIN fold only — mirroring
       Stage 2, which keeps the pretrained encoder and retrains its heads on
       joint train. This also absorbs the instance-0 → instance-2 label shift on
       the proteomics side.
    4. Score once on the joint test fold.

Models produced:

    ridge_prot_pre    full proteomics cohort (~36.9k) -> joint test
    ridge_fmri_pre    full connectome cohort (~40.5k) -> joint test
    late_fusion_pre   both pretrained predictions, combined on joint train (M4)

Read it against the Stage-2 fused row for the same target. Whatever margin
survives *this* comparison is what the architecture is actually worth; the
difference between this and `baseline_joint.py` is what pretraining is worth.
Both are legitimate contributions — they are just different claims.

Usage (server, from ~/TCG-Net, env tcg_net):

    python -u fusion/analysis/baseline_pretrained.py --target age
    python -u fusion/analysis/baseline_pretrained.py --target fluid_iq
    python -u fusion/analysis/baseline_pretrained.py --target age --skip-fmri

The fMRI arm holds ~40.5k × 2,145 in memory (≈700 MB as float64). Use
--skip-fmri on a constrained box.

Prints aggregate metrics only — no individual participant values.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (_REPO_ROOT / "proteomics", _REPO_ROOT / "fusion" / "data"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import (  # noqa: E402
    DATASET_AGE_PATH,
    DATASET_IQ_PATH,
    GLOBAL_SEED,
    SHARED_DATA_DIR,
)
from build_joint_dataset import (  # noqa: E402
    CORR_MATRIX_PATH,
    TARGET_CONFIG,
    discover_target_file,
)

ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
VAL_FRAC_FMRI = 0.15


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R², MAE, Pearson r — SST against the evaluation set's own mean."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    r = (float(np.corrcoef(y_pred, y_true)[0, 1])
         if y_pred.std() > 0 and y_true.std() > 0 else float("nan"))
    return {"r2": r2, "mae": float(np.abs(y_true - y_pred).mean()), "pearson_r": r}


def fit_affine(p: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Least-squares a, b minimising ||t - (a*p + b)||^2."""
    p = np.asarray(p, float).ravel(); t = np.asarray(t, float).ravel()
    var = float(((p - p.mean()) ** 2).sum())
    if var <= 0:
        return 0.0, float(t.mean())
    a = float(((p - p.mean()) * (t - t.mean())).sum() / var)
    return a, float(t.mean() - a * p.mean())


def select_alpha(x_tr, y_tr, x_va, y_va, label: str):
    """Fit ridge over the alpha grid, keep the best validation R²."""
    from sklearn.linear_model import Ridge
    best = None
    for a in ALPHAS:
        t0 = time.time()
        m = Ridge(alpha=a).fit(x_tr, y_tr)
        v = metrics(y_va, m.predict(x_va))
        print(f"    {label} alpha={a:<8g} val r2={v['r2']:+.4f}  ({time.time() - t0:.1f}s)")
        if best is None or v["r2"] > best[1]["r2"]:
            best = (m, v, a)
    return best


def adapt_and_score(model, x_joint, yz, tr, te, name: str, results: dict, extra: dict):
    """Affine-adapt the pretrained model's output on joint train, score on joint test."""
    pred = model.predict(x_joint)
    a, b = fit_affine(pred[tr], yz[tr])
    results[name] = metrics(yz[te], a * pred[te] + b)
    results[name].update(extra, adapt_scale=a)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("age", "fluid_iq"), default="age")
    parser.add_argument("--skip-fmri", action="store_true", dest="skip_fmri")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    log = logging.getLogger("baseline_pretrained")
    logging.basicConfig(level=logging.INFO, format="    %(message)s")
    rng = np.random.default_rng(args.seed)

    # ---- joint dataset: defines the evaluation set and the excluded EIDs ----
    joint_path = SHARED_DATA_DIR / f"dataset_joint_{args.target}.pt"
    jb = torch.load(joint_path, map_location="cpu", weights_only=False)
    j_eid = jb["eid"].numpy().astype(np.int64)
    y = jb["y"].numpy().astype(np.float64).ravel()
    tr = jb["train_idx"].numpy(); va = jb["val_idx"].numpy(); te = jb["test_idx"].numpy()
    mu, sd = y[tr].mean(), y[tr].std(ddof=0)
    yz = (y - mu) / sd

    # Anything the joint model is validated or tested on must never train a
    # baseline either, or the comparison is rigged in the baseline's favour.
    banned = np.concatenate([j_eid[va], j_eid[te]])

    print(f"target:  {args.target}   joint dataset: {joint_path}")
    print(f"joint:   train={tr.size:,d}  val={va.size:,d}  test={te.size:,d}")
    print(f"held out of every baseline fit: {banned.size:,d} EIDs\n")

    results: dict[str, dict] = {
        "mean": metrics(yz[te], np.full(te.size, yz[tr].mean()))
    }
    preds: dict[str, np.ndarray] = {}

    # ---- proteomics arm: full cohort ---------------------------------------
    full_path = DATASET_AGE_PATH if args.target == "age" else DATASET_IQ_PATH
    fb = torch.load(full_path, map_location="cpu", weights_only=False)
    f_x = fb["x"].numpy().astype(np.float64)
    f_y = fb["y"].numpy().astype(np.float64).ravel()
    f_eid = fb["eid"].numpy().astype(np.int64)
    keep = ~np.isin(f_eid, banned)
    f_tr = fb["train_idx"].numpy(); f_va = fb["val_idx"].numpy()
    f_tr = f_tr[keep[f_tr]]; f_va = f_va[keep[f_va]]

    print(f"  proteomics cohort: {f_eid.size:,d} subjects, fitting on "
          f"{f_tr.size:,d} train / {f_va.size:,d} val after exclusion")
    print(f"    (label column is instance-0 here; the joint target is "
          f"instance-2 — the affine adaptation absorbs the shift)")
    m, v, alpha = select_alpha(f_x[f_tr], f_y[f_tr], f_x[f_va], f_y[f_va], "prot")
    preds["prot"] = adapt_and_score(
        m, jb["x_p"].numpy().astype(np.float64), yz, tr, te,
        "ridge_prot_pre", results, {"alpha": alpha, "n_fit": int(f_tr.size)},
    )

    # ---- fMRI arm: full connectome cohort ----------------------------------
    if not args.skip_fmri:
        print(f"\n  connectome cohort: loading {CORR_MATRIX_PATH}")
        mat = sio.loadmat(str(CORR_MATRIX_PATH))
        fc_ids = np.array(mat["Patient_IDs"]).squeeze().astype(np.int64)
        tri_i, tri_j = np.triu_indices(66, k=1)
        fc = mat["FC_Matrices"][:, tri_i, tri_j].astype(np.float64)
        del mat

        # Labels for the full FC cohort, at the same instance the joint model uses.
        column = TARGET_CONFIG[args.target]["column"]
        import pandas as pd
        df = pd.read_csv(discover_target_file(column, log), usecols=["eid", column]).dropna()
        lut = dict(zip(df["eid"].to_numpy(np.int64), df[column].to_numpy(np.float64)))
        has = np.array([e in lut for e in fc_ids])
        usable = np.flatnonzero(has & ~np.isin(fc_ids, banned))
        rng.shuffle(usable)
        n_val = int(round(usable.size * VAL_FRAC_FMRI))
        g_va, g_tr = usable[:n_val], usable[n_val:]
        # Only rows in `usable` are ever indexed, and those are guaranteed to be
        # in the lookup — but the array spans the whole cohort, so participants
        # without the target need a placeholder rather than a KeyError. Age is
        # complete on the imaging cohort (40,517/40,517); fluid IQ is not.
        g_y = np.array([lut.get(int(e), np.nan) for e in fc_ids], dtype=np.float64)

        print(f"  connectome cohort: {fc_ids.size:,d} subjects, {int(has.sum()):,d} "
              f"with '{column}', fitting on {g_tr.size:,d} train / {g_va.size:,d} val "
              f"after exclusion")
        m, v, alpha = select_alpha(fc[g_tr], g_y[g_tr], fc[g_va], g_y[g_va], "fmri")
        preds["fmri"] = adapt_and_score(
            m, jb["x_f"].numpy().astype(np.float64), yz, tr, te,
            "ridge_fmri_pre", results, {"alpha": alpha, "n_fit": int(g_tr.size)},
        )
        del fc

    # ---- late fusion of the two pretrained predictions (M4) ----------------
    if "prot" in preds and "fmri" in preds:
        from sklearn.linear_model import LinearRegression
        z = np.column_stack([preds["prot"], preds["fmri"]])
        lf = LinearRegression().fit(z[tr], yz[tr])
        results["late_fusion_pre"] = metrics(yz[te], lf.predict(z[te]))
        results["late_fusion_pre"]["weights"] = [float(w) for w in lf.coef_]

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"JOINT TEST SET — {args.target}  (n={te.size:,d})   PRETRAINED BASELINES")
    print("=" * 60)
    print(f"{'model':<20}{'R2':>10}{'MAE':>10}{'pearson r':>12}")
    for k, mm in results.items():
        print(f"{k:<20}{mm['r2']:>+10.4f}{mm['mae']:>10.4f}{mm['pearson_r']:>+12.4f}")
    print("=" * 60)
    print("These baselines saw the same large pretraining cohorts as the encoders.\n"
          "Margin over THESE rows = what the architecture buys.\n"
          "Gap vs baseline_joint.py's rows = what pretraining buys.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = _REPO_ROOT / "fusion" / "logs" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"baseline_pretrained_{args.target}_{stamp}.json").write_text(
        json.dumps({"timestamp": stamp, "target": args.target,
                    "joint_dataset": str(joint_path), "n_test": int(te.size),
                    "n_excluded_eids": int(banned.size), "results": results},
                   indent=2), encoding="utf-8",
    )
    print(f"\nwritten: {out / f'baseline_pretrained_{args.target}_{stamp}.json'}")


if __name__ == "__main__":
    main()
