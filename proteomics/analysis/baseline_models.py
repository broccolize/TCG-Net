"""
Classical baselines for the proteomics Stage-1 targets.

Why this exists
---------------
Stage 1 reports test R² 0.868 for age and 0.107 for fluid IQ. Neither number
says anything about the *encoder* until a model with no learned architecture is
scored on the same split. Plasma proteomic age clocks built with plain
regularised linear regression on an Olink panel are strong, so "R² 0.868" is
entirely compatible with "age is linearly decodable from 2,920 proteins and the
SE-gated MLP contributed nothing". This script settles that.

The comparison is only valid if nothing differs but the model, so this loads the
same dataset bundle Stage 1 loads, honours the same ``--exclude-eids`` list, and
computes R² the same way ``train_stage1.evaluate`` does (SST against the
evaluation set's own mean). Hyperparameters are chosen on the validation fold and
the test fold is touched once, at the end.

Ladder
------
    mean          predict the training mean            — the floor, R² ≈ 0
    ridge         L2 on all proteins, alpha on val     — the baseline that matters
    top-k ridge   k proteins by |r| on train           — how much rides on a handful
    lasso         L1, sparse (--with-lasso)            — the proteomic-clock standard
    gbm           HistGradientBoosting (--with-gbm)    — is there nonlinearity to find

Read the result as: if ``ridge`` matches Stage 1, the encoder is decoration. If
Stage 1 clears ridge by more than seed noise (±0.005–0.009 R² on this pipeline),
the architecture is earning its place.

Usage (server, from ~/TCG-Net, env tcg_net):

    python -u proteomics/analysis/baseline_models.py --target age
    python -u proteomics/analysis/baseline_models.py --target fluid_iq

Add ``--exclude-eids /shared/tcgnet_data_2/stage1_exclude_eids_1234.npy`` to
match a leakage-free Stage-1 run. Compare only against a Stage-1 run made with
the same setting.

Prints aggregate metrics only — no individual participant values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from config import (  # noqa: E402
    DATASET_AGE_PATH,
    DATASET_IQ_PATH,
    GLOBAL_SEED,
    LOG_DIR_EXPERIMENT,
)

RIDGE_ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
LASSO_ALPHAS = (1e-3, 1e-2, 1e-1, 1e0)
TOP_K = (10, 50, 200)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    R², MAE and Pearson r, computed exactly as train_stage1.evaluate does:
    SST uses the *evaluation set's* mean, not the training mean.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.abs(y_true - y_pred).mean())
    if y_pred.std() > 0 and y_true.std() > 0:
        r = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        r = float("nan")
    return {"r2": r2, "mae": mae, "pearson_r": r}


def apply_exclusion(
    eid: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, path: Path | None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop Stage-2 evaluation participants from train and val, as Stage 1 does."""
    if path is None:
        return train_idx, val_idx, {"applied": False}
    if not path.exists():
        raise FileNotFoundError(f"--exclude-eids file not found: {path}")

    exclude = np.load(path).astype(np.int64)
    banned = np.flatnonzero(np.isin(eid, exclude))
    train_kept = train_idx[~np.isin(train_idx, banned)]
    val_kept = val_idx[~np.isin(val_idx, banned)]
    stats = {
        "applied": True,
        "file": str(path),
        "listed": int(exclude.size),
        "matched": int(banned.size),
        "train_removed": int(train_idx.size - train_kept.size),
        "val_removed": int(val_idx.size - val_kept.size),
    }
    return train_kept, val_kept, stats


def select_on_val(make_model, grid, x_tr, y_tr, x_va, y_va, label: str) -> tuple:
    """Fit ``make_model(p)`` for each p in ``grid``; keep the best val R²."""
    best = None
    for p in grid:
        t0 = time.time()
        model = make_model(p)
        model.fit(x_tr, y_tr)
        val = metrics(y_va, model.predict(x_va))
        print(
            f"    {label}={p:<8g} val r2={val['r2']:+.4f} "
            f"mae={val['mae']:.4f}  ({time.time() - t0:.1f}s)"
        )
        if best is None or val["r2"] > best[1]["r2"]:
            best = (model, val, p)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("age", "fluid_iq"), required=True)
    parser.add_argument(
        "--exclude-eids", type=Path, default=None, dest="exclude_eids",
        help="Joint val/test EID list, to match a leakage-free Stage-1 run.",
    )
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--with-lasso", action="store_true", dest="with_lasso")
    parser.add_argument("--with-gbm", action="store_true", dest="with_gbm")
    args = parser.parse_args()

    # Imported here so --help works without sklearn present.
    from sklearn.linear_model import Lasso, Ridge

    np.random.seed(args.seed)

    path = DATASET_AGE_PATH if args.target == "age" else DATASET_IQ_PATH
    bundle = torch.load(path, map_location="cpu", weights_only=False)

    x = bundle["x"].numpy().astype(np.float64)
    y = bundle["y"].numpy().astype(np.float64).ravel()
    eid = bundle["eid"].numpy().astype(np.int64)
    train_idx = bundle["train_idx"].numpy().astype(np.int64)
    val_idx = bundle["val_idx"].numpy().astype(np.int64)
    test_idx = bundle["test_idx"].numpy().astype(np.int64)

    train_idx, val_idx, excl = apply_exclusion(
        eid, train_idx, val_idx, args.exclude_eids
    )

    print(f"dataset:  {path}")
    print(f"target:   {args.target} ({bundle.get('target_name')})")
    print(f"features: {x.shape[1]:,d}")
    print(
        f"split:    train={train_idx.size:,d}  val={val_idx.size:,d}  "
        f"test={test_idx.size:,d}"
    )
    if excl["applied"]:
        print(
            f"exclusion: {excl['train_removed']:,d} dropped from train, "
            f"{excl['val_removed']:,d} from val ({excl['file']})"
        )
    else:
        print(
            "exclusion: none — compare only against a Stage-1 run that also "
            "ran without --exclude-eids"
        )
    print()

    x_tr, y_tr = x[train_idx], y[train_idx]
    x_va, y_va = x[val_idx], y[val_idx]
    x_te, y_te = x[test_idx], y[test_idx]

    results: dict[str, dict] = {}

    # --- floor -------------------------------------------------------------
    results["mean"] = metrics(y_te, np.full_like(y_te, y_tr.mean()))
    print(f"  mean predictor           test r2={results['mean']['r2']:+.4f}")

    # --- ridge on all proteins ---------------------------------------------
    print("\n  ridge (all proteins):")
    model, val, alpha = select_on_val(
        lambda a: Ridge(alpha=a), RIDGE_ALPHAS, x_tr, y_tr, x_va, y_va, "alpha"
    )
    results["ridge"] = metrics(y_te, model.predict(x_te))
    results["ridge"]["alpha"] = alpha
    results["ridge"]["val_r2"] = val["r2"]

    # --- top-k univariate screen, then ridge -------------------------------
    # Screening uses TRAIN correlations only; picking k on val, not test.
    print("\n  top-k ridge (proteins ranked by |r| on train):")
    yc = y_tr - y_tr.mean()
    xc = x_tr - x_tr.mean(axis=0)
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r_uni = np.where(denom > 0, (xc * yc[:, None]).sum(axis=0) / denom, 0.0)
    order = np.argsort(-np.abs(r_uni))

    best_topk = None
    for k in TOP_K:
        cols = order[:k]
        sub, val_k, alpha_k = select_on_val(
            lambda a: Ridge(alpha=a), RIDGE_ALPHAS,
            x_tr[:, cols], y_tr, x_va[:, cols], y_va, f"k={k} alpha",
        )
        if best_topk is None or val_k["r2"] > best_topk[1]["r2"]:
            best_topk = (sub, val_k, (k, alpha_k), cols)
    sub, val_k, (k, alpha_k), cols = best_topk
    results["ridge_topk"] = metrics(y_te, sub.predict(x_te[:, cols]))
    results["ridge_topk"].update(k=k, alpha=alpha_k, val_r2=val_k["r2"])

    # --- lasso -------------------------------------------------------------
    if args.with_lasso:
        print("\n  lasso (coordinate descent — the slow one):")
        model, val, alpha = select_on_val(
            lambda a: Lasso(alpha=a, max_iter=5000, tol=1e-3),
            LASSO_ALPHAS, x_tr, y_tr, x_va, y_va, "alpha",
        )
        results["lasso"] = metrics(y_te, model.predict(x_te))
        results["lasso"].update(
            alpha=alpha, val_r2=val["r2"],
            n_nonzero=int((model.coef_ != 0).sum()),
        )

    # --- gradient boosting -------------------------------------------------
    if args.with_gbm:
        from sklearn.ensemble import HistGradientBoostingRegressor
        print("\n  hist gradient boosting:")
        t0 = time.time()
        gbm = HistGradientBoostingRegressor(
            max_iter=400, early_stopping=True, random_state=args.seed
        )
        gbm.fit(x_tr, y_tr)
        val = metrics(y_va, gbm.predict(x_va))
        print(f"    val r2={val['r2']:+.4f}  ({time.time() - t0:.1f}s)")
        results["gbm"] = metrics(y_te, gbm.predict(x_te))
        results["gbm"]["val_r2"] = val["r2"]

    # --- report ------------------------------------------------------------
    print("\n" + "=" * 62)
    print(f"TEST-SET RESULTS — {args.target}  (n={test_idx.size:,d})")
    print("=" * 62)
    print(f"{'model':<16}{'R2':>10}{'MAE':>10}{'pearson r':>12}")
    for name, m in results.items():
        print(
            f"{name:<16}{m['r2']:>+10.4f}{m['mae']:>10.4f}{m['pearson_r']:>+12.4f}"
        )
    print("=" * 62)
    print(
        "Compare the best row against the Stage-1 encoder on this same target "
        "and split. A gap inside +/-0.009 R2 is seed noise, not an architecture."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR_EXPERIMENT / f"baselines_{args.target}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "target": args.target,
                "dataset": str(path),
                "seed": args.seed,
                "n_features": int(x.shape[1]),
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "n_test": int(test_idx.size),
                "exclusion": excl,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
