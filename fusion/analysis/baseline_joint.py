"""
Classical baselines on the joint dataset — the comparison the whole project rests on.

Stage 2 reports fused R² ≈ 0.067 (0.077 calibrated) for fluid IQ. Ridge on the
raw 2,920 proteins matches the proteomics Stage-1 encoder exactly on age
(0.8690 vs 0.868, 2026-08-01), so the same question has to be asked of the joint
model: does the two-encoder architecture beat ridge on the concatenated raw
features of the same participants?

Four models, same joint split, same test subjects as Stage 2:

    ridge_fmri    2,145 FC edges                  -> is the GNN encoder earning its place?
    ridge_prot    2,920 proteins                  -> the proteomics ceiling on this cohort
    ridge_concat  both, 5,065 features            -> the no-fusion multimodal baseline
    mean          the floor

``ridge_concat`` is the one that matters. It is naive early fusion with no
gating, no Shapley values and no learned representation. If it matches the
Stage-2 fused output, the architecture is not what produces the number.

Alpha is chosen on the joint validation fold; the test fold is scored once. R²
uses SST against the evaluation set's own mean, matching train_stage2.evaluate,
so these numbers sit directly beside the Stage-2 table.

Usage (server, from ~/TCG-Net, env tcg_net):

    python -u fusion/analysis/baseline_joint.py --target fluid_iq

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

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PROTEOMICS_PKG = _REPO_ROOT / "proteomics"
if str(_PROTEOMICS_PKG) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG))

from config import GLOBAL_SEED, SHARED_DATA_DIR  # noqa: E402

ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R², MAE, Pearson r — SST against the evaluation set's own mean."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    r = (float(np.corrcoef(y_pred, y_true)[0, 1])
         if y_pred.std() > 0 and y_true.std() > 0 else float("nan"))
    return {"r2": r2, "mae": float(np.abs(y_true - y_pred).mean()), "pearson_r": r}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("fluid_iq", "age"), default="fluid_iq")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    from sklearn.linear_model import Ridge

    np.random.seed(args.seed)
    path = args.dataset or (SHARED_DATA_DIR / f"dataset_joint_{args.target}.pt")
    b = torch.load(path, map_location="cpu", weights_only=False)

    x_p = b["x_p"].numpy().astype(np.float64)
    x_f = b["x_f"].numpy().astype(np.float64)
    y = b["y"].numpy().astype(np.float64).ravel()
    tr = b["train_idx"].numpy(); va = b["val_idx"].numpy(); te = b["test_idx"].numpy()

    # Stage 2 trains on the z-scored target; match it so R² is comparable.
    mu, sd = y[tr].mean(), y[tr].std(ddof=0)
    yz = (y - mu) / sd

    print(f"dataset: {path}")
    print(f"target:  {args.target}   x_f={x_f.shape[1]:,d} edges  x_p={x_p.shape[1]:,d} proteins")
    print(f"split:   train={tr.size:,d}  val={va.size:,d}  test={te.size:,d}")
    if not b.get("scaler_fit_leakage_free", True):
        print("note:    proteomics scaler fit saw joint eval subjects (item 6, "
              "measured negligible 2026-08-01)")
    print()

    blocks = {
        "ridge_fmri": x_f,
        "ridge_prot": x_p,
        "ridge_concat": np.hstack([x_f, x_p]),
    }
    results = {"mean": metrics(yz[te], np.full(te.size, yz[tr].mean()))}

    for name, x in blocks.items():
        print(f"  {name} ({x.shape[1]:,d} features):")
        best = None
        for a in ALPHAS:
            t0 = time.time()
            m = Ridge(alpha=a).fit(x[tr], yz[tr])
            v = metrics(yz[va], m.predict(x[va]))
            print(f"    alpha={a:<8g} val r2={v['r2']:+.4f}  ({time.time() - t0:.1f}s)")
            if best is None or v["r2"] > best[1]["r2"]:
                best = (m, v, a)
        m, v, a = best
        results[name] = metrics(yz[te], m.predict(x[te]))
        results[name]["alpha"] = a
        results[name]["val_r2"] = v["r2"]

    print("\n" + "=" * 58)
    print(f"JOINT TEST SET — {args.target}  (n={te.size:,d})")
    print("=" * 58)
    print(f"{'model':<16}{'R2':>10}{'MAE':>10}{'pearson r':>12}")
    for k, m in results.items():
        print(f"{k:<16}{m['r2']:>+10.4f}{m['mae']:>10.4f}{m['pearson_r']:>+12.4f}")
    print("=" * 58)
    print("Compare ridge_concat against the Stage-2 fused row for the same "
          "target.\nIf they match, the fusion architecture is not what produces "
          "the result.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = _REPO_ROOT / "fusion" / "logs" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"baseline_joint_{args.target}_{stamp}.json").write_text(
        json.dumps(
            {"timestamp": stamp, "target": args.target, "dataset": str(path),
             "n_train": int(tr.size), "n_val": int(va.size), "n_test": int(te.size),
             "results": results}, indent=2,
        ), encoding="utf-8",
    )
    print(f"\nwritten: {out / f'baseline_joint_{args.target}_{stamp}.json'}")


if __name__ == "__main__":
    main()
