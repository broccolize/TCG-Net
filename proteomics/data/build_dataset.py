"""
Target-linking script for the TCG-Net proteomics encoder branch.

This script joins the processed proteomics tensor produced by
``preprocess.py`` to a cognitive target variable read from one of the
``phesantallphenos_*.csv`` tables, and emits a ready-to-train dataset
bundle ``(x, y, eid, features, train_idx, val_idx, test_idx)``.

It replaces the legacy ``create_age_dataset.py`` and
``create_final_dataset.py`` scripts. Both targets (``age`` and
``fluid_iq``) are handled by the same code path, parameterised via the
required ``--target`` CLI argument.

Pipeline (executed in this exact order):

    1. Load ``data/proteomics_tensor.pt``. Validate that all required
       keys are present.
    2. Scan the headers of every ``phesantallphenos_*.csv`` file in
       ``uk_BIOBANK_DATA/table/`` to identify which file contains the
       target column for the requested target. Only header rows are
       inspected during this scan.
    3. Once the correct file is identified, read it in full, drop rows
       where the target column is NaN.
    4. Intersect EIDs between the proteomics tensor and the target
       table, preserving the proteomics-tensor EID order.
    5. Filter the persisted train/val/test indices down to the
       intersected subset (no re-shuffling — the global partition from
       ``preprocess.py`` is preserved).
    6. Save the dataset bundle to the target-specific output path.
    7. Emit a per-target dataset-build JSON log under
       ``logs/preprocessing/``.

CRITICAL DATA PROTECTION
------------------------
Header-only reads on every ``phesantallphenos_*.csv`` until the correct
file is identified, after which exactly one full read of that one file
is permitted. No raw data values are printed or logged at any stage —
only aggregate counts, shapes, and target mean/std after the join.
All paths are imported from ``config.py``; nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from config import (  # noqa: E402
    DATASET_AGE_PATH,
    DATASET_IQ_PATH,
    GLOBAL_SEED,
    HOME_ROOT,
    LOG_DIR_PREPROCESS,
    LOG_DIR_PROCESS,
    PROCESSED_TENSOR_PATH,
    SPLIT_SEED,
    SPLITS_DIR,
)
from utils.logging_utils import dump_json, make_process_logger, make_run_id  # noqa: E402
from utils.seed import set_global_seeds  # noqa: E402


# Directory holding the phesantallphenos_*.csv tables. Imported indirectly
# via HOME_ROOT (which itself is derived from config.py path anchors) so
# the script never hardcodes an absolute filesystem path.
PHESANT_DIR = HOME_ROOT / "uk_BIOBANK_DATA" / "table"


# Target → (column_name, output_path_constant). The CSV file is discovered
# at runtime by header scan — no filename is hardcoded here.
TARGET_CONFIG: dict[str, dict] = {
    "age": {
        "column": "21003-0.0",
        "output": DATASET_AGE_PATH,
    },
    "fluid_iq": {
        "column": "20016-0.0",
        "output": DATASET_IQ_PATH,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the script's CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build a target-linked dataset for the proteomics encoder."
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=sorted(TARGET_CONFIG.keys()),
        help="Which target to link the proteomics tensor against.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help=(
            "Split seed. Must match the seed used in preprocess.py — "
            "the persisted split indices in the proteomics tensor are "
            "what gets filtered down to the intersected subset."
        ),
    )
    return parser.parse_args()


def discover_target_file(column: str, logger) -> Path:
    """
    Scan the headers of every ``phesantallphenos_*.csv`` and return the
    one that contains ``column``.

    The scan reads ``nrows=0`` so no row-level data values are loaded.
    Raises ``FileNotFoundError`` if no file contains the column.
    """
    candidates = sorted(PHESANT_DIR.glob("phesantallphenos_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No phesantallphenos_*.csv files found in {PHESANT_DIR}."
        )
    logger.info("Scanning %d phesantallphenos files for column '%s'.",
                len(candidates), column)
    for path in candidates:
        # Header-only read: nrows=0 fetches the column index without
        # touching any cell value.
        header = pd.read_csv(path, nrows=0).columns
        if column in header:
            logger.info("Target column '%s' found in: %s", column, path.name)
            return path
    raise FileNotFoundError(
        f"None of the phesantallphenos files contain the column '{column}'."
    )


def load_proteomics_tensor(path: Path, logger) -> dict:
    """
    Load the dict produced by ``preprocess.py`` and validate that all
    required keys are present. Returns the raw dict (tensors not moved
    off CPU).
    """
    required_keys = {"x", "eid", "features", "train_idx", "val_idx", "test_idx"}
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(
            f"Proteomics tensor at {path} is missing required keys: {sorted(missing)}."
        )
    x = bundle["x"]
    eid = bundle["eid"]
    logger.info(
        "Proteomics tensor loaded: x.shape=%s, n_participants=%d, n_features=%d.",
        tuple(x.shape), eid.shape[0], x.shape[1],
    )
    return bundle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point. See module docstring for the full pipeline definition."""
    # Step 1: seed every RNG that any downstream sklearn / numpy call may touch.
    set_global_seeds(GLOBAL_SEED)

    args = parse_args()
    target = args.target
    target_cfg = TARGET_CONFIG[target]
    target_column = target_cfg["column"]
    output_path: Path = target_cfg["output"]

    # Step 2: process logger.
    logger, log_path = make_process_logger("build_dataset", LOG_DIR_PROCESS)

    try:
        run_id = make_run_id()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        logger.info("Script: build_dataset.py")
        logger.info("run_id: %s", run_id)
        logger.info("Target: %s, target column: %s", target, target_column)
        logger.info("Split seed: %d", args.seed)
        logger.info("Resolved proteomics tensor path: %s", PROCESSED_TENSOR_PATH)
        logger.info("Resolved phesantallphenos dir: %s", PHESANT_DIR)
        logger.info("Resolved output dataset path: %s", output_path)
        logger.info("Process log path: %s", log_path)

        # Make sure all output directories exist before any write.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR_PREPROCESS.mkdir(parents=True, exist_ok=True)

        # Step 4: load the processed proteomics tensor.
        bundle = load_proteomics_tensor(PROCESSED_TENSOR_PATH, logger)
        x_proteomics = bundle["x"]
        eid_proteomics = bundle["eid"].numpy().astype(np.int64)
        feature_names = list(bundle["features"])
        train_idx_global = bundle["train_idx"].numpy().astype(np.int64)
        val_idx_global = bundle["val_idx"].numpy().astype(np.int64)
        test_idx_global = bundle["test_idx"].numpy().astype(np.int64)

        # Step 5: identify which phesantallphenos file holds the target.
        target_csv = discover_target_file(target_column, logger)

        # Read only the required columns to keep memory footprint small;
        # this is the single full read of the discovered file. The phesant
        # tables can contain ~2300+ columns, so loading just two is a real
        # speed-up.
        target_df = pd.read_csv(target_csv, usecols=["eid", target_column])
        rows_before_dropna = int(target_df.shape[0])
        target_df = target_df.dropna(subset=[target_column])
        rows_after_dropna = int(target_df.shape[0])
        logger.info(
            "Target CSV rows before dropna: %d, after dropna: %d.",
            rows_before_dropna, rows_after_dropna,
        )

        # Step 6: intersect EIDs, preserving the proteomics-tensor order.
        target_lookup = dict(
            zip(target_df["eid"].to_numpy(dtype=np.int64),
                target_df[target_column].to_numpy(dtype=np.float64))
        )
        proteomics_eid_set_size = int(eid_proteomics.size)
        target_eid_set_size = int(target_df["eid"].nunique())

        intersection_mask = np.array(
            [eid in target_lookup for eid in eid_proteomics], dtype=bool
        )
        intersected_positions = np.flatnonzero(intersection_mask)
        intersection_size = int(intersected_positions.size)
        participants_lost = proteomics_eid_set_size - intersection_size
        loss_frac = (
            participants_lost / proteomics_eid_set_size
            if proteomics_eid_set_size else 0.0
        )

        logger.info(
            "Proteomics participants: %d, target participants: %d, "
            "intersection size: %d, lost in intersection: %d (%.2f%%).",
            proteomics_eid_set_size, target_eid_set_size,
            intersection_size, participants_lost, 100.0 * loss_frac,
        )
        if loss_frac > 0.20:
            logger.warning(
                "More than 20%% of proteomics participants (%.2f%%) "
                "have no matching target value.", 100.0 * loss_frac,
            )

        if intersection_size == 0:
            raise RuntimeError(
                "Intersection between proteomics tensor and target CSV is empty."
            )

        # Build the target vector in proteomics-tensor EID order over the
        # intersected positions only. Same order is used for x_intersected.
        kept_eids = eid_proteomics[intersected_positions]
        y_values = np.array([target_lookup[e] for e in kept_eids], dtype=np.float32)
        x_intersected = x_proteomics[torch.from_numpy(intersected_positions)]

        # Step 7: re-emit the splits on the intersected subset. The global
        # split is preserved — we just keep whichever positions are still
        # in the intersection, then remap them onto the *new* positional
        # axis (0 .. intersection_size - 1) used by the saved dataset.
        global_to_local = -np.ones(eid_proteomics.size, dtype=np.int64)
        global_to_local[intersected_positions] = np.arange(intersection_size, dtype=np.int64)

        def _filter(global_idx: np.ndarray) -> np.ndarray:
            """Keep only globals that survived the intersection, remap to local."""
            mask = intersection_mask[global_idx]
            return global_to_local[global_idx[mask]]

        train_idx = _filter(train_idx_global)
        val_idx = _filter(val_idx_global)
        test_idx = _filter(test_idx_global)
        logger.info(
            "Intersected split sizes — train=%d, val=%d, test=%d (total=%d).",
            train_idx.size, val_idx.size, test_idx.size, intersection_size,
        )

        intersected_split_path = SPLITS_DIR / f"split_{target}_{args.seed}.npz"
        np.savez(
            intersected_split_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            eid_order=kept_eids,
        )
        logger.info("Intersected split indices written: %s", intersected_split_path)

        # Step 8: save the final dataset bundle.
        torch.save(
            {
                "x": x_intersected,
                "y": torch.from_numpy(y_values),
                "eid": torch.from_numpy(kept_eids.astype(np.int64)),
                "features": feature_names,
                "target_name": target,
                "train_idx": torch.from_numpy(train_idx.astype(np.int64)),
                "val_idx": torch.from_numpy(val_idx.astype(np.int64)),
                "test_idx": torch.from_numpy(test_idx.astype(np.int64)),
            },
            output_path,
        )
        logger.info(
            "Dataset saved: %s (x.shape=%s, y.shape=%s).",
            output_path, tuple(x_intersected.shape), tuple(y_values.shape),
        )

        # Step 9: dataset-build JSON log.
        target_mean = float(y_values.mean()) if y_values.size else 0.0
        target_std = float(y_values.std(ddof=0)) if y_values.size else 0.0
        json_payload = {
            "timestamp": timestamp,
            "run_id": run_id,
            "target": target,
            "input_proteomics_file": str(PROCESSED_TENSOR_PATH),
            "input_target_csv": str(target_csv),
            "target_column": target_column,
            "rows_in_target_csv_before_dropna": rows_before_dropna,
            "rows_after_dropna": rows_after_dropna,
            "intersection_size": intersection_size,
            "participants_lost": participants_lost,
            "output_file": str(output_path),
            "target_mean": target_mean,
            "target_std": target_std,
            "train_size": int(train_idx.size),
            "val_size": int(val_idx.size),
            "test_size": int(test_idx.size),
            "split_indices_file": str(intersected_split_path),
            "split_seed": int(args.seed),
            "global_seed": int(GLOBAL_SEED),
            "process_log_path": str(log_path),
        }
        json_path = (
            LOG_DIR_PREPROCESS / f"dataset_build_{target}_{timestamp}.json"
        )
        dump_json(json_path, json_payload)
        logger.info("Dataset build JSON written: %s", json_path)

        logger.info(
            "Dataset build complete. Target: %s. run_id: %s", target, run_id
        )
    except Exception:
        logger.error("Dataset build failed:\n%s", traceback.format_exc())
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
