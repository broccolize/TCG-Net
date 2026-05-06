"""
Split-aware preprocessing for the TCG-Net proteomics encoder branch.

This is the single canonical preprocessing entry point for the proteomics
pipeline. It replaces the legacy ``process_proteomics.py`` and
``process_proteomics_gpu.py`` scripts entirely.

Pipeline (executed in this exact order):

    1. Load the raw plasma proteomics CSV (path resolved from
       ``config.RAW_PROTEIN_CSV``).
    2. Drop the seven metadata columns; keep ``eid`` separately.
    3. Compute a fixed 70/15/15 train/val/test split on the EID-level row
       order, using ``SPLIT_SEED``. Persist the index arrays to
       ``data/splits/split_<seed>.npz`` *before* any imputation or
       normalisation runs, so the same partition can be replayed
       deterministically across stages.
    4. Fit ``sklearn.impute.KNNImputer`` on training rows only and
       transform the three splits separately. Persist the fitted imputer.
    5. (Optional, ``--residualize``) Fit a per-protein
       ``sklearn.linear_model.LinearRegression`` on the training split
       against covariates identified from the CSV header (matching the
       categories age, sex, batch, plate, fasting time, assay warning
       flags as closely as possible). Subtract predicted values to
       obtain residuals. Persist coefficients.
    6. Fit ``sklearn.preprocessing.StandardScaler`` on training rows
       only and transform the three splits. Persist scaler object and
       mean/std arrays.
    7. Reconstruct the full processed matrix in original EID order and
       save it to ``data/proteomics_tensor.pt``.
    8. Emit a preprocessing JSON log under ``logs/preprocessing/``.

This script is the single source of truth for all preprocessing artefacts
needed by Stage 1 (single-modality pretraining) and Stage 2 (TCG-Net
joint training). Re-running this script with the same ``--seed``,
``--knn-neighbors`` and ``--residualize`` flag must produce bitwise
identical tensors and artefact files.

CRITICAL DATA PROTECTION
------------------------
The raw CSV lives under ``uk_BIOBANK_DATA/`` which is read-only confidential
research data. This script:
    * Imports its path exclusively from ``config.RAW_PROTEIN_CSV`` —
      never hardcodes it.
    * Only writes to ``TCG-Net/proteomics/`` subdirectories.
    * Never prints, logs or otherwise exposes any individual data values
      from the raw CSV. Logged quantities are aggregate counts and
      shapes only.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Make sibling packages importable when this file is run as a script.
_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from config import (  # noqa: E402  (sys.path tweak above)
    ARTEFACTS_DIR,
    DATA_DIR,
    GLOBAL_SEED,
    HOME_ROOT,
    KNN_NEIGHBORS,
    LOG_DIR_PREPROCESS,
    LOG_DIR_PROCESS,
    PROCESSED_TENSOR_PATH,
    RAW_PROTEIN_CSV,
    RESIDUALIZE_DEFAULT,
    SPLIT_SEED,
    SPLITS_DIR,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)

# Directory holding the phesantallphenos_*.csv tables. Derived from HOME_ROOT
# (a config.py path anchor) so no absolute path to uk_BIOBANK_DATA/ is hardcoded.
PHESANT_DIR = HOME_ROOT / "uk_BIOBANK_DATA" / "table"
from utils.logging_utils import dump_json, make_process_logger, make_run_id  # noqa: E402
from utils.seed import set_global_seeds  # noqa: E402


# The seven metadata columns that are not protein measurements. Anything
# present in this set is removed from the protein matrix before splitting.
# ``eid`` is removed separately and retained as the row-key array.
METADATA_COLUMNS = [
    "eid",
    "batch",
    "fastingtime_0_0",
    "bloodsample_date_0_0",
    "ProteinNum_0_0",
    "PlateID_0_0",
    "WellID_0_0",
]


# Mapping from the rebuild-plan covariate categories to candidate column
# names in the raw CSV. The first match (case-insensitive substring) wins
# per category. Categories that produce no match are simply skipped and
# documented in the preprocessing log.
COVARIATE_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "age": ["age", "21003"],
    "sex": ["sex", "31-0", "gender"],
    "batch": ["batch"],
    "plate": ["plate", "plateid"],
    "fasting time": ["fasting", "fasting_time", "fastingtime"],
    "assay warning flags": ["warning", "assay_warning", "assaywarning"],
}

# Columns within the matched covariate set that are categorical and must be
# one-hot encoded before being fed to ``LinearRegression``. Anything not in
# this set is treated as numeric. Match is by substring on the column name.
CATEGORICAL_COVARIATE_HINTS = ("batch", "plate", "warning", "sex")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the script's CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Split-aware preprocessing for the proteomics encoder."
    )
    parser.add_argument(
        "--residualize",
        action="store_true",
        default=RESIDUALIZE_DEFAULT,
        help=(
            "If passed, fit a per-protein covariate LinearRegression on the "
            "training split and subtract its predictions before z-score "
            "normalisation. Default: off."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help="Split seed (controls train/val/test partition).",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=KNN_NEIGHBORS,
        help="Number of neighbours used by the KNNImputer.",
    )
    return parser.parse_args()


def identify_covariate_columns(
    header: list[str], logger
) -> tuple[list[str], dict[str, str]]:
    """
    Match CSV header columns against the rebuild-plan covariate categories.

    Returns
    -------
    columns:
        The list of CSV column names that will be used as covariates.
    matches:
        Dict mapping each matched column name back to its category label.
    """
    columns: list[str] = []
    matches: dict[str, str] = {}
    header_lower = [h.lower() for h in header]
    for category, patterns in COVARIATE_CATEGORY_PATTERNS.items():
        matched_col: str | None = None
        for pattern in patterns:
            for h, hl in zip(header, header_lower):
                if pattern.lower() in hl and h not in columns and h.lower() != "eid":
                    matched_col = h
                    break
            if matched_col is not None:
                break
        if matched_col is None:
            logger.info("Covariate category '%s' had no matching column in CSV header.", category)
            continue
        columns.append(matched_col)
        matches[matched_col] = category
        logger.info("Covariate category '%s' matched to column '%s'.", category, matched_col)
    return columns, matches


def encode_covariates(cov_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Convert a covariate dataframe to a numeric matrix suitable for
    LinearRegression. Categorical-looking columns are one-hot encoded
    (drop_first=True to keep a single intercept identifiable).
    Numeric columns pass through; remaining NaNs are mean-imputed.
    """
    categorical = [
        c
        for c in cov_df.columns
        if any(hint in c.lower() for hint in CATEGORICAL_COVARIATE_HINTS)
        or cov_df[c].dtype == object
    ]
    numeric = [c for c in cov_df.columns if c not in categorical]

    pieces: list[pd.DataFrame] = []
    if numeric:
        num_block = cov_df[numeric].copy()
        num_block = num_block.fillna(num_block.mean(numeric_only=True))
        pieces.append(num_block)
    if categorical:
        cat_block = pd.get_dummies(
            cov_df[categorical].astype("category"),
            drop_first=True,
            dummy_na=False,
        )
        pieces.append(cat_block)

    encoded = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=cov_df.index)
    encoded = encoded.fillna(0.0)
    return encoded.to_numpy(dtype=np.float64), list(encoded.columns)


def compute_split_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce 70/15/15 train/val/test positional indices into a length-``n_rows``
    array. The shuffle is driven by a dedicated ``np.random.default_rng(seed)``
    so it is independent of the global RNG state.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_train = int(round(n_rows * TRAIN_FRAC))
    n_val = int(round(n_rows * VAL_FRAC))
    # Force the three splits to cover all rows exactly even after rounding.
    n_test = n_rows - n_train - n_val
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val:n_train + n_val + n_test])
    return train_idx, val_idx, test_idx


def find_phesant_columns(
    table_dir: Path, columns: list[str], logger
) -> dict[str, Path]:
    """
    Scan the headers of every ``phesantallphenos_*.csv`` in ``table_dir`` and
    return a mapping from each requested column name to the file it lives in.

    Only ``nrows=0`` (header-only) reads are performed during the scan — no
    row-level data values are loaded. The scan stops per column as soon as a
    match is found; the first file (alphabetical order) that contains the
    column wins.

    Parameters
    ----------
    table_dir:
        Directory containing the phesantallphenos_*.csv files.
    columns:
        List of column names to search for (e.g. ``["21003-0.0", "31-0.0"]``).
    logger:
        Active process logger; used to record each match and each miss.

    Returns
    -------
    dict mapping column name → Path of the containing file. Columns not found
    in any file are absent from the returned dict (and a WARNING is logged).
    """
    candidates = sorted(table_dir.glob("phesantallphenos_*.csv"))
    if not candidates:
        logger.warning(
            "No phesantallphenos_*.csv files found in %s. "
            "Age and sex covariates will be unavailable.", table_dir,
        )
        return {}

    remaining = set(columns)
    col_to_file: dict[str, Path] = {}

    for path in candidates:
        if not remaining:
            break
        # Header-only read: nrows=0 loads only the column index.
        header = set(pd.read_csv(path, nrows=0).columns)
        for col in list(remaining):
            if col in header:
                col_to_file[col] = path
                remaining.discard(col)
                logger.info(
                    "Phesant column '%s' found in: %s", col, path.name
                )

    for col in remaining:
        logger.warning(
            "Phesant column '%s' not found in any phesantallphenos file. "
            "This covariate will be omitted from residualization.", col,
        )

    return col_to_file


def fit_residualizer(
    x_train: np.ndarray, cov_train: np.ndarray
) -> LinearRegression:
    """
    Fit a single multi-output ``LinearRegression`` mapping covariates →
    protein values on the training rows. Equivalent to one regression per
    protein but vectorised through sklearn's multi-output support.
    """
    model = LinearRegression()
    model.fit(cov_train, x_train)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point. See module docstring for the full pipeline definition."""
    # Step 1: seed every RNG that any downstream sklearn / numpy call may touch.
    set_global_seeds(GLOBAL_SEED)

    args = parse_args()

    # Step 2: process logger. Created before anything that could fail so that
    # any exception on the way down has somewhere to land.
    logger, log_path = make_process_logger("preprocess", LOG_DIR_PROCESS)

    try:
        run_id = make_run_id()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        logger.info("Script: preprocess.py")
        logger.info("run_id: %s", run_id)
        logger.info(
            "CLI args: residualize=%s, seed=%d, knn_neighbors=%d",
            args.residualize, args.seed, args.knn_neighbors,
        )
        logger.info("Global seed: %d", GLOBAL_SEED)
        logger.info("Resolved input path: %s", RAW_PROTEIN_CSV)
        logger.info("Resolved output tensor path: %s", PROCESSED_TENSOR_PATH)
        logger.info("Resolved artefacts dir: %s", ARTEFACTS_DIR)
        logger.info("Resolved splits dir: %s", SPLITS_DIR)
        logger.info("Process log path: %s", log_path)

        # Make sure all output directories exist before any write.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR_PREPROCESS.mkdir(parents=True, exist_ok=True)

        # Step 4a: read the header first to discover metadata + covariate columns.
        # Reading nrows=0 pulls only the header row, which is the only
        # CSV-touching operation that does not load row-level data values.
        header_df = pd.read_csv(RAW_PROTEIN_CSV, nrows=0)
        full_header: list[str] = list(header_df.columns)
        logger.info("Header read: %d columns detected.", len(full_header))

        # Discover covariate columns (only used if --residualize).
        covariate_cols: list[str] = []
        covariate_matches: dict[str, str] = {}
        if args.residualize:
            covariate_cols, covariate_matches = identify_covariate_columns(
                full_header, logger
            )
            if not covariate_cols:
                logger.warning(
                    "Residualization requested but no covariate columns were "
                    "matched in the CSV header. Proceeding without residualization."
                )

        # Step 4b: load the full CSV.
        df = pd.read_csv(RAW_PROTEIN_CSV)
        logger.info("Raw CSV loaded: rows=%d, columns=%d.", df.shape[0], df.shape[1])

        # Step 5: keep eid; (optionally) keep covariates; drop metadata.
        if "eid" not in df.columns:
            raise ValueError("Raw CSV is missing the required 'eid' column.")
        eid_array = df["eid"].to_numpy()

        # Snapshot covariates BEFORE dropping metadata columns, since several
        # of them (batch, fastingtime_0_0, PlateID_0_0) are themselves in the
        # metadata-drop list.
        cov_df: pd.DataFrame | None = None
        if args.residualize and covariate_cols:
            cov_df = df[covariate_cols].copy()

        present_metadata = [c for c in METADATA_COLUMNS if c in df.columns]
        protein_df = df.drop(columns=present_metadata)
        logger.info(
            "Dropped %d metadata columns: %s",
            len(present_metadata), ", ".join(present_metadata),
        )

        feature_names: list[str] = list(protein_df.columns)
        x_raw = protein_df.to_numpy(dtype=np.float64)
        n_rows, n_features = x_raw.shape
        logger.info("Protein matrix shape: (%d, %d).", n_rows, n_features)

        # Step 6: pre-imputation missing-value diagnostic.
        n_missing_before = int(np.isnan(x_raw).sum())
        total_cells = int(x_raw.size)
        missing_frac_before = n_missing_before / total_cells if total_cells else 0.0
        logger.info(
            "Missing values before imputation: %d (%.4f%% of %d cells).",
            n_missing_before, 100.0 * missing_frac_before, total_cells,
        )
        if missing_frac_before > 0.05:
            logger.warning(
                "Missing-value fraction (%.4f%%) exceeds the 5%% threshold.",
                100.0 * missing_frac_before,
            )

        # Step 7: compute splits BEFORE imputing or normalising. This is the
        # core fix for the historical fit-on-full-data leakage bug.
        train_idx, val_idx, test_idx = compute_split_indices(n_rows, args.seed)
        logger.info(
            "Split sizes — train=%d, val=%d, test=%d (total=%d).",
            train_idx.size, val_idx.size, test_idx.size, n_rows,
        )

        split_path = SPLITS_DIR / f"split_{args.seed}.npz"
        np.savez(
            split_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            eid_order=eid_array,
        )
        logger.info("Split indices written: %s", split_path)

        # Step 8: KNN imputation. Fit on training rows only.
        x_train = x_raw[train_idx]
        x_val = x_raw[val_idx]
        x_test = x_raw[test_idx]

        imputer = KNNImputer(n_neighbors=args.knn_neighbors)
        imputer.fit(x_train)
        logger.info(
            "KNNImputer(n_neighbors=%d) fit on TRAIN split only.",
            args.knn_neighbors,
        )

        x_train_imp = imputer.transform(x_train)
        x_val_imp = imputer.transform(x_val)
        x_test_imp = imputer.transform(x_test)
        logger.info("KNN imputation complete on train, val and test splits.")

        n_missing_after = int(
            np.isnan(x_train_imp).sum()
            + np.isnan(x_val_imp).sum()
            + np.isnan(x_test_imp).sum()
        )
        logger.info("Missing values after imputation: %d.", n_missing_after)
        if n_missing_after > 0:
            logger.warning(
                "Imputation left %d missing values across all splits.",
                n_missing_after,
            )

        imputer_path = ARTEFACTS_DIR / f"knn_imputer_{args.seed}.joblib"
        joblib.dump(imputer, imputer_path)
        logger.info("KNNImputer artefact written: %s", imputer_path)

        # Step 9: optional covariate residualization on imputed values.
        residualization_applied = False
        residualization_artefact_path: Path | None = None
        normalisation_method = "z-score"
        # Tracking variables populated only when --residualize is active.
        covariate_sources: dict[str, str] = {}   # col → source-file path string
        age_sex_missing_filled: int = 0           # total missings filled across age+sex

        if args.residualize:
            # ----------------------------------------------------------------
            # Sub-step 9a: extend covariates with age (21003-0.0) and sex
            # (31-0.0) sourced from phesantallphenos files.  These two columns
            # are not present in RAW_PROTEIN_CSV but are critical confounders.
            # ----------------------------------------------------------------
            phesant_target_cols = ["21003-0.0", "31-0.0"]
            col_to_file = find_phesant_columns(PHESANT_DIR, phesant_target_cols, logger)

            for col, source_file in col_to_file.items():
                # Read only the two columns we need — never load the full table.
                col_df = pd.read_csv(source_file, usecols=["eid", col])
                lookup: dict = dict(
                    zip(col_df["eid"].to_numpy(), col_df[col].to_numpy())
                )
                # Align to the protein matrix row order via eid_array.
                col_array = np.array(
                    [lookup.get(int(e), np.nan) for e in eid_array],
                    dtype=np.float64,
                )

                n_missing_total = int(np.isnan(col_array).sum())

                # Fill strategy is determined by the column type.
                # Stats are computed on training rows ONLY to prevent leakage.
                train_col_vals = col_array[train_idx]
                if col == "21003-0.0":          # age — fill with training median
                    fill_method = "training-set median"
                    fill_val = float(np.nanmedian(train_col_vals))
                else:                           # sex (31-0.0) — fill with training mode
                    fill_method = "training-set mode"
                    non_nan = train_col_vals[~np.isnan(train_col_vals)]
                    vals, counts = np.unique(non_nan, return_counts=True)
                    fill_val = float(vals[np.argmax(counts)])

                if n_missing_total > 0:
                    logger.warning(
                        "Column '%s': %d missing values across all rows; "
                        "filling with %s.", col, n_missing_total, fill_method,
                    )
                else:
                    logger.info("Column '%s': 0 missing values; no fill needed.", col)
                logger.info("Column '%s' fill strategy applied: %s.", col, fill_method)

                col_array = np.where(np.isnan(col_array), fill_val, col_array)
                age_sex_missing_filled += n_missing_total
                covariate_sources[col] = str(source_file)

                # Determine the covariate category label for this column.
                category = "age" if col == "21003-0.0" else "sex"
                covariate_matches[col] = category
                covariate_cols.append(col)

                # Merge the filled array into cov_df.
                if cov_df is None:
                    cov_df = pd.DataFrame({col: col_array})
                else:
                    cov_df = cov_df.reset_index(drop=True)
                    cov_df[col] = col_array

            # Record source file for protein-CSV-derived covariates so the
            # JSON log shows all six covariate column → file mappings.
            for col in covariate_cols:
                if col not in covariate_sources:
                    covariate_sources[col] = str(RAW_PROTEIN_CSV)

            # ----------------------------------------------------------------
            # Sub-step 9b: encode and fit the full covariate matrix, then
            # subtract predictions to obtain residuals.
            # ----------------------------------------------------------------
            if cov_df is not None and not cov_df.empty:
                logger.info(
                    "Residualization: encoding %d covariate columns: %s",
                    len(cov_df.columns), ", ".join(cov_df.columns),
                )
                cov_matrix, cov_feature_names = encode_covariates(cov_df)
                logger.info(
                    "Covariate matrix encoded: %d rows × %d features. "
                    "Encoded feature names: %s",
                    cov_matrix.shape[0], cov_matrix.shape[1],
                    ", ".join(cov_feature_names) if cov_feature_names else "(none)",
                )

                cov_train = cov_matrix[train_idx]
                cov_val = cov_matrix[val_idx]
                cov_test = cov_matrix[test_idx]

                resid_model = fit_residualizer(x_train_imp, cov_train)
                logger.info("LinearRegression residualizer fit on TRAIN split only.")

                x_train_imp = x_train_imp - resid_model.predict(cov_train)
                x_val_imp = x_val_imp - resid_model.predict(cov_val)
                x_test_imp = x_test_imp - resid_model.predict(cov_test)

                residualization_artefact_path = (
                    ARTEFACTS_DIR / f"covariate_resid_{args.seed}.npz"
                )
                np.savez(
                    residualization_artefact_path,
                    # multi-output LR: coef_.shape == (n_proteins, n_covariates)
                    coefficients=resid_model.coef_,
                    intercepts=resid_model.intercept_,
                    covariate_columns=np.array(list(cov_df.columns), dtype=object),
                    covariate_categories=np.array(
                        [covariate_matches[c] for c in cov_df.columns], dtype=object,
                    ),
                    encoded_feature_names=np.array(cov_feature_names, dtype=object),
                    feature_names=np.array(feature_names, dtype=object),
                )
                logger.info(
                    "Residualization artefact written: %s",
                    residualization_artefact_path,
                )
                residualization_applied = True
                normalisation_method = "covariate-residualized"
            else:
                logger.warning(
                    "Residualization flag was set but no usable covariate "
                    "columns were available; falling back to plain z-score."
                )

        # Step 10: standardise. Fit on (post-residualization, if any) training
        # rows only.
        scaler = StandardScaler()
        scaler.fit(x_train_imp)
        logger.info("StandardScaler fit on TRAIN split only.")

        x_train_z = scaler.transform(x_train_imp)
        x_val_z = scaler.transform(x_val_imp)
        x_test_z = scaler.transform(x_test_imp)
        logger.info("StandardScaler transform applied to all three splits.")

        scaler_joblib_path = ARTEFACTS_DIR / f"scaler_{args.seed}.joblib"
        joblib.dump(scaler, scaler_joblib_path)
        scaler_npz_path = ARTEFACTS_DIR / f"scaler_{args.seed}.npz"
        np.savez(
            scaler_npz_path,
            mean=scaler.mean_,
            std=scaler.scale_,
            feature_names=np.array(feature_names, dtype=object),
        )
        logger.info("Scaler artefacts written: %s, %s", scaler_joblib_path, scaler_npz_path)

        # Step 11: reconstruct the full processed matrix in original EID order.
        x_full = np.empty((n_rows, n_features), dtype=np.float32)
        x_full[train_idx] = x_train_z.astype(np.float32)
        x_full[val_idx] = x_val_z.astype(np.float32)
        x_full[test_idx] = x_test_z.astype(np.float32)

        torch.save(
            {
                "x": torch.from_numpy(x_full),
                "eid": torch.from_numpy(eid_array.astype(np.int64)),
                "features": feature_names,
                "train_idx": torch.from_numpy(train_idx.astype(np.int64)),
                "val_idx": torch.from_numpy(val_idx.astype(np.int64)),
                "test_idx": torch.from_numpy(test_idx.astype(np.int64)),
            },
            PROCESSED_TENSOR_PATH,
        )
        logger.info(
            "Processed tensor written: %s (shape: %s).",
            PROCESSED_TENSOR_PATH, tuple(x_full.shape),
        )

        # Step 12: preprocessing JSON log.
        json_payload = {
            "timestamp": timestamp,
            "run_id": run_id,
            "input_file": str(RAW_PROTEIN_CSV),
            "input_shape": [int(n_rows), int(n_features)],
            "missing_values_before_imputation": n_missing_before,
            "missing_values_after_imputation": n_missing_after,
            "imputer_fit_split": "train",
            "normalisation_method": normalisation_method,
            "scaler_fit_split": "train",
            "output_shape": [int(n_rows), int(n_features)],
            "output_file": str(PROCESSED_TENSOR_PATH),
            "normalisation_statistics_file": str(scaler_npz_path),
            "split_seed": int(args.seed),
            "train_size": int(train_idx.size),
            "val_size": int(val_idx.size),
            "test_size": int(test_idx.size),
            "imputer_artefact_path": str(imputer_path),
            "covariate_residualization_applied": residualization_applied,
            "residualization_artefact_path": (
                str(residualization_artefact_path) if residualization_applied else None
            ),
            "covariate_columns_used": (
                list(cov_df.columns) if (residualization_applied and cov_df is not None) else []
            ),
            "covariate_category_matches": (
                covariate_matches if residualization_applied else {}
            ),
            "covariate_sources": covariate_sources if residualization_applied else {},
            "age_sex_missing_filled": age_sex_missing_filled,
            "knn_neighbors": int(args.knn_neighbors),
            "global_seed": int(GLOBAL_SEED),
            "split_indices_file": str(split_path),
            "process_log_path": str(log_path),
        }
        json_path = LOG_DIR_PREPROCESS / f"preprocess_log_{timestamp}.json"
        dump_json(json_path, json_payload)
        logger.info("Preprocessing JSON written: %s", json_path)

        # Step 13: completion line and clean shutdown of file handlers.
        logger.info("Preprocessing complete. run_id: %s", run_id)
    except Exception:
        # On any failure, log the full traceback then re-raise so the
        # process exits non-zero. The .log already on disk shows where
        # we stopped.
        logger.error("Preprocessing failed:\n%s", traceback.format_exc())
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
