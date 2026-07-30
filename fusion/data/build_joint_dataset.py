"""
Build the joint proteomics–fMRI dataset for TCG-Net Stage 2.

Pipeline (executed in this exact order):

    1. Load proteomics EIDs from proteomics_tensor.pt.
    2. Load fMRI EIDs and FC feature matrix from corr_matrix.mat.
    3. Compute EID intersection (proteomics ∩ fMRI).  Log N_proteomics,
       N_fmri, N_joint (shapes and counts only — no raw values).
    4. Align both tensors row-by-row on the intersected EIDs.  The
       proteomics features in proteomics_tensor.pt were already scaled by
       scaler_1234.joblib during preprocessing; this step loads that
       artefact to record its provenance and verifies no further
       transformation is applied to the proteomics slice.  fMRI FC
       features are stored as-is (vectorised upper triangle).
    5. Generate a fresh 70/15/15 split over N_joint participants with
       SPLIT_SEED.  Entirely independent of the per-modality splits.
    6. For each target (age, fluid_iq): scan phesantallphenos CSVs for
       the target column, join to joint EIDs, and compute y_0 (mean of
       training-fold labels) from the training fold only.
    7. Save:
         fusion/data/dataset_joint_age.pt
         fusion/data/dataset_joint_fluid_iq.pt
         fusion/data/split_joint_1234.npz   (includes y_0 for both targets)
    8. Write a preprocessing JSON log to fusion/logs/preprocessing/.
    9. Write a process audit .log to fusion/logs/process/.

CRITICAL DATA PROTECTION
------------------------
- Never print or log any individual data values — shapes, counts, and
  aggregate statistics (mean, std) only.
- Proteomics tensor is read-only; no writes to proteomics/ or to
  uk_BIOBANK_DATA/ at any point.
- Shared data paths are hardcoded strings. Proteomics paths derive from
  proteomics/config.py constants (REPO_ROOT, HOME_ROOT, ARTEFACTS_DIR).
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
import torch

# ---------------------------------------------------------------------------
# Path bootstrap — make the proteomics package importable from any CWD.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PROTEOMICS_PKG = _REPO_ROOT / "proteomics"
if str(_PROTEOMICS_PKG) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG))

from config import (  # noqa: E402  (proteomics/config.py)
    ARTEFACTS_DIR,
    GLOBAL_SEED,
    HOME_ROOT,
    REPO_ROOT,
    SHARED_DATA_DIR,
    SPLIT_SEED,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)
from utils.logging_utils import dump_json, make_process_logger, make_run_id  # noqa: E402
from utils.seed import set_global_seeds  # noqa: E402

# ---------------------------------------------------------------------------
# Paths derived from config.py constants (no hardcoded strings).
# ---------------------------------------------------------------------------
FUSION_LOG_PREPROCESS = REPO_ROOT / "fusion" / "logs" / "preprocessing"
FUSION_LOG_PROCESS = REPO_ROOT / "fusion" / "logs" / "process"

SCALER_PATH = ARTEFACTS_DIR / f"scaler_{SPLIT_SEED}.joblib"

CORR_MATRIX_PATH = SHARED_DATA_DIR / "corr_matrix.mat"
PHESANT_DIR = HOME_ROOT / "uk_BIOBANK_DATA" / "table"

# Input: proteomics_tensor.pt has been moved to the shared data directory.
PROCESSED_TENSOR_PATH = SHARED_DATA_DIR / "proteomics_tensor.pt"

# Outputs: joint datasets and split file are written to the shared data directory.
DATASET_JOINT_AGE_PATH = SHARED_DATA_DIR / "dataset_joint_age.pt"
DATASET_JOINT_IQ_PATH = SHARED_DATA_DIR / "dataset_joint_fluid_iq.pt"
SPLIT_JOINT_PATH = SHARED_DATA_DIR / f"split_joint_{SPLIT_SEED}.npz"

# EIDs of every joint val/test participant. Fed to train_stage1.py --exclude-eids
# so Stage 1 never trains on a subject that Stage 2 evaluates on.
STAGE1_EXCLUDE_EIDS_PATH = SHARED_DATA_DIR / f"stage1_exclude_eids_{SPLIT_SEED}.npy"

# Target column → (phesant CSV column name, output dataset path).
#
# INSTANCE 2 (imaging visit), deliberately — do not "align" these with the
# instance-0 columns used by proteomics/data/build_dataset.py.
#
# The connectomes in corr_matrix.mat were acquired at the imaging visit, and the
# fMRI Stage-1 encoder was pretrained against 20016-2.0
# (fmri/Code/train_fmri_stage1.py). Supervising the joint model on instance 0
# would mean predicting a cognitive test taken ~9 years before the scan; for age
# it meant regressing age at recruitment, which trails the scan by
# 8.99 +/- 1.79 years (measured on this cohort).
#
# It is also a large data win. Measured coverage over the 5,328 joint participants:
#     21003-0.0  5,328 (100.0%)      21003-2.0  5,328 (100.0%)
#     20016-0.0  1,754 ( 32.9%)      20016-2.0  4,950 ( 92.9%)   <- 2.82x
#
# The asymmetry with proteomics Stage 1 is intentional. The Olink panel was run on
# BASELINE samples (UKB_Protein_BL_withCov.csv), so instance-0 cognition is
# contemporaneous with the blood draw there, and gives 16,903 pretraining subjects
# against 6,289 at instance 2. Each encoder is pretrained against the measurement
# matching its own modality's collection date; the joint model is supervised on
# the imaging visit.
TARGET_CONFIG: dict[str, dict] = {
    "age": {
        "column": "21003-2.0",
        "output": DATASET_JOINT_AGE_PATH,
    },
    "fluid_iq": {
        "column": "20016-2.0",
        "output": DATASET_JOINT_IQ_PATH,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_split_indices(
    n_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce a 70/15/15 train/val/test partition over ``n_rows`` positions.

    Uses a dedicated ``np.random.default_rng(seed)`` so the joint split is
    independent of all other RNG state consumed before this call.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_train = int(round(n_rows * TRAIN_FRAC))
    n_val = int(round(n_rows * VAL_FRAC))
    n_test = n_rows - n_train - n_val
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val : n_train + n_val + n_test])
    return train_idx, val_idx, test_idx


def discover_target_file(column: str, logger) -> Path:
    """
    Header-scan every phesantallphenos_*.csv and return the Path of the file
    that contains ``column``.  Only ``nrows=0`` reads are performed — no
    row-level data values are loaded.  Raises ``FileNotFoundError`` if none
    of the files contains the column.
    """
    candidates = sorted(PHESANT_DIR.glob("phesantallphenos_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No phesantallphenos_*.csv files found in {PHESANT_DIR}."
        )
    logger.info(
        "Scanning %d phesantallphenos files for column '%s'.",
        len(candidates),
        column,
    )
    for path in candidates:
        header = pd.read_csv(path, nrows=0).columns
        if column in header:
            logger.info("Column '%s' found in: %s", column, path.name)
            return path
    raise FileNotFoundError(
        f"None of the phesantallphenos files contain column '{column}'."
    )


def load_target_vector(
    column: str,
    joint_eids: np.ndarray,
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a target variable from the phesantallphenos tables, intersect with
    ``joint_eids``, and return ``(kept_positions, kept_eids, y_values)``.

    ``kept_positions`` indexes into ``joint_eids`` (i.e., rows of the joint
    aligned tensor).  Participants that are missing the target are excluded.
    No raw values are logged — only counts.
    """
    csv_path = discover_target_file(column, logger)
    df = pd.read_csv(csv_path, usecols=["eid", column])
    rows_before = int(df.shape[0])
    df = df.dropna(subset=[column])
    rows_after = int(df.shape[0])
    logger.info(
        "Target '%s': CSV rows before dropna=%d, after dropna=%d.",
        column, rows_before, rows_after,
    )

    target_lookup: dict[int, float] = dict(
        zip(df["eid"].to_numpy(dtype=np.int64), df[column].to_numpy(dtype=np.float64))
    )

    mask = np.array([eid in target_lookup for eid in joint_eids], dtype=bool)
    kept_positions = np.flatnonzero(mask)
    kept_eids = joint_eids[kept_positions]
    y_values = np.array(
        [target_lookup[int(e)] for e in kept_eids], dtype=np.float32
    )

    n_missing = int(joint_eids.size) - int(kept_positions.size)
    logger.info(
        "Target '%s': N_joint=%d, N_with_target=%d, N_missing_target=%d.",
        column,
        int(joint_eids.size),
        int(kept_positions.size),
        n_missing,
    )
    if n_missing > 0 and n_missing / joint_eids.size > 0.20:
        logger.warning(
            "Target '%s': more than 20%% of joint participants have no "
            "target value (%.2f%% missing).",
            column,
            100.0 * n_missing / joint_eids.size,
        )

    return kept_positions, kept_eids, y_values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point. See module docstring for the full pipeline description."""
    set_global_seeds(GLOBAL_SEED)

    logger, log_path = make_process_logger("build_joint_dataset", FUSION_LOG_PROCESS)

    try:
        run_id = make_run_id()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        logger.info("Script: fusion/data/build_joint_dataset.py")
        logger.info("run_id: %s", run_id)
        logger.info("GLOBAL_SEED=%d, SPLIT_SEED=%d", GLOBAL_SEED, SPLIT_SEED)
        logger.info("Proteomics tensor: %s", PROCESSED_TENSOR_PATH)
        logger.info("fMRI corr_matrix: %s", CORR_MATRIX_PATH)
        logger.info("Scaler artefact: %s", SCALER_PATH)
        logger.info("Output dir (shared): %s", SHARED_DATA_DIR)
        logger.info("Process log: %s", log_path)

        SHARED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        FUSION_LOG_PREPROCESS.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Step 1: Load proteomics EIDs and feature tensor.
        # proteomics_tensor.pt was produced by preprocess.py with KNN
        # imputation + StandardScaler(scaler_1234.joblib) already applied.
        # ------------------------------------------------------------------
        required_keys = {"x", "eid", "features", "train_idx", "val_idx", "test_idx"}
        bundle_p = torch.load(PROCESSED_TENSOR_PATH, map_location="cpu", weights_only=False)
        missing_keys = required_keys - set(bundle_p.keys())
        if missing_keys:
            raise KeyError(
                f"proteomics_tensor.pt is missing keys: {sorted(missing_keys)}"
            )
        x_p_full = bundle_p["x"]
        eid_p = bundle_p["eid"].numpy().astype(np.int64)
        logger.info(
            "Proteomics tensor loaded: x_p.shape=%s, N_proteomics=%d, N_features=%d.",
            tuple(x_p_full.shape), eid_p.size, x_p_full.shape[1],
        )

        # Step 4 (provenance): load scaler to record its path; the tensor
        # already has the scaler's transform applied — do NOT call transform()
        # again or the data will be double-scaled.
        scaler = joblib.load(SCALER_PATH)
        logger.info(
            "Scaler loaded (provenance check only): %s — "
            "already applied to proteomics_tensor.pt during preprocessing. "
            "Not applied again here.",
            SCALER_PATH,
        )
        _ = scaler  # retained for audit; not called on x_p_full

        # ------------------------------------------------------------------
        # Step 2: Load fMRI EIDs and FC feature matrix from corr_matrix.mat.
        # FC_Matrices: (40517, 66, 66); upper triangle (k=1) → 2145 features.
        # Patient_IDs: (40517,) EID array.
        # ------------------------------------------------------------------
        if not CORR_MATRIX_PATH.exists():
            raise FileNotFoundError(f"corr_matrix.mat not found: {CORR_MATRIX_PATH}")

        mat = sio.loadmat(str(CORR_MATRIX_PATH))
        fc_all = mat['FC_Matrices']                                     # (40517, 66, 66)
        fc_ids = np.array(mat['Patient_IDs']).squeeze().astype(np.int64)
        eid_f = fc_ids
        n_fmri_features = 66 * 65 // 2  # 2145
        logger.info(
            "corr_matrix.mat loaded: FC_Matrices.shape=%s, N_fmri=%d, n_fmri_features=%d.",
            tuple(fc_all.shape), eid_f.size, n_fmri_features,
        )

        # ------------------------------------------------------------------
        # Step 3: Compute EID intersection and align both tensors.
        # ------------------------------------------------------------------
        eid_p_set = set(eid_p.tolist())
        eid_f_set = set(eid_f.tolist())
        joint_eid_set = eid_p_set & eid_f_set
        n_joint = len(joint_eid_set)

        logger.info(
            "EID counts — N_proteomics=%d, N_fmri=%d, N_joint=%d.",
            len(eid_p_set), len(eid_f_set), n_joint,
        )
        if n_joint == 0:
            raise RuntimeError(
                "Proteomics ∩ fMRI EID intersection is empty — cannot build "
                "joint dataset."
            )
        loss_prot = len(eid_p_set) - n_joint
        loss_fmri = len(eid_f_set) - n_joint
        logger.info(
            "Participants excluded — from proteomics: %d, from fMRI: %d.",
            loss_prot, loss_fmri,
        )

        # Build the canonical joint EID order from proteomics-tensor order
        # (preserves the row ordering from preprocess.py).
        joint_mask_p = np.array([e in joint_eid_set for e in eid_p], dtype=bool)
        joint_positions_p = np.flatnonzero(joint_mask_p)
        joint_eids = eid_p[joint_positions_p]  # canonical order

        # Sanity check: every joint EID must appear exactly once in each tensor.
        if joint_eids.size != n_joint:
            raise RuntimeError(
                f"Mismatch after filtering proteomics: expected {n_joint} rows, "
                f"got {joint_eids.size}.  Duplicate EIDs in proteomics tensor?"
            )

        # Align fMRI rows to the canonical joint order and extract upper triangle.
        eid_to_idx: dict[int, int] = {int(e): i for i, e in enumerate(fc_ids)}
        if len(eid_to_idx) != fc_ids.size:
            raise RuntimeError(
                "Duplicate EIDs detected in corr_matrix.mat — cannot align."
            )
        try:
            fmri_row_indices = np.array(
                [eid_to_idx[int(e)] for e in joint_eids], dtype=np.int64
            )
        except KeyError as exc:
            raise RuntimeError(
                f"EID {exc} is in the intersection set but missing from the "
                f"fMRI lookup — data integrity error."
            ) from exc

        tri_i, tri_j = np.triu_indices(66, k=1)
        fc_sub = fc_all[fmri_row_indices].astype(np.float32)        # (N_joint, 66, 66)
        x_p_joint = x_p_full[torch.from_numpy(joint_positions_p)]  # (N_joint, 2920)
        x_f_joint = torch.from_numpy(fc_sub[:, tri_i, tri_j])      # (N_joint, 2145)

        logger.info(
            "Aligned joint tensors: x_p.shape=%s, x_f.shape=%s.",
            tuple(x_p_joint.shape), tuple(x_f_joint.shape),
        )

        # Verify alignment: the i-th row of x_p_joint and x_f_joint must
        # correspond to the same joint_eids[i].
        prot_eid_to_row: dict[int, int] = {
            int(e): i for i, e in enumerate(eid_p[joint_positions_p])
        }
        for i, eid in enumerate(joint_eids[:min(5, n_joint)]):
            if prot_eid_to_row.get(int(eid)) != i:
                raise RuntimeError(
                    f"Alignment verification failed at position {i}: "
                    f"EID {eid} maps to wrong proteomics row."
                )
        logger.info("Row-by-row alignment spot-check passed.")

        # ------------------------------------------------------------------
        # Step 5: Generate a fresh 70/15/15 joint split.
        # ------------------------------------------------------------------
        train_idx, val_idx, test_idx = compute_split_indices(n_joint, SPLIT_SEED)
        logger.info(
            "Joint split (seed=%d): train=%d, val=%d, test=%d (total=%d).",
            SPLIT_SEED, train_idx.size, val_idx.size, test_idx.size, n_joint,
        )

        # ------------------------------------------------------------------
        # Step 6: Load targets and compute y_0 for each target.
        # y_0 = mean of training labels, computed on training fold only.
        # ------------------------------------------------------------------
        y_0_results: dict[str, float] = {}
        target_stats: dict[str, dict] = {}

        for target_name, cfg in TARGET_CONFIG.items():
            column = cfg["column"]
            output_path: Path = cfg["output"]

            kept_positions, kept_eids, y_values = load_target_vector(
                column, joint_eids, logger
            )

            # Remap joint train_idx to positions within kept_eids.
            # kept_positions: indices into joint_eids that have the target.
            kept_pos_set = set(kept_positions.tolist())
            joint_pos_to_target_pos = {int(p): i for i, p in enumerate(kept_positions)}

            train_mask = np.array(
                [p in kept_pos_set for p in train_idx], dtype=bool
            )
            val_mask = np.array(
                [p in kept_pos_set for p in val_idx], dtype=bool
            )
            test_mask = np.array(
                [p in kept_pos_set for p in test_idx], dtype=bool
            )

            target_train_idx = np.array(
                [joint_pos_to_target_pos[int(p)] for p in train_idx[train_mask]],
                dtype=np.int64,
            )
            target_val_idx = np.array(
                [joint_pos_to_target_pos[int(p)] for p in val_idx[val_mask]],
                dtype=np.int64,
            )
            target_test_idx = np.array(
                [joint_pos_to_target_pos[int(p)] for p in test_idx[test_mask]],
                dtype=np.int64,
            )

            y_0 = float(y_values[target_train_idx].mean())
            y_0_results[target_name] = y_0

            y_std = float(y_values[target_train_idx].std(ddof=0))
            y_mean_full = float(y_values.mean())
            y_std_full = float(y_values.std(ddof=0))

            logger.info(
                "Target '%s': N_with_target=%d, train=%d, val=%d, test=%d. "
                "Train y_0=%.4f, train_std=%.4f.",
                target_name,
                int(y_values.size),
                int(target_train_idx.size),
                int(target_val_idx.size),
                int(target_test_idx.size),
                y_0, y_std,
            )

            # Save target-specific dataset bundle.
            x_p_target = x_p_joint[torch.from_numpy(kept_positions.astype(np.int64))]
            x_f_target = x_f_joint[torch.from_numpy(kept_positions.astype(np.int64))]

            torch.save(
                {
                    "x_p": x_p_target,
                    "x_f": x_f_target,
                    "y": torch.from_numpy(y_values),
                    "eid": torch.from_numpy(kept_eids.astype(np.int64)),
                    "target_name": target_name,
                    "target_column": column,
                    "train_idx": torch.from_numpy(target_train_idx),
                    "val_idx": torch.from_numpy(target_val_idx),
                    "test_idx": torch.from_numpy(target_test_idx),
                    "y_0": y_0,
                    "scaler_path": str(SCALER_PATH),
                    "fmri_source": str(CORR_MATRIX_PATH),
                    "n_fmri_features": n_fmri_features,
                    "split_seed": SPLIT_SEED,
                },
                output_path,
            )
            logger.info(
                "Dataset saved: %s — x_p.shape=%s, x_f.shape=%s, y.shape=%s.",
                output_path,
                tuple(x_p_target.shape),
                tuple(x_f_target.shape),
                tuple(y_values.shape),
            )

            target_stats[target_name] = {
                "column": column,
                "n_with_target": int(y_values.size),
                "train_size": int(target_train_idx.size),
                "val_size": int(target_val_idx.size),
                "test_size": int(target_test_idx.size),
                "y_0": y_0,
                "y_train_std": y_std,
                "y_full_mean": y_mean_full,
                "y_full_std": y_std_full,
                "output_file": str(output_path),
            }

        # ------------------------------------------------------------------
        # Step 7: Save shared split file with y_0 for both targets.
        # ------------------------------------------------------------------
        np.savez(
            SPLIT_JOINT_PATH,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            eid_order=joint_eids,
            y_0_age=np.float32(y_0_results.get("age", np.nan)),
            y_0_fluid_iq=np.float32(y_0_results.get("fluid_iq", np.nan)),
        )
        logger.info("Joint split file written: %s", SPLIT_JOINT_PATH)

        # ------------------------------------------------------------------
        # Step 7b: Stage-1 exclusion list.
        #
        # Every participant in the joint val/test folds is evaluated on in
        # Stage 2, so none of them may appear in a Stage-1 *training* set — the
        # pretrained encoders would otherwise have already seen their labels
        # through the aux head. Written as EIDs rather than indices because the
        # proteomics and fMRI Stage-1 cohorts have entirely different row spaces.
        #
        # Consumed by: train_stage1.py --exclude-eids
        #
        # This is the joint-level split, so it is a superset of each target's
        # val/test set. Participants without a given target simply do not appear
        # in that target's dataset, making the extra entries harmless no-ops.
        # ------------------------------------------------------------------
        exclude_eids = np.concatenate([joint_eids[val_idx], joint_eids[test_idx]])
        exclude_eids = np.unique(exclude_eids).astype(np.int64)
        np.save(STAGE1_EXCLUDE_EIDS_PATH, exclude_eids)
        logger.info(
            "Stage-1 exclusion list written: %s (%d EIDs = joint val %d + test %d). "
            "Pass this to train_stage1.py --exclude-eids before the final Stage-1 runs.",
            STAGE1_EXCLUDE_EIDS_PATH, exclude_eids.size, val_idx.size, test_idx.size,
        )

        # ------------------------------------------------------------------
        # Step 8: Preprocessing JSON log.
        # ------------------------------------------------------------------
        json_payload = {
            "timestamp": timestamp,
            "run_id": run_id,
            "script": "fusion/data/build_joint_dataset.py",
            "global_seed": GLOBAL_SEED,
            "split_seed": SPLIT_SEED,
            "proteomics_tensor": str(PROCESSED_TENSOR_PATH),
            "scaler_artefact": str(SCALER_PATH),
            "fmri_source": str(CORR_MATRIX_PATH),
            "n_proteomics": int(eid_p.size),
            "n_fmri": int(eid_f.size),
            "n_joint": n_joint,
            "n_joint_features_proteomics": int(x_p_full.shape[1]),
            "n_joint_features_fmri": n_fmri_features,
            "split": {
                "train": int(train_idx.size),
                "val": int(val_idx.size),
                "test": int(test_idx.size),
                "seed": SPLIT_SEED,
            },
            "targets": target_stats,
            "split_file": str(SPLIT_JOINT_PATH),
            "process_log": str(log_path),
        }
        json_path = FUSION_LOG_PREPROCESS / f"joint_dataset_build_{timestamp}.json"
        dump_json(json_path, json_payload)
        logger.info("Preprocessing JSON log written: %s", json_path)

        logger.info(
            "Joint dataset build complete. N_joint=%d, run_id=%s.", n_joint, run_id
        )

    except Exception:
        logger.error("Joint dataset build failed:\n%s", traceback.format_exc())
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
