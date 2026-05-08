"""
K-fold cross-validation runner for Stage 1 proteomics pretraining.

Operates on the 85 %% (train + val) pool only.  The 15 %% test set is
held out and never used during CV training.  Each fold runs the same
training loop as train_stage1.py, producing fold-level experiment JSONs,
training CSVs, and checkpoints.  After all folds, a summary CSV and JSON
are written.

The training loop for each fold is implemented in run_one_fold(), which
mirrors the logic in train_stage1.py.  The two scripts share the same
helpers (evaluate, save_checkpoint, etc.) via local imports, but the loop
itself is duplicated here for clarity and to avoid circular imports.
NOTE: If train_stage1.py changes the loop logic, run_one_fold() here
must be updated to match.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset

_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CV_FOLDS,
    CV_SEED,
    DATASET_AGE_PATH,
    DATASET_IQ_PATH,
    DROPOUT_1,
    DROPOUT_2,
    DROPOUT_3,
    EPOCHS,
    GLOBAL_SEED,
    LAMBDA_AUX,
    LAMBDA_SPARSE,
    LOG_DIR_EXPERIMENT,
    LOG_DIR_PROCESS,
    LOG_DIR_TRAINING,
    LR_STAGE1,
    SPLIT_SEED,
    SPLITS_DIR,
    WEIGHT_DECAY,
)
from models.encoder import ProteomicsEncoder  # noqa: E402
from models.aux_head import ProteomicsAuxHead  # noqa: E402
from training.losses import ProteomicsStage1Loss  # noqa: E402
from training.train_stage1 import (  # noqa: E402
    CSV_HEADER,
    TARGET_MAP,
    count_params,
    evaluate,
    save_checkpoint,
)
from utils.seed import make_split_generator, seed_worker, set_global_seeds  # noqa: E402
from utils.logging_utils import (  # noqa: E402
    append_csv_row,
    dump_json,
    make_process_logger,
    make_run_id,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the CV runner."""
    parser = argparse.ArgumentParser(
        description="K-fold CV runner for TCG-Net Stage 1 proteomics pretraining."
    )
    parser.add_argument(
        "--target", required=True, choices=list(TARGET_MAP),
        help="Regression target: 'age' or 'fluid_iq'.",
    )
    parser.add_argument(
        "--folds", type=int, default=CV_FOLDS,
        help="Number of CV folds (K).",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Epochs per fold.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, dest="batch_size",
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=LR_STAGE1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lambda-sparse", type=float, default=LAMBDA_SPARSE, dest="lambda_sparse",
        help="Sparsity regularisation coefficient.",
    )
    parser.add_argument(
        "--seed", type=int, default=GLOBAL_SEED,
        help="Global random seed.",
    )
    return parser.parse_args()


def load_dataset(target: str, logger) -> dict:
    """
    Load the target-specific dataset bundle; identical to train_stage1.py.

    Raises FileNotFoundError if the dataset does not exist, KeyError if
    required keys are missing.
    """
    path = TARGET_MAP[target]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run data/preprocess.py then data/build_dataset.py first."
        )
    required_keys = {"x", "y", "eid", "features", "target_name",
                     "train_idx", "val_idx", "test_idx"}
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(f"Dataset at {path} missing keys: {sorted(missing)}")
    logger.info(
        "Dataset loaded: x.shape=%s, y.shape=%s, target=%s.",
        tuple(bundle["x"].shape), tuple(bundle["y"].shape), bundle["target_name"],
    )
    return bundle


def run_one_fold(
    fold_k: int,
    parent_run_id: str,
    fold_train_idx: torch.Tensor,
    fold_val_idx: torch.Tensor,
    dataset: TensorDataset,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> dict:
    """
    Train and validate one CV fold.

    Mirrors the training loop in train_stage1.py.  Each fold gets freshly
    initialised models (set_global_seeds called with args.seed + fold_k
    before instantiation) and its own experiment JSON, training CSV, and
    pair of checkpoints.

    Returns the best-epoch val_metrics dict for fold summary aggregation.

    NOTE: This function duplicates the per-epoch loop from train_stage1.py.
    If that loop changes, update here too.
    """
    fold_run_id = f"{parent_run_id}_fold{fold_k}"

    # Fresh deterministic initialisation for this fold.
    set_global_seeds(args.seed + fold_k)

    encoder  = ProteomicsEncoder().to(device)
    aux_head = ProteomicsAuxHead().to(device)
    enc_params  = count_params(encoder)
    head_params = count_params(aux_head)

    optimizer = torch.optim.Adam(
        [{"params": encoder.parameters()},
         {"params": aux_head.parameters()}],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = ProteomicsStage1Loss(lambda_sparse=args.lambda_sparse)

    # Fold DataLoaders.
    g = make_split_generator(args.seed + fold_k)
    train_loader = DataLoader(
        Subset(dataset, fold_train_idx), batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, fold_val_idx), batch_size=args.batch_size * 2,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Fold experiment JSON.
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR_TRAINING / f"train_log_{args.target}_{fold_run_id}.csv"
    exp_json_path = LOG_DIR_EXPERIMENT / f"experiment_{fold_run_id}.json"

    dump_json(
        exp_json_path,
        {
            "run_id": fold_run_id,
            "timestamp": timestamp_utc,
            "target": args.target,
            "device": str(device),
            "cv_parent_run_id": parent_run_id,
            "fold_index": fold_k,
            "fold_train_size": len(fold_train_idx),
            "fold_val_size": len(fold_val_idx),
            "model_architecture": {
                "stage1": "Linear(2920,1024) -> BN1d(1024) -> GELU -> Dropout(0.3)",
                "se_block": "Linear(1024,256) -> GELU -> Linear(256,1024) -> Sigmoid",
                "stage2": "Linear(1024,512) -> BN1d(512) -> GELU -> Dropout(0.3)",
                "stage3": "Linear(512,256) -> BN1d(256) -> GELU -> Dropout(0.2)",
                "proj": "Linear(256,128)",
                "aux_head": "Linear(128,1)",
            },
            "hyperparameters": {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "weight_decay": WEIGHT_DECAY,
                "lambda_sparse": args.lambda_sparse,
                "lambda_aux": LAMBDA_AUX,
                "dropout_1": DROPOUT_1,
                "dropout_2": DROPOUT_2,
                "dropout_3": DROPOUT_3,
            },
            "dataset_sizes": {
                "fold_train": len(fold_train_idx),
                "fold_val": len(fold_val_idx),
            },
            "split_seed": SPLIT_SEED,
            "global_seed": args.seed,
            "encoder_param_count": enc_params,
            "aux_head_param_count": head_params,
            "checkpoint_save_path": str(CHECKPOINT_DIR),
            "training_log_save_path": str(csv_path),
        },
    )
    logger.info("Fold %d experiment JSON: %s", fold_k, exp_json_path)

    # CSV header.
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics: dict = {}
    best_ckpt_path = CHECKPOINT_DIR / f"stage1_{args.target}_{fold_run_id}_best.pt"
    last_ckpt_path = CHECKPOINT_DIR / f"stage1_{args.target}_{fold_run_id}_last.pt"

    for epoch in range(1, args.epochs + 1):

        # Train phase.
        encoder.train()
        aux_head.train()
        train_loss_sum = 0.0
        train_batches  = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            z_p, s_p = encoder(x_batch)
            y_p = aux_head(z_p)
            loss_dict = loss_fn(y_batch, y_p, s_p)
            loss_dict["total"].backward()
            optimizer.step()
            train_loss_sum += loss_dict["l_yp"].item()
            train_batches  += 1
        train_loss = train_loss_sum / train_batches

        # Validation phase (pooled).
        val_metrics = evaluate(encoder, aux_head, val_loader, device)
        val_loss      = val_metrics["val_loss"]
        val_r2        = val_metrics["val_r2"]
        val_mae       = val_metrics["val_mae"]
        val_pearson_r = val_metrics["val_pearson_r"]
        sp_l1_norm    = val_metrics["sp_l1_norm"]

        append_csv_row(
            csv_path,
            {
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_r2": val_r2,
                "val_mae": val_mae, "val_pearson_r": val_pearson_r,
                "sp_l1_norm": sp_l1_norm,
            },
            CSV_HEADER,
        )
        logger.info(
            "  Fold %d | Epoch %d/%d — train_loss=%.6f, val_loss=%.6f, "
            "val_r2=%.4f, val_mae=%.4f, val_pearson_r=%.4f, sp_l1_norm=%.6f.",
            fold_k, epoch, args.epochs,
            train_loss, val_loss, val_r2, val_mae, val_pearson_r, sp_l1_norm,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_val_metrics = {
                "val_loss": val_loss, "val_r2": val_r2,
                "val_mae": val_mae, "val_pearson_r": val_pearson_r,
                "sp_l1_norm": sp_l1_norm, "epoch": epoch,
            }
            save_checkpoint(
                best_ckpt_path, "stage1", fold_run_id, epoch,
                encoder, aux_head, optimizer, args,
                val_metrics=best_val_metrics,
            )

        save_checkpoint(
            last_ckpt_path, "stage1", fold_run_id, epoch,
            encoder, aux_head, optimizer, args,
            val_metrics={
                "val_loss": val_loss, "val_r2": val_r2,
                "val_mae": val_mae, "val_pearson_r": val_pearson_r,
                "sp_l1_norm": sp_l1_norm, "epoch": epoch,
            },
        )

    logger.info(
        "Fold %d complete. Best epoch=%d, val_loss=%.6f, "
        "val_r2=%.4f, val_pearson_r=%.4f.",
        fold_k, best_epoch,
        best_val_metrics.get("val_loss", float("nan")),
        best_val_metrics.get("val_r2", float("nan")),
        best_val_metrics.get("val_pearson_r", float("nan")),
    )
    best_val_metrics["best_epoch"] = best_epoch
    best_val_metrics["fold"] = fold_k
    return best_val_metrics


def main() -> None:
    """
    Full K-fold CV pipeline.

    Execution order:
    seeds → logger → parent_run_id → dataset → trainval pool →
    KFold split → persist fold indices → per-fold training →
    CV summary CSV → CV summary JSON.
    """
    args = parse_args()

    # Step 1: seeds.
    set_global_seeds(args.seed)

    # Step 2: process logger.
    logger, log_path = make_process_logger("cv_stage1", LOG_DIR_PROCESS)

    try:
        logger.info("Script: cv_stage1.py")
        logger.info(
            "Args: target=%s, folds=%d, epochs=%d, batch_size=%d, "
            "lr=%g, lambda_sparse=%g, seed=%d.",
            args.target, args.folds, args.epochs, args.batch_size,
            args.lr, args.lambda_sparse, args.seed,
        )
        logger.info("CV folds: %d", args.folds)

        # Step 3: parent run_id.
        parent_run_id = make_run_id()
        logger.info("CV parent run_id: %s", parent_run_id)

        # Step 4: load dataset; extract split indices.
        bundle = load_dataset(args.target, logger)
        x = bundle["x"].float()
        y = bundle["y"].float()
        train_idx = bundle["train_idx"].long().numpy()
        val_idx   = bundle["val_idx"].long().numpy()
        test_idx  = bundle["test_idx"].long().numpy()

        # Combine train + val into the 85 % trainval pool.
        trainval_idx = np.concatenate([train_idx, val_idx])
        logger.info(
            "Trainval pool size: %d, held-out test size: %d.",
            len(trainval_idx), len(test_idx),
        )

        dataset = TensorDataset(x, y)
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)

        # Step 5: create K folds over the trainval pool.
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=CV_SEED)
        fold_splits = list(kf.split(trainval_idx))

        # Persist all fold indices before training starts.
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        npz_payload = {}
        for k, (tr, vl) in enumerate(fold_splits):
            npz_payload[f"fold_{k}_train"] = trainval_idx[tr]
            npz_payload[f"fold_{k}_val"]   = trainval_idx[vl]
        fold_npz_path = (
            SPLITS_DIR /
            f"cv_{args.target}_{args.folds}fold_seed{CV_SEED}_{parent_run_id}.npz"
        )
        np.savez(fold_npz_path, **npz_payload)
        logger.info("Fold indices saved: %s", fold_npz_path)

        # Step 6: per-fold training.
        all_fold_metrics: list[dict] = []

        for k, (tr_local, vl_local) in enumerate(fold_splits):
            logger.info("Starting fold %d / %d.", k + 1, args.folds)

            fold_train_global = trainval_idx[tr_local]
            fold_val_global   = trainval_idx[vl_local]

            fold_metrics = run_one_fold(
                fold_k=k,
                parent_run_id=parent_run_id,
                fold_train_idx=torch.from_numpy(fold_train_global),
                fold_val_idx=torch.from_numpy(fold_val_global),
                dataset=dataset,
                args=args,
                device=device,
                logger=logger,
            )
            all_fold_metrics.append(fold_metrics)

        # Step 7-8: summary statistics and CSV.
        metric_keys = ["val_loss", "val_r2", "val_mae", "val_pearson_r", "sp_l1_norm"]
        summary_csv_path = (
            LOG_DIR_TRAINING / f"cv_summary_{args.target}_{parent_run_id}.csv"
        )
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

        summary_header = ["fold", "best_epoch"] + metric_keys
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_header)
            writer.writeheader()
            for fm in all_fold_metrics:
                writer.writerow({
                    "fold": fm["fold"],
                    "best_epoch": fm["best_epoch"],
                    **{k: fm[k] for k in metric_keys},
                })
            # Final mean ± std row.
            means = {k: float(np.mean([fm[k] for fm in all_fold_metrics])) for k in metric_keys}
            stds  = {k: float(np.std( [fm[k] for fm in all_fold_metrics])) for k in metric_keys}
            writer.writerow({
                "fold": "mean±std",
                "best_epoch": "",
                **{k: f"{means[k]:.6f}±{stds[k]:.6f}" for k in metric_keys},
            })
        logger.info("CV summary CSV written: %s", summary_csv_path)

        # Step 9: CV summary JSON.
        summary_json_path = (
            LOG_DIR_EXPERIMENT / f"cv_summary_{args.target}_{parent_run_id}.json"
        )
        dump_json(
            summary_json_path,
            {
                "parent_run_id": parent_run_id,
                "target": args.target,
                "n_folds": args.folds,
                "cv_seed": CV_SEED,
                "global_seed": args.seed,
                "per_fold_metrics": all_fold_metrics,
                "summary_mean": means,
                "summary_std": stds,
            },
        )
        logger.info("CV summary JSON written: %s", summary_json_path)

        # Step 10: final log line.
        logger.info(
            "CV complete. Target: %s. Folds: %d. "
            "Mean val_r2: %.4f ± %.4f. run_id: %s",
            args.target, args.folds,
            means["val_r2"], stds["val_r2"], parent_run_id,
        )

    except Exception:
        logger.error("cv_stage1.py failed:\n%s", traceback.format_exc())
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
