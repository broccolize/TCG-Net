"""
Stage 1 single-modality pretraining loop for the TCG-Net proteomics encoder.

Trains ProteomicsEncoder + ProteomicsAuxHead on one of two regression
targets (age, fluid_iq) using the dataset bundle produced by
build_dataset.py.  Produces four artefacts per run:
  - logs/process/process_train_stage1_<timestamp>.log
  - logs/experiments/experiment_<run_id>.json
  - logs/training/train_log_<target>_<run_id>.csv
  - checkpoints/stage1_<target>_<run_id>_best.pt
  - checkpoints/stage1_<target>_<run_id>_last.pt

Loss:  L = MSE(y, y_p) + lambda_sparse * ||s_p||_1
       (lambda_aux is stored in config for Stage 2; not applied here.)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATASET_AGE_PATH,
    DATASET_IQ_PATH,
    DROPOUT_1,
    DROPOUT_2,
    DROPOUT_3,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GLOBAL_SEED,
    LAMBDA_AUX,
    LAMBDA_SPARSE,
    LOG_DIR_EXPERIMENT,
    LOG_DIR_PROCESS,
    LOG_DIR_TRAINING,
    LR_STAGE1,
    SPLIT_SEED,
    WEIGHT_DECAY,
)
from models.encoder import ProteomicsEncoder  # noqa: E402
from models.aux_head import ProteomicsAuxHead  # noqa: E402
from training.losses import ProteomicsStage1Loss  # noqa: E402
from utils.seed import make_split_generator, seed_worker, set_global_seeds  # noqa: E402
from utils.logging_utils import (  # noqa: E402
    append_csv_row,
    dump_json,
    make_process_logger,
    make_run_id,
)

# Maps CLI --target values to dataset paths defined in config.py.
TARGET_MAP = {
    "age": DATASET_AGE_PATH,
    "fluid_iq": DATASET_IQ_PATH,
}

# Column order for the per-epoch training CSV.
CSV_HEADER = [
    "epoch", "train_loss", "val_loss", "val_r2",
    "val_mae", "val_pearson_r", "sp_l1_norm",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Stage 1 training script."""
    parser = argparse.ArgumentParser(
        description="TCG-Net Stage 1 proteomics pretraining."
    )
    parser.add_argument(
        "--target", required=True, choices=list(TARGET_MAP),
        help="Regression target: 'age' or 'fluid_iq'.",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, dest="batch_size",
        help="Mini-batch size for the train loader.",
    )
    parser.add_argument(
        "--lr", type=float, default=LR_STAGE1,
        help="Learning rate for Adam optimiser.",
    )
    parser.add_argument(
        "--lambda-sparse", type=float, default=LAMBDA_SPARSE, dest="lambda_sparse",
        help="Coefficient on the ||s_p||_1 sparsity regularisation term.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY, dest="weight_decay",
        help=(
            "Adam weight decay. The 1e-5 default is very light for a 4.2M-parameter "
            "model on a weak target; try 1e-3 to 1e-2 for fluid_iq."
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=EARLY_STOP_PATIENCE, dest="patience",
        help=(
            "Stop after this many consecutive epochs without a val_loss improvement. "
            "Pass 0 to disable early stopping and always run --epochs epochs."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=GLOBAL_SEED,
        help="Global random seed (RNG, weight init, DataLoader workers).",
    )
    parser.add_argument(
        "--exclude-eids", type=Path, default=None, dest="exclude_eids",
        help=(
            "Path to a .npy array of EIDs to drop from the training and validation "
            "splits. Use the file written by fusion/data/build_joint_dataset.py "
            "(stage1_exclude_eids_<seed>.npy) so Stage 1 never trains on a "
            "participant that Stage 2 evaluates on."
        ),
    )
    parser.add_argument(
        "--no-test-eval", action="store_true", dest="no_test_eval",
        help="Skip test-set evaluation after training completes.",
    )
    return parser.parse_args()


def load_dataset(target: str, logger) -> dict:
    """
    Load and validate the target-specific dataset bundle from disk.

    Expects the bundle to contain keys: x, y, eid, features, target_name,
    train_idx, val_idx, test_idx.  Raises KeyError if any are missing and
    FileNotFoundError if the file does not exist.
    """
    path = TARGET_MAP[target]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Run data/preprocess.py then data/build_dataset.py first."
        )
    required_keys = {"x", "y", "eid", "features", "target_name",
                     "train_idx", "val_idx", "test_idx"}
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(f"Dataset at {path} is missing keys: {sorted(missing)}")
    logger.info(
        "Dataset loaded: path=%s, x.shape=%s, y.shape=%s, target=%s.",
        path, tuple(bundle["x"].shape), tuple(bundle["y"].shape),
        bundle["target_name"],
    )
    return bundle


def apply_eid_exclusion(bundle: dict, exclude_path: Path, logger) -> dict:
    """
    Drop participants listed in ``exclude_path`` from train_idx and val_idx.

    Stage 2 evaluates on the joint val/test folds, so those participants must
    never appear in a Stage-1 *training* set — otherwise the pretrained encoder
    has already been optimised against their labels via the auxiliary head, and
    every Stage-2 validation number is inflated.

    val_idx is filtered as well as train_idx: validation drives best-epoch
    selection, so leaving them there would still let their labels influence which
    checkpoint is promoted.

    test_idx is deliberately left intact. It is the proteomics-side held-out
    split, is never trained on, and is what the Stage-1 numbers are reported
    against.

    Returns
    -------
    dict with the exclusion counts, for the experiment JSON.
    """
    if not exclude_path.exists():
        raise FileNotFoundError(
            f"--exclude-eids file not found: {exclude_path}. It is written by "
            "fusion/data/build_joint_dataset.py, which must be run first."
        )

    exclude_eids = np.load(exclude_path).astype(np.int64)
    eids = bundle["eid"].numpy().astype(np.int64)

    # Row positions in *this* dataset belonging to a held-out participant.
    banned = np.flatnonzero(np.isin(eids, exclude_eids))
    logger.info(
        "Exclusion list: %d EIDs from %s; %d present in this dataset.",
        exclude_eids.size, exclude_path, banned.size,
    )

    stats: dict = {
        "exclude_eids_file": str(exclude_path),
        "exclude_eids_total": int(exclude_eids.size),
        "exclude_eids_matched": int(banned.size),
    }

    for key in ("train_idx", "val_idx"):
        before = bundle[key].numpy()
        kept = before[~np.isin(before, banned)]
        removed = before.size - kept.size
        logger.info(
            "  %s: %d -> %d (removed %d, %.2f%%).",
            key, before.size, kept.size, removed,
            100.0 * removed / before.size if before.size else 0.0,
        )
        if kept.size == 0:
            raise ValueError(
                f"Exclusion emptied {key} — every row was held out downstream. "
                "Check that the exclusion list matches this dataset's split seed."
            )
        bundle[key] = torch.from_numpy(kept.astype(np.int64))
        stats[f"{key}_before_exclusion"] = int(before.size)
        stats[f"{key}_removed"] = int(removed)

    return stats


def build_dataloaders(
    bundle: dict, batch_size: int, seed: int, logger
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct train / val / test DataLoaders from the persisted split indices.

    Uses the split indices embedded in the dataset bundle so the partition
    is identical to the one used in preprocessing.  The train loader is
    shuffled with a seeded generator; val and test loaders are unshuffled.
    """
    x = bundle["x"].float()
    y = bundle["y"].float()
    train_idx = bundle["train_idx"].long()
    val_idx   = bundle["val_idx"].long()
    test_idx  = bundle["test_idx"].long()

    dataset = TensorDataset(x, y)
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    g = make_split_generator(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    logger.info(
        "DataLoaders built: train=%d, val=%d, test=%d samples, batch_size=%d.",
        len(train_set), len(val_set), len(test_set), batch_size,
    )
    return train_loader, val_loader, test_loader


def count_params(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    encoder, aux_head, train_loader, optimizer, loss_fn, device
) -> tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns (train_loss, train_sp_l1_norm) where train_loss is the mean
    per-batch l_yp (MSE) and train_sp_l1_norm is the mean per-batch
    sp_l1_norm — both suitable for logging.
    """
    encoder.train()
    aux_head.train()

    train_loss_sum = 0.0
    train_sp_l1_sum = 0.0
    train_batches = 0

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
        train_sp_l1_sum += loss_dict["sp_l1_norm"].item()
        train_batches += 1

    train_loss = train_loss_sum / train_batches
    train_sp_l1 = train_sp_l1_sum / train_batches
    return train_loss, train_sp_l1


def evaluate(encoder, aux_head, loader, device) -> dict:
    """
    Compute pooled validation (or test) metrics over an entire DataLoader.

    All predictions and targets are concatenated before any metric is
    computed so that R², MAE, and Pearson r are unbiased by batch size.
    Returns a dict with keys: val_loss, val_r2, val_mae, val_pearson_r,
    sp_l1_norm.
    """
    encoder.eval()
    aux_head.eval()

    all_preds = []
    all_targets = []
    all_sp_l1 = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            z_p, s_p = encoder(x_batch.to(device))
            y_p = aux_head(z_p)
            all_preds.append(y_p.cpu())
            all_targets.append(y_batch.cpu())
            all_sp_l1.append(s_p.abs().mean(dim=1).mean().item())

    preds   = torch.cat(all_preds).squeeze()
    targets = torch.cat(all_targets).squeeze()

    val_loss = F.mse_loss(preds, targets).item()

    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    val_r2 = (1.0 - ss_res / ss_tot).item()

    val_mae = (preds - targets).abs().mean().item()

    val_pearson_r = torch.corrcoef(
        torch.stack([preds, targets])
    )[0, 1].item()

    sp_l1_norm = sum(all_sp_l1) / len(all_sp_l1)

    return {
        "val_loss": val_loss,
        "val_r2": val_r2,
        "val_mae": val_mae,
        "val_pearson_r": val_pearson_r,
        "sp_l1_norm": sp_l1_norm,
    }


def save_checkpoint(
    path: Path, stage: str, run_id: str, epoch: int,
    encoder, aux_head, optimizer, args, val_metrics: dict
) -> None:
    """
    Persist encoder + aux_head + optimiser state plus full metadata.

    The encoder and aux_head state dicts are stored under separate keys so
    Stage 2 can load only the encoder without any coupling to the aux head.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "run_id": run_id,
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "aux_head_state_dict": aux_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "lambda_sparse": args.lambda_sparse,
                "lambda_aux": LAMBDA_AUX,
                "target": args.target,
                "global_seed": args.seed,
                "split_seed": SPLIT_SEED,
            },
            "val_metrics": val_metrics,
        },
        path,
    )


def main() -> None:
    """
    Full Stage 1 training pipeline.

    Execution order follows the rebuild plan §2.11 exactly:
    seeds → logger → run_id → dataset → loaders → models → optimiser →
    loss → experiment JSON → CSV header → training loop → test eval.
    """
    args = parse_args()

    # Step 1: seed every RNG before any other work.
    set_global_seeds(args.seed)

    # Step 2: process logger.
    logger, log_path = make_process_logger("train_stage1", LOG_DIR_PROCESS)

    try:
        logger.info("Script: train_stage1.py")
        logger.info(
            "Args: target=%s, epochs=%d, batch_size=%d, lr=%g, weight_decay=%g, "
            "patience=%d, lambda_sparse=%g, seed=%d, no_test_eval=%s.",
            args.target, args.epochs, args.batch_size, args.lr, args.weight_decay,
            args.patience, args.lambda_sparse, args.seed, args.no_test_eval,
        )
        logger.info("Resolved dataset path: %s", TARGET_MAP[args.target])
        logger.info("Checkpoint dir: %s", CHECKPOINT_DIR)
        logger.info("Log dir (training): %s", LOG_DIR_TRAINING)

        # Step 3: run identifier.
        run_id = make_run_id()
        logger.info("run_id: %s", run_id)

        # Step 4: load dataset.
        bundle = load_dataset(args.target, logger)

        # Step 4b: drop Stage-2 evaluation subjects from train/val, if asked.
        exclusion_stats: dict = {"applied": False}
        if args.exclude_eids is not None:
            exclusion_stats = apply_eid_exclusion(bundle, args.exclude_eids, logger)
            exclusion_stats["applied"] = True
        else:
            logger.warning(
                "No --exclude-eids given. This is fine for a diagnostic run, but a "
                "Stage-1 encoder trained without it will have seen the participants "
                "Stage 2 evaluates on."
            )

        # Step 5-6: DataLoaders and device.
        train_loader, val_loader, test_loader = build_dataloaders(
            bundle, args.batch_size, args.seed, logger
        )
        n_train = len(bundle["train_idx"])
        n_val   = len(bundle["val_idx"])
        n_test  = len(bundle["test_idx"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)

        # Step 7: instantiate models.
        encoder  = ProteomicsEncoder().to(device)
        aux_head = ProteomicsAuxHead().to(device)
        enc_params  = count_params(encoder)
        head_params = count_params(aux_head)
        logger.info("ProteomicsEncoder instantiated — trainable params: %d.", enc_params)
        logger.info("ProteomicsAuxHead instantiated — trainable params: %d.", head_params)

        # Step 8: optimiser.
        optimizer = torch.optim.Adam(
            [{"params": encoder.parameters()},
             {"params": aux_head.parameters()}],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Step 9: loss function.
        loss_fn = ProteomicsStage1Loss(lambda_sparse=args.lambda_sparse)

        # Step 10: experiment JSON (written before training begins).
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = LOG_DIR_TRAINING / f"train_log_{args.target}_{run_id}.csv"
        exp_json_path = LOG_DIR_EXPERIMENT / f"experiment_{run_id}.json"

        experiment_payload = {
            "run_id": run_id,
            "timestamp": timestamp_utc,
            "target": args.target,
            "device": str(device),
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
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "lambda_sparse": args.lambda_sparse,
                "lambda_aux": LAMBDA_AUX,
                "dropout_1": DROPOUT_1,
                "dropout_2": DROPOUT_2,
                "dropout_3": DROPOUT_3,
            },
            "dataset_sizes": {"train": n_train, "val": n_val, "test": n_test},
            "eid_exclusion": exclusion_stats,
            "split_seed": SPLIT_SEED,
            "global_seed": args.seed,
            "encoder_param_count": enc_params,
            "aux_head_param_count": head_params,
            "checkpoint_save_path": str(CHECKPOINT_DIR),
            "training_log_save_path": str(csv_path),
            "process_log_path": str(log_path),
        }
        dump_json(exp_json_path, experiment_payload)
        logger.info("Experiment JSON written: %s", exp_json_path)

        # Step 11: create CSV with header only (no data rows yet).
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        logger.info("Training CSV created: %s", csv_path)

        # Step 12: training loop.
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_since_improvement = 0
        stopped_early = False
        best_ckpt_path = CHECKPOINT_DIR / f"stage1_{args.target}_{run_id}_best.pt"
        last_ckpt_path = CHECKPOINT_DIR / f"stage1_{args.target}_{run_id}_last.pt"

        for epoch in range(1, args.epochs + 1):

            # ---- TRAIN PHASE ------------------------------------------------
            train_loss, _ = train_one_epoch(
                encoder, aux_head, train_loader, optimizer, loss_fn, device
            )

            # ---- VALIDATION PHASE (pooled) ----------------------------------
            val_metrics = evaluate(encoder, aux_head, val_loader, device)
            val_loss      = val_metrics["val_loss"]
            val_r2        = val_metrics["val_r2"]
            val_mae       = val_metrics["val_mae"]
            val_pearson_r = val_metrics["val_pearson_r"]
            sp_l1_norm    = val_metrics["sp_l1_norm"]

            # ---- CSV ROW ----------------------------------------------------
            append_csv_row(
                csv_path,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "val_mae": val_mae,
                    "val_pearson_r": val_pearson_r,
                    "sp_l1_norm": sp_l1_norm,
                },
                CSV_HEADER,
            )
            logger.info(
                "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f, val_r2=%.4f, "
                "val_mae=%.4f, val_pearson_r=%.4f, sp_l1_norm=%.6f.",
                epoch, args.epochs,
                train_loss, val_loss, val_r2, val_mae, val_pearson_r, sp_l1_norm,
            )

            # ---- CHECKPOINT: best validation --------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                save_checkpoint(
                    best_ckpt_path, "stage1", run_id, epoch,
                    encoder, aux_head, optimizer, args,
                    val_metrics={
                        "val_loss": val_loss, "val_r2": val_r2,
                        "val_mae": val_mae, "val_pearson_r": val_pearson_r,
                        "sp_l1_norm": sp_l1_norm, "epoch": epoch,
                    },
                )
                logger.info(
                    "New best checkpoint — epoch=%d, val_loss=%.6f, val_r2=%.4f.",
                    epoch, val_loss, val_r2,
                )
                epochs_since_improvement = 0
            else:
                # NaN val_loss also lands here, since `nan < x` is False. That is
                # deliberate: a diverged epoch must never reset the patience clock.
                epochs_since_improvement += 1

            # ---- CHECKPOINT: last.pt (overwrites every epoch) ---------------
            save_checkpoint(
                last_ckpt_path, "stage1", run_id, epoch,
                encoder, aux_head, optimizer, args,
                val_metrics={
                    "val_loss": val_loss, "val_r2": val_r2,
                    "val_mae": val_mae, "val_pearson_r": val_pearson_r,
                    "sp_l1_norm": sp_l1_norm, "epoch": epoch,
                },
            )
            logger.info("Last checkpoint updated: %s.", last_ckpt_path)

            # ---- EARLY STOPPING ---------------------------------------------
            if args.patience > 0 and epochs_since_improvement >= args.patience:
                stopped_early = True
                logger.info(
                    "Early stopping at epoch %d — no val_loss improvement for %d "
                    "consecutive epochs (best epoch %d, val_loss %.6f).",
                    epoch, epochs_since_improvement, best_epoch, best_val_loss,
                )
                break

        # Step 13: post-training summary.
        logger.info(
            "Training complete (%s). Best epoch: %d. Best val_loss: %.6f.",
            "early-stopped" if stopped_early else f"ran all {args.epochs} epochs",
            best_epoch, best_val_loss,
        )

        # Step 14: test set evaluation (unless --no-test-eval).
        if not args.no_test_eval:
            # `val_loss < best_val_loss` is False for NaN, so a run that diverges on
            # every epoch never writes best.pt. Fall back to last.pt rather than
            # dying on FileNotFoundError and discarding the whole run.
            if best_epoch == -1 or not best_ckpt_path.exists():
                if not last_ckpt_path.exists():
                    raise FileNotFoundError(
                        f"No checkpoint was written — no epoch completed "
                        f"(epochs={args.epochs}). Nothing to evaluate."
                    )
                logger.warning(
                    "No best checkpoint was written (best_epoch=%d) — val_loss never "
                    "improved on inf, which usually means the run diverged to NaN. "
                    "Falling back to %s for test evaluation; treat these metrics as "
                    "diagnostic, not as a result.",
                    best_epoch, last_ckpt_path,
                )
                eval_ckpt_path = last_ckpt_path
            else:
                eval_ckpt_path = best_ckpt_path
            ckpt = torch.load(eval_ckpt_path, map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            aux_head.load_state_dict(ckpt["aux_head_state_dict"])

            test_metrics = evaluate(encoder, aux_head, test_loader, device)
            logger.info(
                "Test metrics — test_loss=%.6f, test_r2=%.4f, test_mae=%.4f, "
                "test_pearson_r=%.4f, test_sp_l1_norm=%.6f.",
                test_metrics["val_loss"], test_metrics["val_r2"],
                test_metrics["val_mae"], test_metrics["val_pearson_r"],
                test_metrics["sp_l1_norm"],
            )

            test_json_path = (
                LOG_DIR_EXPERIMENT / f"test_metrics_{args.target}_{run_id}.json"
            )
            dump_json(
                test_json_path,
                {
                    "run_id": run_id,
                    "target": args.target,
                    "best_epoch": best_epoch,
                    "stopped_early": stopped_early,
                    "epochs_run": epoch,
                    "checkpoint_evaluated": str(eval_ckpt_path),
                    "test_loss": test_metrics["val_loss"],
                    "test_r2": test_metrics["val_r2"],
                    "test_mae": test_metrics["val_mae"],
                    "test_pearson_r": test_metrics["val_pearson_r"],
                    "test_sp_l1_norm": test_metrics["sp_l1_norm"],
                },
            )
            logger.info("Test metrics JSON written: %s", test_json_path)

        # Step 15: final log line.
        logger.info("Stage 1 training complete. run_id: %s", run_id)

    except Exception:
        logger.error("train_stage1.py failed:\n%s", traceback.format_exc())
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
