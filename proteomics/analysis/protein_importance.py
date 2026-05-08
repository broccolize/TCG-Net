"""
Per-protein importance analysis for the TCG-Net proteomics encoder.

Attribution method: gradient × input (integrated gradient approximation).
For each participant in the full dataset (train + val + test), we compute
|dy_p/dx * x| via a single backward pass, then average over all
participants to obtain a 2920-d importance vector — one score per protein.

A separate 1024-d artefact (sp_mean) is saved: the mean SE gate vector
across all participants, interpretable as latent-channel importance.

Outputs
-------
  analysis_outputs/protein_importance_<target>_<run_id>.csv
  analysis_outputs/sp_mean_<target>_<run_id>.npy
  logs/experiments/importance_<target>_<run_id>.json
  logs/process/process_protein_importance_<timestamp>.log
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from config import (  # noqa: E402
    ANALYSIS_DIR,
    BATCH_SIZE,
    DATASET_AGE_PATH,
    DATASET_IQ_PATH,
    GLOBAL_SEED,
    LOG_DIR_EXPERIMENT,
    LOG_DIR_PROCESS,
)
from models.encoder import ProteomicsEncoder  # noqa: E402
from models.aux_head import ProteomicsAuxHead  # noqa: E402
from utils.seed import set_global_seeds  # noqa: E402
from utils.logging_utils import dump_json, make_process_logger  # noqa: E402

TARGET_MAP = {
    "age": DATASET_AGE_PATH,
    "fluid_iq": DATASET_IQ_PATH,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the importance analysis script."""
    parser = argparse.ArgumentParser(
        description="Compute per-protein importance from a Stage 1 checkpoint."
    )
    parser.add_argument(
        "--target", required=True, choices=list(TARGET_MAP),
        help="Regression target used when training the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to a Stage 1 best checkpoint .pt file.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, dest="batch_size",
        help="Batch size for attribution forward/backward passes.",
    )
    parser.add_argument(
        "--seed", type=int, default=GLOBAL_SEED,
        help="Global random seed.",
    )
    return parser.parse_args()


def load_checkpoint(ckpt_path: Path, logger) -> dict:
    """
    Load and validate a Stage 1 checkpoint.

    Verifies that the required keys are present.  Returns the raw dict.
    """
    required_keys = {
        "encoder_state_dict", "aux_head_state_dict",
        "run_id", "config", "val_metrics",
    }
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    missing = required_keys - set(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint at {ckpt_path} missing keys: {sorted(missing)}")
    logger.info(
        "Checkpoint loaded: run_id=%s, best_epoch=%s, val_r2=%.4f.",
        ckpt["run_id"],
        ckpt["val_metrics"].get("epoch", "?"),
        ckpt["val_metrics"].get("val_r2", float("nan")),
    )
    return ckpt


def load_full_dataset(target: str, logger) -> dict:
    """
    Load the full dataset bundle for the given target.

    Returns the raw bundle dict (all of train + val + test combined via a
    simple integer range over the full x tensor).  Post-hoc interpretability
    uses all participants, not just the training split.
    """
    path = TARGET_MAP[target]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run data/preprocess.py then data/build_dataset.py first."
        )
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    n = bundle["x"].shape[0]
    logger.info(
        "Dataset loaded for importance analysis: path=%s, total_participants=%d.",
        path, n,
    )
    return bundle


def compute_attribution(
    encoder: ProteomicsEncoder,
    aux_head: ProteomicsAuxHead,
    x_all: torch.Tensor,
    batch_size: int,
    device: torch.device,
    logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient × input attribution and mean SE gate over all participants.

    For each batch:
      1. Enable grad on x_batch.
      2. Forward pass through encoder and aux_head.
      3. Backward on y_p.sum() to obtain dx.
      4. Accumulate |dx * x| (shape: 2920) and s_p (shape: 1024).

    Both encoder and aux_head must be in eval() mode before calling.
    Despite using .backward(), eval mode is preserved — it is inference,
    not training.

    Returns
    -------
    attribution_mean : torch.Tensor, shape (2920,)
        Per-protein mean importance across all participants.
    sp_mean : torch.Tensor, shape (1024,)
        Mean SE gate vector across all participants.
    """
    # Accumulators (CPU tensors to avoid OOM on large datasets).
    n_proteins = x_all.shape[1]
    attribution_sum = torch.zeros(n_proteins, dtype=torch.float64)
    sp_sum = torch.zeros(1024, dtype=torch.float64)
    n_participants = 0

    n_total = x_all.shape[0]
    for start in range(0, n_total, batch_size):
        x_batch = x_all[start: start + batch_size].float().to(device)
        x_batch.requires_grad_(True)

        z_p, s_p = encoder(x_batch)
        y_p = aux_head(z_p)

        encoder.zero_grad()
        aux_head.zero_grad()
        y_p.sum().backward()

        # grad is None if no path contributed — guard defensively.
        if x_batch.grad is None:
            raise RuntimeError(
                "x_batch.grad is None after backward. "
                "Check that encoder parameters have requires_grad=True."
            )

        grad_input = (x_batch.grad * x_batch.detach()).abs()  # (B, 2920)
        attribution_sum += grad_input.sum(dim=0).double().cpu()
        sp_sum          += s_p.detach().double().sum(dim=0).cpu()
        n_participants  += x_batch.shape[0]

    attribution_mean = (attribution_sum / n_participants).float()  # (2920,)
    sp_mean          = (sp_sum          / n_participants).float()  # (1024,)

    logger.info(
        "Attribution computation complete. n_participants=%d.", n_participants
    )
    return attribution_mean, sp_mean


def main() -> None:
    """
    Full importance analysis pipeline.

    Execution order follows rebuild plan §2.13:
    seeds → logger → checkpoint → models → dataset → device →
    attribution → save CSV → save sp_mean → summary JSON.
    """
    args = parse_args()

    # Step 1: seeds.
    set_global_seeds(args.seed)

    # Step 2: process logger.
    logger, log_path = make_process_logger("protein_importance", LOG_DIR_PROCESS)

    try:
        logger.info("Script: protein_importance.py")
        logger.info("Target: %s", args.target)
        logger.info("Checkpoint: %s", args.checkpoint)

        # Step 3: load checkpoint.
        ckpt = load_checkpoint(args.checkpoint, logger)
        ckpt_run_id = ckpt["run_id"]

        # Step 4: instantiate and load models (CPU first, moved to device below).
        encoder  = ProteomicsEncoder()
        aux_head = ProteomicsAuxHead()
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        aux_head.load_state_dict(ckpt["aux_head_state_dict"])
        encoder.eval()
        aux_head.eval()
        logger.info("Encoder and aux_head loaded from checkpoint.")

        # Step 5: full dataset (all splits combined for post-hoc analysis).
        bundle = load_full_dataset(args.target, logger)
        x_all = bundle["x"].float()
        feature_names = list(bundle["features"])
        n_total = x_all.shape[0]
        logger.info("Total participants used for attribution: %d.", n_total)

        # Step 6: device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.to(device)
        aux_head.to(device)
        logger.info("Device: %s", device)

        # Step 7: gradient × input attribution.
        attribution_mean, sp_mean = compute_attribution(
            encoder, aux_head, x_all, args.batch_size, device, logger
        )

        # Step 8: build DataFrame, sort, save CSV.
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        importance_csv_path = (
            ANALYSIS_DIR / f"protein_importance_{args.target}_{ckpt_run_id}.csv"
        )
        df = pd.DataFrame({
            "Protein": feature_names,
            "Importance": attribution_mean.numpy(),
        })
        df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
        df.to_csv(importance_csv_path, index=False)

        top10_proteins = df["Protein"].iloc[:10].tolist()
        logger.info(
            "Protein importance CSV saved: %s. Top 5 proteins: %s.",
            importance_csv_path,
            ", ".join(df["Protein"].iloc[:5].tolist()),
        )

        # Step 9: save sp_mean.
        sp_mean_path = (
            ANALYSIS_DIR / f"sp_mean_{args.target}_{ckpt_run_id}.npy"
        )
        np.save(sp_mean_path, sp_mean.numpy())
        logger.info(
            "sp_mean saved: %s (shape=%s).", sp_mean_path, tuple(sp_mean.shape)
        )

        # Step 10: process summary JSON.
        summary_json_path = (
            LOG_DIR_EXPERIMENT / f"importance_{args.target}_{ckpt_run_id}.json"
        )
        dump_json(
            summary_json_path,
            {
                "run_id": ckpt_run_id,
                "target": args.target,
                "checkpoint_path": str(args.checkpoint),
                "n_participants": n_total,
                "attribution_method": "gradient_times_input",
                "protein_importance_file": str(importance_csv_path),
                "sp_mean_file": str(sp_mean_path),
                "top_10_proteins": top10_proteins,
            },
        )
        logger.info("Importance summary JSON written: %s", summary_json_path)

        # Step 11: final log.
        logger.info(
            "Importance analysis complete. Target: %s. run_id: %s",
            args.target, ckpt_run_id,
        )

    except Exception:
        logger.error("protein_importance.py failed:\n%s", traceback.format_exc())
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
