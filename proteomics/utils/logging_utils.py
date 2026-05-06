"""
Logging helpers for the TCG-Net proteomics pipeline.

The pipeline emits four log types. This module provides the helpers that
produce each one consistently across all entry scripts. All timestamps are
in UTC (per the central decision in ``config.TIMEZONE``).

    1. Process logs (.log)
       Plain-text, human-readable audit trail of one script invocation.
       Format: ``[YYYY-MM-DD HH:MM:SS] [LEVEL] message``.
       Path:   ``logs/process/process_<script_name>_<YYYYmmdd_HHMMSS>.log``
       Helper: ``make_process_logger``.

    2. Experiment logs (.json)
       One JSON file per training run capturing the full configuration:
       run_id, seeds, architecture, hyperparameters, dataset sizes, paths.
       Written once before the first training epoch, never updated.
       Path:   ``logs/experiments/experiment_<run_id>_<YYYYmmdd_HHMMSS>.json``
       Helper: ``dump_json``.

    3. Training logs (.csv)
       One CSV per training run with one row per epoch:
       ``epoch, train_loss, val_loss, val_r2, val_mae, val_pearson_r, sp_l1_norm``.
       Each row is opened-appended-closed atomically so partial runs
       still produce a usable log.
       Path:   ``logs/training/train_log_<target>_<YYYYmmdd_HHMMSS>.csv``
       Helper: ``append_csv_row``.

    4. Preprocessing logs (.json)
       Single-shot JSON summary written by ``preprocess.py`` and
       ``build_dataset.py`` after artefacts have been persisted.
       Path:   ``logs/preprocessing/preprocess_log_<YYYYmmdd_HHMMSS>.json``
               ``logs/preprocessing/dataset_build_<target>_<YYYYmmdd_HHMMSS>.json``
       Helper: ``dump_json``.

The ``make_run_id`` helper produces the run identifier embedded in
experiment / training / checkpoint filenames.
"""

import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


# Format used in all filenames that embed a timestamp. UTC.
_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def _utc_timestamp() -> str:
    """Return the current UTC time formatted as ``YYYYmmdd_HHMMSS``."""
    return datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)


def make_run_id() -> str:
    """
    Produce a unique run identifier for a single training invocation.

    Format: ``YYYYmmdd_HHMMSS_<8-char hex>``. The timestamp prefix keeps
    runs lexicographically sortable; the hex suffix from ``uuid4`` makes
    two runs that fire within the same second still distinguishable.
    Timestamp is in UTC.
    """
    return f"{_utc_timestamp()}_{uuid4().hex[:8]}"


def make_process_logger(script_name: str, log_dir: Path) -> tuple[logging.Logger, Path]:
    """
    Build a logger that writes simultaneously to a per-invocation .log file
    and to stdout, using a fixed format.

    Args:
        script_name: Basename of the entry script without the ``.py`` extension
            (e.g. ``"train_stage1"``). Used both as the log file's filename
            stem and as the logger's name.
        log_dir: Directory where the .log file will be written. Created if
            it does not exist. Conventionally ``config.LOG_DIR_PROCESS``.

    Returns:
        ``(logger, log_path)`` — the configured ``logging.Logger`` and the
        absolute path of the file it is writing to. The caller embeds
        ``log_path`` in the experiment JSON so process and experiment logs
        can be cross-referenced.

    The file handler is unbuffered (``delay=False``) and Python's default
    ``FileHandler`` flushes on every emit, so a killed process leaves a
    .log that ends at the last successfully emitted line.
    """
    # Ensure the destination exists. parents=True makes nested logs/process
    # directories materialise on first use.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # File path: process_<script_name>_<YYYYmmdd_HHMMSS>.log (UTC).
    timestamp = _utc_timestamp()
    log_path = log_dir / f"process_{script_name}_{timestamp}.log"

    # Each script invocation gets its own logger instance. We name it after
    # the script + timestamp so that re-invoking does not pile handlers onto
    # the same singleton (which would duplicate every log line).
    logger = logging.getLogger(f"{script_name}_{timestamp}")
    logger.setLevel(logging.INFO)
    # Don't bubble up to the root logger (which may have its own handlers).
    logger.propagate = False

    # Shared formatter — exact format mandated by the rebuild plan §4.4.
    # ``converter = time.gmtime`` forces ``%(asctime)s`` to render in UTC,
    # matching the UTC timestamps in filenames and run IDs (decision 9).
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = time.gmtime

    # FileHandler: writes to disk. delay=False means the file is opened
    # immediately so the path is valid even if the script never logs.
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8", delay=False)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # StreamHandler: mirrors everything to stdout for live monitoring.
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_path


def dump_json(path: Path, payload: dict) -> None:
    """
    Write a dict to disk as pretty-printed JSON.

    Indented with two spaces and ``sort_keys=False`` so the field order in
    the source dict is preserved (which is meaningful in our experiment /
    preprocessing JSONs — readers expect ``run_id`` and ``timestamp`` first).

    Creates parent directories if they do not exist. Re-raises after logging
    on any I/O or serialisation failure: a silent failure here would leave
    the pipeline running while believing its config has been persisted.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
    except Exception:
        # Best-effort error log; we still re-raise so the caller's top-level
        # try/except can record the failure in the process .log and exit
        # non-zero.
        logging.getLogger(__name__).error("Failed to write JSON to %s", path)
        raise


def append_csv_row(path: Path, row: dict, header: list[str]) -> None:
    """
    Append a single row to a CSV, writing the header lazily.

    The header is written iff the file does not exist or is empty. This
    keeps the function idempotent across the boundary of "first epoch" vs
    "subsequent epochs": the training loop calls this once per epoch and
    does not need to track whether the header has been written yet.

    The file is opened, the row is written, the underlying file object is
    flushed, and then closed — all within this call. That guarantee is
    what makes partial runs leave a usable CSV: any kill (SIGINT, OOM,
    reboot) leaves exactly the rows that completed.

    Args:
        path: Destination CSV path. Parent directories are created if missing.
        row: Mapping from column name → value. Must include every key in
            ``header`` (extra keys are ignored by ``DictWriter``).
        header: Column names in the order they should appear.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Decide whether to emit the header line. The two conditions cover
    # "file doesn't exist yet" and "file exists but is zero bytes" (which
    # can happen if a prior crash created the file before any row landed).
    write_header = (not path.exists()) or path.stat().st_size == 0

    # Open in append mode with newline="" per csv module recommendation
    # (avoids spurious blank lines on Windows). The `with` block flushes
    # and closes on exit, which is the partial-run-safety guarantee.
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        # Explicit flush before the with-block closes, belt and braces in
        # case the underlying buffering policy changes.
        f.flush()
