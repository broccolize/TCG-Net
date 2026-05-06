"""
Reproducibility utilities for the TCG-Net proteomics pipeline.

This module must be imported and ``set_global_seeds(GLOBAL_SEED)`` called
as the **first action** in ``main()`` of every entry script in the pipeline
(preprocessing, dataset construction, training, cross-validation, analysis).
That guarantees identical random draws across modules — Python's ``random``,
NumPy, PyTorch CPU, and PyTorch CUDA — and disables non-deterministic cuDNN
algorithm selection.

Three helpers are exposed:

    set_global_seeds(seed)        — call once at process start.
    make_split_generator(seed)    — pass into ``random_split`` / DataLoader.
    seed_worker(worker_id)        — pass into ``DataLoader(worker_init_fn=...)``.

Together they cover the four sources of nondeterminism in a typical
PyTorch training run: global RNG, the split generator, per-worker NumPy /
random seeding, and cuDNN kernel selection.
"""

import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """
    Seed every RNG the proteomics pipeline touches and force deterministic cuDNN.

    Must be the first call in ``main()`` of every entry script. Seeding
    after data has been loaded or modules have been imported is too late:
    NumPy / Python ``random`` may have already been consumed by upstream
    code, and PyTorch may have already initialised parameters.
    """
    # Python's built-in pseudo-random generator (used by sklearn, etc.).
    random.seed(seed)
    # NumPy global RNG (used by sklearn, pandas sample, np.random.choice, ...).
    np.random.seed(seed)
    # PyTorch CPU RNG — covers tensor init, dropout masks, etc.
    torch.manual_seed(seed)
    # PyTorch CUDA RNG on every visible device. Safe to call even with no GPU.
    torch.cuda.manual_seed_all(seed)
    # Force cuDNN to pick the same convolution algorithm every run.
    # `deterministic=True` together with `benchmark=False` is the documented
    # recipe; setting only one is not enough.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_split_generator(seed: int) -> torch.Generator:
    """
    Return a freshly seeded ``torch.Generator`` for use in ``random_split``
    and ``DataLoader(generator=...)``.

    Using a dedicated generator (rather than relying on the global one) means
    the partition is invariant to whatever random work happened before the
    split was constructed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """
    Per-worker initializer for ``DataLoader(worker_init_fn=seed_worker)``.

    PyTorch already seeds each worker's ``torch`` RNG deterministically via
    ``torch.initial_seed()``, but it does **not** propagate that seed into
    NumPy or the Python ``random`` module. Without this hook, two workers
    would pull identical NumPy random draws (e.g. for augmentation), and
    re-runs would not be reproducible.

    We derive a 32-bit seed from ``torch.initial_seed()`` (which is unique
    per worker because PyTorch offsets it by ``worker_id``) and use it to
    seed both NumPy and Python random — so each worker is internally
    deterministic and distinct from its siblings.
    """
    # torch.initial_seed() is 64-bit; NumPy needs a uint32, hence the modulo.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
