"""
Forward-pass shape smoke test for the TCG-Net proteomics encoder pipeline.

Verifies that SEBlock, ProteomicsEncoder, and ProteomicsAuxHead all produce
the expected output shapes and value ranges without requiring any training
data, GPU, or pre-trained weights.

Run with:
    python TCG-Net/proteomics/tests/test_shapes.py

or from within the proteomics package directory:
    python tests/test_shapes.py

All tests use random tensors seeded with torch.manual_seed(42) so results
are reproducible.  No pytest dependency — plain Python exit codes.

Exit codes:
    0  all tests passed
    1  one or more tests failed
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Make the proteomics package root importable regardless of where the script
# is invoked from.
# ---------------------------------------------------------------------------
_PROTEOMICS_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PROTEOMICS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG_DIR))

from models.se_block import SEBlock              # noqa: E402
from models.encoder import ProteomicsEncoder    # noqa: E402
from models.aux_head import ProteomicsAuxHead   # noqa: E402

# ---------------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------------
torch.manual_seed(42)                           # reproducible random tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _pass(name: str) -> None:
    """Print a PASS line for a named test."""
    print(f"  PASS  {name}")


def _fail(name: str, detail: str) -> None:
    """Print a FAIL line for a named test."""
    print(f"  FAIL  {name}")
    print(f"        {detail}")


def _check_shape(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    results: list[bool],
) -> None:
    """Assert shape equality; record result in ``results``."""
    actual = tuple(tensor.shape)
    if actual == expected:
        _pass(f"{name}: shape {actual}")
        results.append(True)
    else:
        _fail(name, f"expected {expected}, got {actual}")
        results.append(False)


def _check_range(
    name: str,
    tensor: torch.Tensor,
    lo: float,
    hi: float,
    results: list[bool],
) -> None:
    """Assert all values in [lo, hi]; record result."""
    lo_ok = bool((tensor >= lo).all().item())
    hi_ok = bool((tensor <= hi).all().item())
    if lo_ok and hi_ok:
        _pass(f"{name}: all values in [{lo}, {hi}]")
        results.append(True)
    else:
        _fail(name, f"values outside [{lo}, {hi}]; "
                    f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
        results.append(False)


def _check_not_all_zero(
    name: str,
    tensor: torch.Tensor,
    results: list[bool],
) -> None:
    """Assert tensor is not identically zero; record result."""
    if bool(tensor.abs().sum().item() > 0):
        _pass(f"{name}: is not all-zeros")
        results.append(True)
    else:
        _fail(name, "tensor is identically zero")
        results.append(False)


def _check_scalar(
    name: str,
    tensor: torch.Tensor,
    results: list[bool],
) -> None:
    """Assert tensor is a 0-d scalar; record result."""
    if tensor.dim() == 0:
        _pass(f"{name}: is a scalar tensor")
        results.append(True)
    else:
        _fail(name, f"expected 0-d scalar, got shape {tuple(tensor.shape)}")
        results.append(False)


def _check_positive(
    name: str,
    tensor: torch.Tensor,
    results: list[bool],
) -> None:
    """Assert tensor value > 0; record result."""
    if bool(tensor.item() > 0):
        _pass(f"{name}: value > 0 (={tensor.item():.6f})")
        results.append(True)
    else:
        _fail(name, f"expected > 0, got {tensor.item():.6f}")
        results.append(False)


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_se_block(results: list[bool]) -> torch.Tensor:
    """
    Test 1: SEBlock forward pass.

    Verifies that h_tilde_p and s_p have the correct shapes and that s_p
    values are in [0, 1] (Sigmoid output invariant).
    """
    print("\nTest 1: SEBlock forward pass")
    se = SEBlock(in_dim=1024, reduction_dim=256).to(device).eval()
    x = torch.randn(4, 1024, device=device)   # batch=4, hidden=1024

    with torch.no_grad():
        h_tilde_p, s_p = se(x)

    # Shape checks
    _check_shape("h_tilde_p", h_tilde_p, (4, 1024), results)
    _check_shape("s_p", s_p, (4, 1024), results)
    # Sigmoid invariant: s_p values must lie in (0, 1).
    _check_range("s_p values", s_p, 0.0, 1.0, results)

    return s_p  # returned for downstream tests if needed


def test_encoder_forward(results: list[bool]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Test 2: ProteomicsEncoder forward pass.

    Verifies z_p and s_p shapes, s_p range, and that z_p is not degenerate.
    """
    print("\nTest 2: ProteomicsEncoder forward pass")
    encoder = ProteomicsEncoder().to(device).eval()
    x = torch.randn(4, 2920, device=device)   # batch=4, proteins=2920

    with torch.no_grad():
        z_p, s_p = encoder(x)

    _check_shape("z_p", z_p, (4, 128), results)
    _check_shape("s_p", s_p, (4, 1024), results)
    _check_not_all_zero("z_p", z_p, results)
    _check_range("s_p values", s_p, 0.0, 1.0, results)

    return z_p, s_p


def test_aux_head(z_p: torch.Tensor, results: list[bool]) -> None:
    """
    Test 3: ProteomicsAuxHead forward pass.

    Takes the z_p from test 2 and checks y_p shape is (B, 1).
    """
    print("\nTest 3: ProteomicsAuxHead forward pass")
    aux_head = ProteomicsAuxHead().to(device).eval()

    with torch.no_grad():
        y_p = aux_head(z_p)

    _check_shape("y_p", y_p, (z_p.shape[0], 1), results)


def test_full_pipeline(results: list[bool]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Test 4: Full pipeline (encoder → aux_head) with batch size 8.

    Verifies that z_p, s_p, and y_p all have correct shapes when the
    encoder and head are wired together exactly as in training.
    """
    print("\nTest 4: Full pipeline (encoder + aux_head, batch=8)")
    encoder = ProteomicsEncoder().to(device).eval()
    aux_head = ProteomicsAuxHead().to(device).eval()
    x = torch.randn(8, 2920, device=device)

    with torch.no_grad():
        z_p, s_p = encoder(x)
        y_p = aux_head(z_p)

    _check_shape("z_p", z_p, (8, 128), results)
    _check_shape("s_p", s_p, (8, 1024), results)
    _check_shape("y_p", y_p, (8, 1), results)

    return z_p, s_p, y_p


def test_batch_size_independence(results: list[bool]) -> None:
    """
    Test 5: Batch-size independence.

    Runs the full pipeline at batch sizes 1, 16, 64, 128 and checks
    that output shapes scale correctly with the batch dimension.
    """
    print("\nTest 5: Batch-size independence (B = 1, 16, 64, 128)")
    encoder = ProteomicsEncoder().to(device).eval()
    aux_head = ProteomicsAuxHead().to(device).eval()

    for B in [1, 16, 64, 128]:
        x = torch.randn(B, 2920, device=device)
        with torch.no_grad():
            z_p, s_p = encoder(x)
            y_p = aux_head(z_p)

        # All three output tensors must have the batch dim equal to B.
        _check_shape(f"z_p (B={B})", z_p, (B, 128), results)
        _check_shape(f"s_p (B={B})", s_p, (B, 1024), results)
        _check_shape(f"y_p (B={B})", y_p, (B, 1), results)


def test_sp_l1_norm(results: list[bool]) -> None:
    """
    Test 6: s_p L1 norm computation.

    Verifies that ||s_p||_1 (per-sample L1 mean, batch-averaged) is a
    scalar tensor with a positive value.  This is the quantity that
    ProteomicsStage1Loss will add as a regularisation term.
    """
    print("\nTest 6: s_p L1 norm (per-sample mean, batch-averaged)")
    encoder = ProteomicsEncoder().to(device).eval()
    x = torch.randn(8, 2920, device=device)

    with torch.no_grad():
        _, s_p = encoder(x)

    # Per-sample L1 mean: mean over the hidden dimension for each sample.
    per_sample_l1 = s_p.abs().mean(dim=1)   # (B,)
    # Batch average: scalar that is independent of batch size.
    sp_l1_norm = per_sample_l1.mean(dim=0)  # scalar

    _check_scalar("sp_l1_norm", sp_l1_norm, results)
    _check_positive("sp_l1_norm", sp_l1_norm, results)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("TCG-Net proteomics encoder — shape smoke tests")
    print(f"Device: {device}")
    print("=" * 60)

    results: list[bool] = []   # True = pass, False = fail

    # Run all tests. Each test appends to `results`.
    test_se_block(results)
    z_p, s_p = test_encoder_forward(results)
    test_aux_head(z_p, results)
    test_full_pipeline(results)
    test_batch_size_independence(results)
    test_sp_l1_norm(results)

    # Summary
    n_pass = sum(results)
    n_fail = len(results) - n_pass

    print("\n" + "=" * 60)
    print(f"Results: {n_pass} passed, {n_fail} failed  (total {len(results)} checks)")

    if n_fail == 0:
        print("All shape tests passed. Encoder pipeline is ready for training.")
        sys.exit(0)
    else:
        print(f"{n_fail} check(s) FAILED — fix the relevant model file before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
