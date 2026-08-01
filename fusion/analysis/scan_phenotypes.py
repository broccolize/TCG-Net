"""
Find candidate outcome phenotypes and measure their coverage on the joint cohort.

Why
---
The §12 aging-discordance analysis needs outcomes to compare the brain-dominant,
blood-dominant and balanced groups on — §12 lists vascular risk, inflammatory
burden, mental health and cognition. Whether that analysis is feasible depends
entirely on how many of the ~5,300 joint participants actually have those
measurements, and it is cheaper to check before designing around them.

Two modes:

    --list              header-scan every phesantallphenos_*.csv for candidate
                        columns matching a set of named patterns
    --column 30710-0.0  count how many joint participants have that column

CRITICAL DATA PROTECTION
------------------------
Reads only the ``eid`` column and the one requested column, and prints counts
and percentages only. No individual participant value is ever loaded into a
printed object. Header scans use ``nrows=0``.

Usage (server, from ~/TCG-Net, env tcg_net):

    conda run --no-capture-output -n tcg_net python -u fusion/analysis/scan_phenotypes.py --list
    conda run --no-capture-output -n tcg_net python -u fusion/analysis/scan_phenotypes.py --column 30710-0.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PROTEOMICS_PKG = _REPO_ROOT / "proteomics"
if str(_PROTEOMICS_PKG) not in sys.path:
    sys.path.insert(0, str(_PROTEOMICS_PKG))

from config import HOME_ROOT, SHARED_DATA_DIR  # noqa: E402

PHESANT_DIR = HOME_ROOT / "uk_BIOBANK_DATA" / "table"

# Candidate outcomes for the subtype comparison in §12. UKB field ids, matched
# loosely against the header so instance/array suffixes still hit.
CANDIDATES: dict[str, str] = {
    "CRP / inflammation":      r"^30710-",
    "IGF-1":                   r"^30770-",
    "systolic BP":             r"^4080-",
    "diastolic BP":            r"^4079-",
    "BMI":                     r"^21001-",
    "body fat %":              r"^23099-",
    "smoking status":          r"^20116-",
    "alcohol frequency":       r"^1558-",
    "diabetes diagnosed":      r"^2443-",
    "total cholesterol":       r"^30690-",
    "HDL cholesterol":         r"^30760-",
    "triglycerides":           r"^30870-",
    "HbA1c":                   r"^30750-",
    "depressed mood freq":     r"^2050-",
    "neuroticism score":       r"^20127-",
    "education qualifications": r"^6138-",
    "reaction time":           r"^20023-",
    "prospective memory":      r"^20018-",
    "pairs matching":          r"^399-",
    "trail making B":          r"^6350-",
    "matrix pattern":          r"^6373-",
    "numeric memory":          r"^4282-",
    "fluid intelligence":      r"^20016-",
    "age at assessment":       r"^21003-",
}


def phesant_files() -> list[Path]:
    files = sorted(PHESANT_DIR.glob("phesantallphenos_*.csv"))
    if not files:
        raise FileNotFoundError(f"No phesantallphenos_*.csv under {PHESANT_DIR}")
    return files


def joint_eids(target: str) -> np.ndarray:
    path = SHARED_DATA_DIR / f"dataset_joint_{target}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Joint dataset not found: {path}")
    b = torch.load(path, map_location="cpu", weights_only=False)
    return b["eid"].numpy().astype(np.int64)


def do_list(pattern: str | None) -> None:
    """Header-scan only (nrows=0) — no data values are read."""
    pats = {"custom": pattern} if pattern else CANDIDATES
    files = phesant_files()
    print(f"Scanning {len(files)} phesantallphenos file(s) in {PHESANT_DIR}\n")

    found: dict[str, list[str]] = {k: [] for k in pats}
    where: dict[str, str] = {}
    for path in files:
        cols = list(pd.read_csv(path, nrows=0).columns)
        for label, rx in pats.items():
            hits = [c for c in cols if re.search(rx, c)]
            if hits:
                found[label].extend(hits)
                where.setdefault(label, path.name)

    print(f"{'phenotype':<26}{'n cols':>7}  {'example columns':<34}{'file'}")
    print("-" * 96)
    for label, cols in found.items():
        if not cols:
            print(f"{label:<26}{0:>7}  {'— not present —':<34}")
            continue
        ex = ", ".join(sorted(cols)[:3])
        print(f"{label:<26}{len(cols):>7}  {ex:<34}{where[label]}")
    print("\nPick a column and re-run with --column <name> to get joint-cohort coverage.")


def do_column(column: str, target: str) -> None:
    eids = joint_eids(target)
    eid_set = set(eids.tolist())
    for path in phesant_files():
        if column not in pd.read_csv(path, nrows=0).columns:
            continue
        # Only eid and the requested column are ever loaded.
        df = pd.read_csv(path, usecols=["eid", column]).dropna(subset=[column])
        n_overall = int(df.shape[0])
        n_joint = int(df["eid"].isin(eid_set).sum())
        print(f"column:  {column}   (file: {path.name})")
        print(f"target:  {target}   joint cohort: {eids.size:,d} participants")
        print(f"non-missing overall:        {n_overall:,d}")
        print(f"non-missing in joint cohort:{n_joint:>10,d}  "
              f"({100.0 * n_joint / eids.size:.1f}% of the joint cohort)")
        if n_joint < 0.5 * eids.size:
            print("  ! under 50% coverage — a subtype comparison on this column "
                  "would lose most of the cohort")
        return
    print(f"Column '{column}' not found in any phesantallphenos file.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true",
                   help="Header-scan for candidate outcome phenotypes.")
    p.add_argument("--pattern", default=None,
                   help="Custom regex for --list, instead of the built-in set.")
    p.add_argument("--column", default=None,
                   help="Report joint-cohort coverage for this exact column.")
    p.add_argument("--target", choices=("age", "fluid_iq"), default="age",
                   help="Which joint dataset defines the cohort for --column.")
    args = p.parse_args()

    if args.column:
        do_column(args.column, args.target)
    elif args.list or args.pattern:
        do_list(args.pattern)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
