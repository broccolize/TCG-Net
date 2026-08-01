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


# The instance-2 cognitive battery — administered AT the imaging visit, so
# contemporaneous with the connectome. Candidates for a composite g-factor, which
# would be more reliable than 20016-2.0 alone (test-retest 0.636 on this cohort)
# and would raise the attenuation ceiling on every model in the project.
COGNITIVE_INSTANCE2 = [
    "20016-2.0",   # fluid intelligence
    "6350-2.0",    # trail making B (executive)
    "6373-2.0",    # matrix pattern reasoning
    "4282-2.0",    # numeric memory
    "20023-2.0",   # reaction time
    "20018-2.0",   # prospective memory
]


def do_columns(columns: list[str], target: str) -> None:
    """Coverage for one or more columns. Reads only eid plus that column."""
    eids = joint_eids(target)
    eid_set = set(eids.tolist())
    headers = {p: list(pd.read_csv(p, nrows=0).columns) for p in phesant_files()}

    print(f"target:  {target}   joint cohort: {eids.size:,d} participants\n")
    print(f"{'column':<14}{'overall':>12}{'in joint':>11}{'cov%':>8}  file")
    print("-" * 68)

    complete: np.ndarray | None = None
    for column in columns:
        path = next((p for p, cols in headers.items() if column in cols), None)
        if path is None:
            print(f"{column:<14}{'— not found in any phesant file —':>31}")
            continue
        df = pd.read_csv(path, usecols=["eid", column]).dropna(subset=[column])
        present = np.isin(eids, df["eid"].to_numpy(np.int64))
        n_joint = int(present.sum())
        pct = 100.0 * n_joint / eids.size
        flag = "  !" if pct < 50 else ""
        print(f"{column:<14}{df.shape[0]:>12,d}{n_joint:>11,d}{pct:>7.1f}%  "
              f"{path.name}{flag}")
        complete = present if complete is None else (complete & present)

    if complete is not None and len(columns) > 1:
        n = int(complete.sum())
        print("-" * 68)
        print(f"{'ALL of them':<14}{'':>12}{n:>11,d}{100.0 * n / eids.size:>7.1f}%")
        print("\nThat last row is what a complete-case composite would be built on.\n"
              "Items can also be scored pairwise or imputed if it is too lossy.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true",
                   help="Header-scan for candidate outcome phenotypes.")
    p.add_argument("--pattern", default=None,
                   help="Custom regex for --list, instead of the built-in set.")
    p.add_argument("--column", nargs="+", default=None,
                   help="Report joint-cohort coverage for these exact columns.")
    p.add_argument("--cognitive", action="store_true",
                   help="Coverage for the whole instance-2 cognitive battery, "
                        "plus the complete-case count for a composite g-factor.")
    p.add_argument("--target", choices=("age", "fluid_iq"), default="age",
                   help="Which joint dataset defines the cohort for --column.")
    args = p.parse_args()

    if args.cognitive:
        do_columns(COGNITIVE_INSTANCE2, args.target)
    elif args.column:
        do_columns(args.column, args.target)
    elif args.list or args.pattern:
        do_list(args.pattern)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
