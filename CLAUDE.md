# TCG-Net — repository instructions

## Git and commit attribution

**Commits and pushes are made as `broccolize` only.**

- Do **not** add a `Co-Authored-By: Claude ...` trailer to commit messages.
- Do **not** add "Generated with Claude Code" or any equivalent footer to commits.
- Do **not** change `user.name` or `user.email`.

This applies to every commit in this repository, including ones written entirely
by an assistant. Pull request bodies follow the same rule.

## Branching

- Never commit directly to `main`.
- Work on a topic branch (`fix/...`, `feat/...`) and push that branch.
- **Do not merge to `main` until the change has actually been run and the results
  inspected.** Several defects in this repo's history were code that ran without
  erroring but produced meaningless numbers, so "it imports cleanly" is not
  evidence that a change is correct.

## Long-running commands

Training, preprocessing and cross-validation runs take hours. Do not launch them
automatically. Provide the exact command, fully filled in, and state which
directory and conda environment to run it from (normally `~/TCG-Net` and
`tcg_net` on the server).

## Working notes

- [`LOG.md`](LOG.md) — dated log of every change made. Append a new entry at the
  top for each working session; keep it self-contained enough to be read with no
  prior conversation context.
- [`FIX_PLAN.md`](FIX_PLAN.md) — prioritised list of outstanding defects. Tick
  items off there when they land in `LOG.md`.

## Data protection

The raw UK Biobank data under `uk_BIOBANK_DATA/` is read-only confidential
research data. Scripts must import its path from `proteomics/config.py`, never
hardcode it, and must never print or log individual participant values — only
aggregate counts, shapes and summary statistics.

## Caches

`compute_graph_features` caches to disk and validates only the **subject count**,
not the split or the feature definition. Any change to the feature computation or
to `train_idx` requires deleting the cache files by hand, or the stale ones are
silently reloaded.
