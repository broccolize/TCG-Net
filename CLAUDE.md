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

## Assistant reply format

Every reply that follows a run, a pasted result, or a question uses this order,
with nothing inserted between the two parts:

**1. Findings first.** What the numbers say, what changed, what is now known,
and what a previous conclusion has to give up. Lead with the result, not the
process. If a claim was overturned, say so plainly and say why.

**2. Then a numbered command procedure.** Every command the user should run
next, in the order to run them, each in its own fenced `bash` block, one command
per block, fully filled in with no placeholders. For each step state:

- what it answers or produces,
- roughly how long it takes,
- whether it must be run sequentially (Stage 2 writes a fixed `best_model.pt`,
  so two Stage-2 jobs must never run at once).

Commands run on the server from `~/TCG-Net` in the `tcg_net` environment, via
`conda run --no-capture-output -n tcg_net python -u ...`. Heredocs and
`python - <<'PY'` do not survive `conda run`; put anything non-trivial in a
committed script and call that instead. Always include the `git pull` step when
new commits have been pushed during the session.

Do not append optional tangents, alternatives, or open questions after the
command list — those belong in the findings section above it.

## Working notes

- [`LOG.md`](LOG.md) — dated log of every change made. Append a new entry at the
  top for each working session; keep it self-contained enough to be read with no
  prior conversation context.
- [`PROMPT.md`](PROMPT.md) — session handoff: current state, open decisions, and
  the traps specific to this project. Read this first after a context reset.
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
