# Session handoff — read this first

Written 2026-07-30 at the end of a long working session. Assume the reader has
**no** prior conversation context.

**Read in this order:** this file → the top entry of [`LOG.md`](LOG.md) →
[`FIX_PLAN.md`](FIX_PLAN.md). [`CLAUDE.md`](CLAUDE.md) has the repo rules
(commit attribution, no merging to `main` before a change has been run).

---

## Where things stand in one paragraph

The proteomics and fusion pipelines were reviewed line by line and had ~30 real
defects, four of which invalidated published numbers. The critical ones are fixed
and **verified in real runs**, not just in tests. Fluid IQ Stage 1 improved from
R² 0.067 to 0.107 and is now leakage-free; the joint cohort nearly tripled; Stage 2
produced its first genuinely held-out number. **The code is now correct.** What it
reveals is that the central research claim — that two-branch cooperative fusion
beats a single modality — is *not supported* for fluid IQ. That is a scientific
finding, not a bug, and deciding what to do about it is the main open question.

## Branch

`fix/critical-early-stopping-and-phi-int` — 12 commits, pushed, **not merged to
`main`**. Per `CLAUDE.md`, do not merge until results have been run and inspected.

## Current numbers

| | before | now |
|---|---|---|
| Stage 1 fluid IQ, test R² | 0.067 | **0.107 ± 0.009** (4 seeds, leakage-free) |
| Joint cohort | 1,754 | **4,950** (instance-2 labels) |
| Stage 2 fluid IQ, **held-out** test R² | never measured | **0.068** (n=747) |

Stage-2 branch decomposition (the important one):

```
fused 0.0683 | fMRI-only 0.0669 | prot-only -0.0007 | joint-head 0.0689
```

- The fusion adds **0.0014** over fMRI alone — noise. The joint model is the fMRI model.
- `joint-head` (`y_hat_fp`, a plain MLP on concatenated embeddings) **matches** the
  gated output. That is a naive-fusion baseline sitting inside the model, and
  cooperative Shapley fusion does not beat it.
- `prot-only` R² ≈ 0 but **r ≈ 0.17** — weak signal, not absent. R² alone misleads
  here because a shrunk predictor ranks fine while scoring ~0.

Proteomics → fluid IQ signal decomposes cleanly:

| | Pearson r |
|---|---|
| Full proteomics cohort, instance-0 | 0.345 |
| Joint/imaging cohort, instance-0 | 0.254 (cohort restriction) |
| Joint/imaging cohort, instance-2 | 0.171 (+ 9-year blood-to-scan gap) |

The second step is fully explained by attenuation: 0.254 × 0.636 (test-retest) =
0.162 vs 0.171 observed.

---

## OPEN DECISIONS — none of these are settled

**1. What claim is actually being tested?** This governs everything else and was
never stated in the design docs. Three readings, with different verdicts:

- (a) *the fusion method works* → current fluid IQ result is a **null**
- (b) *proteomics adds to fMRI for cognition* → also a **null** (r ≈ 0.17, adds nothing)
- (c) *individualized cognitive prediction by any means* → you have a working
  brain-based predictor at R² 0.068 and proteomics is a side story

**2. `l_shapley` was redefined, not repaired.** The original was a tautology
(identically 0). It now reads `(y_hat - (y0 + phi_f + phi_p))²` — Shapley
efficiency applied to the fused prediction. **This has not been checked against
§6.4 of the spec PDF, which the assistant never saw.** If you want zero behavioural
change beyond making `phi_int` real, pass `lambda_shapley=0`. Note the term now
carries real gradient where it carried none, so new runs are not comparable to any
run before 2026-07-30.

**3. instance-0 vs instance-2.** No configuration has both modalities at their best,
because blood was drawn ~9 years before the scan:

| | instance-0 | instance-2 |
|---|---|---|
| proteomics r | 0.254 | 0.171 |
| fMRI encoder | mismatched | matched |
| joint cohort | 1,754 | 4,950 |

Currently on instance-2. Defensible, needs a methods paragraph.

**4. Which target demonstrates the method?** Proteomics Stage-1 **age** is R² 0.868
versus 0.107 for fluid IQ. Age is where two genuinely informative modalities exist,
so it is where cooperative fusion could actually be shown to work. The user has
said age is not the priority. Blocked on item 15 either way.

**5. Is the fusion architecture worth revising?** Phase 2 degrades immediately; the
gate saturates (T=0.1) or freezes (T=1.0). Headroom looks small — Phase 1 plateaus
at val R² 0.077 while fMRI alone gets 0.067 test.

**6. Report before or after item 6** (the last leakage channel)?

---

## Immediate next actions

**Multi-seed the Stage-2 result** — ~10 minutes, and the branch decomposition
currently rests on a single run:

```bash
for s in 1 2 3; do conda run --no-capture-output -n tcg_net python -u fusion/training/train_stage2.py --seed $s 2>&1 | tee ~/sweep/stage2_seed$s.log; done
```

Then `cat fusion/training/checkpoints/test_metrics_fluid_iq_*.json`.

**Item 6 — proteomics scaler refit.** `fusion/data/build_joint_dataset.py:266` is an
explicit no-op (`_ = scaler`). The scaler was fit on the proteomics Stage-1 train
fold, which overlapped joint val/test. Excluding subjects from Stage-1 *training*
did not fix this. Last remaining leakage channel; expected to be small.

**Item 12 — `l_sparse` has no gradient.** One line in
`fmri/Code/train_fmri_stage1.py:504`: `self.last_attn = (attn_ei.detach(), attn_w)`.
Shared file, so coordinate with the fMRI colleague.

---

## Project-specific traps

- **Graph-feature caches validate only subject count.** Any change to features or
  splits requires deleting `graph_feats_joint*.npz` by hand. A colleague's
  within-module z-score fix landed upstream and will **not** reach fusion otherwise.
- **The two `TARGET_CONFIG`s deliberately disagree** — `proteomics/data/build_dataset.py`
  is instance-0 (blood drawn at baseline), `fusion/data/build_joint_dataset.py` is
  instance-2 (scans at imaging visit). "Aligning" them silently breaks one stage.
  Both carry comments saying so.
- **`best_model.pt` is a fixed filename.** Never run two Stage-2 jobs in parallel —
  the test eval reloads that file and would score another run's model.
- **Stage 2 uses bare `print()`** (item 31), so `python -u` is required or output
  block-buffers and the run looks hung.
- **`train_stage2_age.py` is behind** `train_stage2.py` by every fix from 2026-07-30
  (item 32).
- **GitHub DNS is intermittent** on the China server. Retry `git pull`; it usually
  clears.
- **Do not run `preprocess.py --residualize`** until item 16 is fixed — the covariate
  matcher binds protein `AGER` as the age covariate.

## Mistakes made this session — do not repeat

- **Single-seed conclusions were wrong twice.** wd=1e-2 "beating" wd=1e-5, and a
  seed-42 exclusion "3.3 sd drop" with mechanisms attached. Both dissolved under 3
  more seeds. Seed noise is ±0.005–0.009 R² (Stage 1) and ±0.0015 val MAE (Stage 2).
  **Never compare configurations on one run.**
- **MAE → R² conversion assuming Gaussian residuals was inflated** (~0.17 vs a true
  0.077). Fluid IQ is integer-valued and the shrunk predictor is leptokurtic. Use
  reported R², never a conversion.
- **Findings were reported before all review angles returned**, and four of the
  agents had died on a rate limit. Two of the most severe fusion defects were missed
  in the first report.
- **"Proteomics contributes nothing" was too strong** — R² ≈ 0 with r ≈ 0.17 means
  weak, not absent.

## Environment

- Server: `~/TCG-Net`, conda env `tcg_net`, 2× RTX 4090, SSH (user `minesu`)
- Shared exchange dir: `/shared/tcgnet_data_2` (set via `SHARED_DATA_DIR` in
  `proteomics/config.py`, override with `TCGNET_SHARED_DATA`) — contents documented
  in [`SHARED_DATA_MANIFEST.md`](SHARED_DATA_MANIFEST.md)
- Raw UKB data: `~/uk_BIOBANK_DATA/` — read-only, confidential, aggregate stats only
- Long runs go in `tmux` (`tmux new -s tcg`, `Ctrl-b d` to detach)
- Stage-1 runs ~40 s; Stage-2 runs ~2 min; joint dataset build ~1 min
