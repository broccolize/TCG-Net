# TCG-Net — Fix Log

Dated record of code changes made while working through the review findings.
Newest entry first. Written to be self-contained: if you are picking this up with
no prior context, read [Current state](#current-state) then the latest entry.

Companion file: [`FIX_PLAN.md`](FIX_PLAN.md) — the full prioritised list of what
still needs doing.

---

## Current state

**Branch:** `fix/critical-early-stopping-and-phi-int` (off `main` @ `fd6030a`)

**What is trustworthy right now**

| Result | Status |
|---|---|
| Proteomics Stage 1 — **age** (test R² 0.868, MAE 2.35 yr) | ✅ verified clean, safe to report |
| Proteomics Stage 1 — **fluid IQ** (R² 0.047) | ❌ model never learned; peaked at epoch 1 |
| Proteomics **CV** summary | ⚠️ reports best-epoch, not final-epoch. Fluid IQ sign is wrong: reports **+0.059**, actual **−0.084** |
| All **Stage 2 / fusion** results | ❌ do not report — see `FIX_PLAN.md` |
| `fusion/analysis/` figures + CSVs | ❌ regenerate after re-running Stage 2 |

**Four root causes behind almost every finding**

1. The proteomics fluid-IQ encoder never trained (no early stopping) — and *both*
   Stage-2 runs load it, including the age run.
2. `phi_int` was mathematically always zero, killing the Shapley interaction value,
   the `l_shapley` term, and every interpretability output built on them.
3. Splits are drawn fresh at each stage instead of carried through, so Stage-1
   training subjects land in Stage-2 validation.
4. Nothing validates what it loads — checkpoint target, label instance, test-set
   evaluation, or seeding.

Root causes 1 and 2 are addressed below. **3 and 4 are still open.**

---

## 2026-07-30 — Fluid IQ weight-decay sweep: test R² 0.067 → 0.110

First results from the retrained Stage 1. Diagnostic runs on the **existing**
`dataset_fluid_iq.pt` (no leakage fix yet), so these are comparable to the old
numbers but not yet publishable.

Cohort: train **11,841** / val **2,507** / test **2,555** (16,903 total).
Runs take ~2 s/epoch on a 4090 — a full run is under a minute.

### Weight-decay sweep (seed 42)

| wd | best epoch | best val_loss | test R² | test MAE | test r |
|---|---|---|---|---|---|
| 1e-5 (old default) | 1 | 4.5422 | 0.0667 | 1.6812 | 0.2634 |
| 1e-3 | 1 | 4.5475 | 0.0667 | 1.6810 | 0.2631 |
| 1e-2 | 4 | 4.5397 | 0.0715 | 1.6760 | 0.3139 |
| 3e-2 | 14 | 4.4212 | 0.0828 | 1.6651 | 0.3081 |
| 1e-1 | 12 | 4.2897 | 0.1077 | 1.6407 | 0.3362 |
| **3e-1** | **24** | **4.2724** | **0.1104** | **1.6379** | 0.3346 |

### Seed stability at wd=1e-2

| seed | best epoch | test R² | test r |
|---|---|---|---|
| 42 | 4 | 0.0715 | 0.3139 |
| 1 | 3 | 0.0657 | 0.2744 |
| 2 | 3 | 0.0689 | 0.3022 |
| 3 | 3 | 0.0587 | 0.3115 |

Mean R² **0.0662 ± 0.0055**.

### Findings

1. **Early stopping works.** Every run stops well short of 30 epochs, and the
   epoch-1 collapse is gone above wd=1e-2.
2. **wd=1e-2 does not actually help.** Its four-seed mean R² (0.0662) is
   indistinguishable from wd=1e-5 (0.0667) — the single-seed 0.0715 that looked
   like an improvement was the luckiest of four draws. It changes the training
   *dynamics* (best epoch 1 → 3-4) without changing the outcome. The real
   threshold is **1e-1**, which sits ~7 sd above the wd=1e-2 seed distribution
   and is decisively real.
3. **The best run was truncated, not converged.** wd=3e-1 ran all 30 epochs with
   best epoch 24 and never triggered early stopping — the curve was still
   improving when the epoch cap hit. Higher weight decay needs more epochs.
4. **The calibration gap closed.** At wd=1e-2, r² = 0.0985 vs R² = 0.0715 (a 27%
   shortfall from mis-scaled predictions). At wd=3e-1, r² = 0.1120 vs R² = 0.1104
   — essentially none. Heavy regularisation fixed the over-dispersion as well as
   the overfitting.

Net: test R² **0.067 → 0.110** (+65%), MAE 1.681 → 1.638, r 0.263 → 0.335, against
the number in `TRAINING_SUMMARY.md`.

**Still a weak association** — that is the honest result for plasma proteomics
predicting fluid IQ, and it is now an honest weak result rather than an artefact
of a model that stopped learning at epoch 1.

### Next

Sweep higher with a raised epoch cap, since 3e-1 was cut off:
`--weight-decay {3e-1, 1e0, 3e0} --epochs 100`, then multi-seed the winner.

---

## 2026-07-30 — Revised the item-5 split strategy (no code change)

Documentation only. `FIX_PLAN.md` previously recommended **re-pretraining both
encoders on joint-train only**. That is the wrong trade and has been replaced.

Restricting proteomics Stage 1 to joint-train would cut it from tens of thousands
of subjects to 1,244 — discarding ~97% of the training data to solve a problem
caused by ~510 specific participants. The leakage is not "Stage 1 saw too many
subjects", it is "Stage 1 saw *these particular* subjects, who are now in the
Stage-2 evaluation set".

**New recommendation (option C):** exclude the joint val/test EIDs from Stage-1
`train_idx` and `val_idx`, keep everyone else. 510 subjects for fluid IQ, ~1,598
for age — roughly a 1% reduction. Full-size Stage-2 test set, no leakage.

Consequence for the run order: the joint dataset must be built **before** the
final Stage-1 runs, so the exclusion list exists. No circular dependency —
`build_joint_dataset.py` needs only `proteomics_tensor.pt` and `corr_matrix.mat`,
not any Stage-1 model.

Still to implement: `build_joint_dataset.py` writing the joint val/test EIDs, and
an `--exclude-eids` flag on `train_stage1.py`.

A **diagnostic** Stage-1 retrain on the existing fluid IQ dataset is unaffected by
any of this and is now step 0 in the plan — it answers whether fluid IQ has any
signal at all once early stopping prevents the epoch-1 collapse. If it does not,
the leakage work is moot for that target.

---

## 2026-07-30 — Redirect to `/shared/tcgnet_data_2`; target-specific proteomics checkpoint

Both branches are re-training, so a clean shared directory is being started.

**Path redirection.** `/shared/tcgnet_data` was hardcoded in six places across
three fusion files. Replaced with a single constant,
`proteomics/config.py::SHARED_DATA_DIR`, defaulting to `/shared/tcgnet_data_2`
and overridable via `TCGNET_SHARED_DATA` — so switching generations (or pointing
at the old one to reproduce an earlier result) is one environment variable rather
than six edits. Verified both the default and the override resolve correctly.

`fmri/Code/*` paths were **not** touched — they write to the colleague's own
`~/Project/TCG-Net/Data/` and are outside this branch's scope.

**FIX_PLAN item 3 done at the same time**, because it changes what files need to
be placed. `PROT_CKPT` is now target-specific
(`proteomics_stage1_age.pt` / `proteomics_stage1_fluid_iq.pt`), and both trainers
now assert `ckpt['config']['target']` matches the run, raising a named error
instead of silently warm-starting from the wrong encoder. They also print the
checkpoint's target, epoch and val_r2 on load, which would have made the epoch-1
problem obvious months ago.

**New file:** [`SHARED_DATA_MANIFEST.md`](SHARED_DATA_MANIFEST.md) — exactly what
goes in the new directory, what is deliberately excluded, and why.

> [!danger] Age Stage 2 is blocked
> `fmri_stage1_age_encoder.pt` is not in the colleague's Data folder — it contains
> only fluid-IQ encoders — and no committed script produces it. The old one on
> `/shared/tcgnet_data` was trained on pre-fix graph features, so copying it
> forward would pair a stale encoder with corrected features. Someone needs to
> write and commit an age variant of `train_fmri_stage1.py` (item 15).
> **Fluid IQ Stage 2 is not blocked.**

---

## 2026-07-30 — Rebased onto upstream fMRI feature fix

Fetched `origin/main`, which had moved from `fd6030a` to `28bb8b9` (two commits by
a colleague: `dc0bfa4` "Fix z-score issue to match plan", `28bb8b9` "fixed minor
issue"). **No file overlap** with this branch — upstream touched only
`fmri/Code/*` and `fmri/Reference material/TCG Net Plan.md`, this branch touches
`fusion/` and `proteomics/`. Rebased cleanly, no conflicts.

**What upstream changed:** the within-module degree z-score in
`compute_graph_features` now z-scores within-module strength (Guimerà & Amaral)
rather than total strength. This is a genuine correction — the old version was
"largely redundant with feature 1" per their commit note.

> [!warning] This fix will not reach fusion on its own
> The graph-feature cache validates only subject count, so re-running Stage 2
> reloads the OLD features. **Delete `graph_feats_joint.npz` and
> `graph_feats_joint_age.npz` before any Stage-2 run.** Both `fmri_stage1_*.pt`
> encoders were also trained on the old features and are now stale relative to
> the code. See the ⚠️ section in `FIX_PLAN.md`.

**Checked and unaffected:** `FLUID_IQ_COL = '20016-2.0'` still instance 2 (the
label-mismatch finding stands), `BAD_FC_EIDS` unchanged, and `last_attn` is still
detached — it just moved from line 500 to **line 504**.

---

## 2026-07-30 — Early stopping (Stage 1) + `phi_int` repair (fusion)

Addresses root causes **1** and **2**. Code only — **nothing has been re-trained
yet**, so every number on disk still reflects the pre-fix behaviour.

### Fix 1 — Early stopping in proteomics Stage 1

**Why.** Every fluid IQ CV fold peaked by epoch 3 and then overfit for the
remaining 27 epochs, so "best checkpoint" meant "the moment before it went
wrong". The checkpoint that got promoted to Stage 2 was epoch 1 with
`val_r2 = 0.047`.

| File | Change |
|---|---|
| `proteomics/config.py` | Added `EARLY_STOP_PATIENCE = 10`. `EPOCHS` is now an upper bound. |
| `proteomics/training/train_stage1.py` | Added `--patience` (0 disables) and `--weight-decay` CLI flags. |
| `proteomics/training/train_stage1.py` | Patience counter; `break` when exceeded; `stopped_early` / `epochs_run` / `checkpoint_evaluated` recorded in the test-metrics JSON. |
| `proteomics/training/train_stage1.py` | Optimiser, checkpoint config and experiment JSON now read `args.weight_decay` instead of the module constant, so the value is tunable and correctly recorded. |

`WEIGHT_DECAY = 1e-5` in config is **unchanged** — only made overridable. The
1e-5 default is very light for a 4.2M-parameter model on a weak target.

**Adjacent fix included:** `val_loss < best_val_loss` is `False` for NaN, so a run
that diverged never wrote `best.pt` and then died on `FileNotFoundError` at the
test-eval step, discarding the whole run. It now falls back to `last.pt` with a
warning, or raises a clear error if no epoch completed at all. Included because
raising weight decay increases divergence risk during the retrains below.

**Verified.** Simulated the new patience logic against the real fluid IQ fold-0
`val_loss` curve from `train_log_fluid_iq_20260508_075907_fd84367d_fold0.csv`:

```
patience=10 -> stops at epoch 12, selects epoch 2 (val_loss 4.307), saves 18 epochs
patience= 5 -> stops at epoch  7, selects epoch 2 (val_loss 4.307), saves 23 epochs
```

Same checkpoint selected, far less wasted compute. All four edited files pass
`python -m py_compile`.

### Fix 2 — `phi_int` is now a real interaction index

**Why.** `fusion/models/shapley_fusion.py:48` defined `phi_int` as the residual of
the Shapley efficiency identity. For two players `phi_f + phi_p == y_hat_fp - y0`
is a *theorem*, so the residual was identically zero for every input — pinning the
third softmax logit at 0 and making `w_int` a mechanical function of the other
two. `results/*/summary.json` records `"phi_int_std": 0.0` in both targets.

| File | Change |
|---|---|
| `fusion/models/shapley_fusion.py` | `phi_int = y_hat_fp - y_hat_f - y_hat_p + self.y_baseline` (the 2-way interaction index), with a WARNING comment against reverting to the residual form. |
| `fusion/training/losses.py` | `l_shapley = (y_hat - (y0 + phi_f + phi_p)).pow(2).mean()` — efficiency applied to the **fused** prediction. |
| `fusion/models/shapley_fusion.py` | Added `assert out.phi_int.abs().max() > 1e-6` to the `__main__` self-test as a regression guard. |

> [!warning] The two edits are a pair — do not apply one without the other.
> With the corrected `phi_int`, the **old** `l_shapley` formula reduces exactly to
> `mean(phi_int²)`, i.e. a penalty driving the interaction to zero — the opposite
> of the intent. Verified numerically: `3.012024` both ways.

**Verified** (pure arithmetic, 100k random draws, no torch needed):

```
OLD phi_int   max|.| = 4.441e-16   std = 1.226e-17   <- degenerate
NEW phi_int   max|.| = 7.919e+00   std = 1.736e+00   <- live

OLD l_shapley (tautology)                    = 2.367e-34
TRAP: old formula with NEW phi_int           = 3.012024  == mean(phi_int^2)
NEW l_shapley (y_hat vs y0+phi_f+phi_p)      = 2.016202  <- live constraint

efficiency axiom still holds: max|phi_f+phi_p-(y_hat_fp-y0)| = 4.441e-16
```

`phi_f` and `phi_p` are unchanged and still satisfy efficiency.

### ⚠️ Decision that needs your confirmation

`l_shapley` had to be redefined, not just repaired — the old form is a tautology
by construction. I chose **Shapley efficiency applied to the fused prediction**:
`y_hat` should reproduce `y0 + phi_f + phi_p`. This is non-trivial because `y_hat`
comes from `PredictionHead(z_fusion)` while the decomposition comes from the
sub-heads.

`phi_int` is deliberately excluded — it is the interaction *index*, a diagnostic,
not an additive component of the decomposition.

**Check this against §6.4 of the spec PDF, which I have not seen.** If you would
rather change nothing beyond making `phi_int` real, pass `lambda_shapley=0` to
`Stage2Loss` and the term becomes an inert diagnostic again.

Also note: `l_shapley` was contributing **zero gradient** before this change and
now contributes a real one, so Stage-2 training dynamics will differ from every
run currently on disk. That is intended, but it means old and new runs are not
comparable.

### Next actions

Re-train Stage 1 for both targets. **Long-running — run these yourself**, from
`~/TCG-Net` on the server:

```bash
conda run --no-capture-output -n tcg_net python proteomics/training/train_stage1.py --target age
```

```bash
conda run --no-capture-output -n tcg_net python proteomics/training/train_stage1.py --target fluid_iq --weight-decay 1e-3
```

For fluid IQ, `1e-3` is a starting point; a sweep over `1e-3 … 1e-2` is worth it
given how fast it overfits. Watch `stopped_early` and `best_epoch` in
`logs/experiments/test_metrics_*.json`.

Do **not** re-run Stage 2 yet — items 3–6 in `FIX_PLAN.md` (split strategy,
checkpoint provenance, label instance) should be settled first, or the run is
wasted.

### Explicitly NOT done in this session

- `proteomics/training/cv_stage1.py` — still has **no early stopping** (it retypes
  the training loop rather than importing it) and still aggregates best-epoch
  metrics. The fluid IQ CV sign error is unfixed.
- Nothing in `fusion/data/build_joint_dataset.py` or the Stage-2 trainers.
- `fusion/analysis/*.py` — still uncommitted on the server, still unreviewed.
- No re-training, no data rebuild, no figure regeneration.
