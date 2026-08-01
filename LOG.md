# TCG-Net — Fix Log

Dated record of code changes made while working through the review findings.
Newest entry first. Written to be self-contained: if you are picking this up with
no prior context, read [`PROMPT.md`](PROMPT.md) first, then the latest entry here.

Companion files: [`PROMPT.md`](PROMPT.md) — session handoff and the open
decisions. [`FIX_PLAN.md`](FIX_PLAN.md) — the full prioritised list of what still
needs doing.

---

## 2026-08-01 (evening) — The method beats its baselines on fluid IQ

Eight independent seeds, converged (`--phase2-epoch 999`, best epoch 42–61), plus
the classical baselines on the same 747 test subjects. **This is the first version
of the Stage-2 result that is properly trained, properly seeded and properly
baselined, and it is positive.**

### Against linear baselines

| model | test R² |
|---|---|
| mean predictor | −0.0007 |
| ridge on 2,920 proteins (M1) | +0.0242 |
| ridge on 2,145 FC edges (M2) | +0.0409 |
| ridge on both, concatenated (M3 early fusion) | +0.0317 |
| **Stage-2 fMRI branch (GNN)** | **+0.0655** — 1.60× ridge_fmri |
| **Stage-2 fused, 8 seeds** | **+0.0732** — 1.79× the best ridge |
| **Stage-2 fused, val-calibrated** | **+0.0781** — 1.91× |

Two things worth stating separately. The GNN connectome encoder beats ridge on
the raw edges by 60%, so the graph modelling is doing real work. And
**`ridge_concat` (0.0317) is *worse* than `ridge_fmri` alone (0.0409)** — naive
early fusion on raw features actively destroys signal, which is the strongest
possible motivation for a learned fusion module.

This is the opposite of the age result, where ridge matched the encoder exactly
(0.8690 vs 0.868). The two targets have opposite verdicts on whether depth helps,
and both belong in the report.

### Branch contrasts, 8 seeds

| contrast | mean | sd | signs | t (df=7) |
|---|---|---|---|---|
| fused − fMRI-only | **+0.0078** | 0.0066 | **7/8** | +3.31 |
| joint-head − fused | **−0.0045** | 0.0034 | **1/8** | −3.69 |
| joint-head − fMRI-only | +0.0033 | 0.0055 | 5/8 | +1.69 |

**The Shapley-gated output beats its own naive-concatenation head on 7 of 8
seeds.** That contrast is the cleanest in the table, and it is the reverse of what
was claimed this morning from four truncated runs. Ordering is
`fused > joint-head ≈ fMRI-only`, consistent with the M10 > M6 > M2 pattern §10
predicts (M6 proper — a learned non-Shapley gate — still has to be built).

Val-calibrated `fused − fMRI-only` is **+0.0118, 5/5 seeds** (calibration fields
exist for seeds 4–8 only).

### One anomaly, checked and cleared

Seeds 1–3 average fused R² 0.0667; seeds 4–8 average 0.0771. The fMRI branch does
not move (0.0654 vs 0.0655), so it is specific to the fused head. Being the three
lowest of eight has probability 1/56 ≈ 0.018 — suspicious enough to check.

**It is not a code difference.** Seeds 1–3 ran before commit `cd5cb47` and were
re-run after it, and the two batches produced *byte-identical* metrics. That
duplication proves the calibration commit did not touch training, so the spread is
seed variation. The re-runs are also a clean determinism check: training is fully
reproducible under a fixed seed.

### A bug in our own summariser, twice

First it counted deterministic re-runs as independent, inflating df. Fixed. The
fix then kept the *first* copy of each duplicate — which for seeds 1–3 is the
pre-`cd5cb47` record with no calibration fields — so the calibration block silently
reported 5 runs instead of 8. Now keeps whichever copy carries more fields.

---

## 2026-08-01 (late) — Spec audit: three terms in the design are mathematically inert

First read of `fmri/Reference material/Model Design of TCG.pdf`. No assistant had
seen it before; `PROMPT.md` open decision 2 was blocked on exactly this. Four
findings, all algebraic and checkable by hand.

### 1. §4.2 — `phi_int` is identically zero **in the spec**

The spec defines

```
phi_f   = ½[(ŷ_f − ŷ_0) + (ŷ_fp − ŷ_p)]
phi_p   = ½[(ŷ_p − ŷ_0) + (ŷ_fp − ŷ_f)]
phi_int = ŷ_fp − ŷ_0 − phi_f − phi_p
```

Adding the first two, the cross terms cancel: `phi_f + phi_p = ŷ_fp − ŷ_0`
exactly. This is the Shapley efficiency axiom for two players — a theorem, not a
measurement — so `phi_int ≡ 0` for every participant by construction.

**This was never a coding error. It is a defect in the design document**, and it
disables the third softmax logit, which is why `w_int` sat frozen. The code
already deviates (`shapley_fusion.py:58`) using the standard 2-way Shapley
interaction index `I_fp = ŷ_fp − ŷ_f − ŷ_p + ŷ_0`, which is the correct object.
That deviation is now confirmed as *necessary*, not optional.

### 2. §6.3 half-notices, and the literal formula is a trap

§6.3 says: *"If φ_int is defined exactly as above, this loss will be close to
zero by construction."* The author saw that `L_shapley ≈ 0` but not that `phi_int`
itself is the degenerate quantity.

Worse, applying §6.3 literally **with a corrected `phi_int`** gives
`||ŷ_fp − (ŷ_0 + phi_f + phi_p + I_fp)||² = ||I_fp||² = phi_int²` — a penalty
driving the interaction to zero. The current code
(`losses.py:28`) instead scores the *gated* output against the additive
decomposition, `(y_hat − (y0 + phi_f + phi_p))²`, which is well-posed and carries
real gradient. **Open decision 2 is resolved: the deviation from §6.3 is correct
and must be kept.** Do not "restore spec compliance" here.

### 3. §6.4 — `||w||₁` is a constant

`w` is a softmax output, so its entries are non-negative and sum to 1, giving
`||w||₁ ≡ 1` and zero gradient. Session 3 dropped it empirically; it is provably
inert. The code's `l_sparse = ||α_f||₁` is the only live term of the three, which
makes item 12 (the detached attention) the difference between `l_sparse` existing
and not existing at all.

### 4. §12 — the subtyping needs no Shapley values

§12 defines the subtype score as `D_i = phi_f − phi_p`. Substituting §4.2:

```
phi_f − phi_p = ½[2ŷ_f − 2ŷ_p] = ŷ_f − ŷ_p
```

**The individualised "mechanistic subtyping" is exactly the difference between
the two single-modality predictions.** The Shapley machinery cancels out. This
does not make the analysis wrong, but the interpretability claim in §13 cannot
rest on it — the same subtypes fall out of `ŷ_f − ŷ_p` with no fusion module at
all. It also explains FIX_PLAN item 19: thresholding a z-scored variable at ±0.5
yields ≈30/40/30 by construction, so near-thirds is expected, not a binning bug.

### 5. §7.2 is the source of the truncation

The spec prescribes "first 10 epochs freeze, after 10 unfreeze", which is where
`--phase2-epoch 11` came from. Converged runs put the best epoch at **52–57**, so
the spec's protocol stops phase 1 at roughly a fifth of its useful length. The
deviation is justified and needs a methods sentence.

### What this means for the claims

§13 claims the method "simultaneously improve[s] cognitive prediction and
provide[s] individualized mechanistic interpretation". As specified, three of its
components are inert (`phi_int`, `L_shapley`, `||w||₁`) and the subtyping reduces
to a two-model difference. §10 already anticipates the outcome: *"even if the full
model shows only modest predictive improvement over standard gated fusion, its
individualized interpretability can still provide substantial scientific value."*

§10 also specifies the ablation ladder (M1–M10) with the expected ordering
`M10 > M6 > M5/M4 > M2 > M1`. **M6 — a learned modality gate without Shapley — is
the decisive baseline and has never been implemented.** M1/M2/M3 are covered by
`fusion/analysis/baseline_joint.py`.

---

## 2026-08-01 (late) — Calibration recovers the proteomics branch; ridge matches the age encoder

### Age: the Stage-1 encoder does not beat ridge

Ridge on the 2,920 z-scored proteins, alpha chosen on val, same split, test
touched once (n=7,904):

| model | R² | MAE | r |
|---|---|---|---|
| mean predictor | −0.0000 | 6.989 | — |
| **ridge (all proteins)** | **0.8690** | 2.348 | 0.9322 |
| ridge (top-200 by \|r\|) | 0.7677 | 3.136 | 0.8762 |
| **Stage-1 encoder** | **0.868** | | |

**0.8690 vs 0.868.** The SE-gated encoder and a closed-form linear model are
indistinguishable. Age is linearly decodable from an Olink panel — a known
property of plasma proteomic clocks — and R² 0.868 is a statement about the
data, not the architecture. It cannot be reported as an architectural result.

The top-200 row (0.7677) shows the signal is spread across many proteins rather
than carried by a handful, which is worth a sentence of its own.

### Fluid IQ: val-fitted calibration, 3 converged seeds

| branch | raw R² | calibrated R² | Δ | fitted scale |
|---|---|---|---|---|
| fused | +0.0666 | **+0.0768** | +0.0102 | 0.867 |
| fMRI-only | +0.0655 | +0.0669 | +0.0014 | 1.161 |
| prot-only | +0.0007 | **+0.0294** | +0.0287 | 132.0 |
| joint-head | +0.0647 | +0.0721 | +0.0074 | 0.884 |

**The proteomics branch was amplitude-collapsed, and this proves it.** Calibrated
prot-only scores **0.0294** against an r² ceiling of 0.0291–0.0299 — it recovers
its ceiling to three decimals. Proteomics alone predicts instance-2 fluid IQ at
R² ≈ 0.029 held-out. Every previous statement that the proteomics branch
"contributes nothing" was reading an amplitude artefact, and the required scale
factor of ~132 shows how extreme the collapse was.

**The miscalibration runs in opposite directions, which was not what we
predicted.** The fused head's fitted scale is **below** 1: its output spread
(0.327–0.345) exceeds what its accuracy justifies (r·σ_t = 0.276–0.286), so it is
over-*dispersed*, not over-shrunk. The fMRI branch is the opposite (scale 1.161).
The r² − R² gap measures |miscalibration| and does not distinguish the two.

Calibrated `fused − fMRI-only`: **+0.0099**, 3/3 seeds positive
(+0.0103, +0.0163, +0.0032), t = +2.62 (df=2), p ≈ 0.12. Sign-consistent but the
spread is 5× across three seeds. **Not yet reportable — needs more seeds.**

### A statistics bug of our own making

`summarise_stage2.py` counted deterministic re-runs of the same seed as
independent, turning 3 runs into 6 and reporting `t=+2.17 (df=5)` for a contrast
whose real figures are 2/3 signs and df=2. Now deduplicated on
`(seed, best_epoch, test_r2, test_r2_fmri_only)`. The byte-identical re-run is a
useful determinism check and is still listed — it is just not extra evidence.

### Next

`fusion/analysis/baseline_joint.py` (new) asks of the joint model what ridge just
asked of the age encoder: ridge on 2,145 FC edges, on 2,920 proteins, and on the
5,065-feature concatenation, over the same joint split. `ridge_concat` is naive
early fusion with no gating and no learned representation. If it matches the
Stage-2 fused row, the architecture is not what produces the result.

---

## 2026-08-01 — Stage-2 multi-seed: the fusion null was a single-seed artefact

> [!caution] SUPERSEDED the same day — every run below was undertrained
> `--phase2-epoch` defaults to 11, so phase 1 was cut off at epoch 10. Re-running
> with phase 2 disabled, the best epoch lands at **52–57**. All four runs in this
> entry stopped roughly two-thirds of the way through training, and the branch
> contrasts below were measuring **which branch converges fastest**, not which
> performs better. Both headline contrasts reverse or vanish at convergence. Keep
> this entry for the record; do not quote its numbers. See the entry above it.

Four runs carrying the branch decomposition (seeds 42, 1, 2, 3; wd=1e-3; same
joint split, same encoders, stale-but-consistent graph features). **Two of the
three conclusions drawn from the single seed-42 run are wrong.**

| branch | mean R² | sd | min | max |
|---|---|---|---|---|
| fused | +0.0677 | 0.0009 | +0.0665 | +0.0683 |
| fMRI-only | +0.0623 | 0.0038 | +0.0578 | +0.0669 |
| prot-only | −0.0014 | 0.0019 | −0.0030 | +0.0011 |
| joint-head | **+0.0710** | 0.0022 | +0.0689 | +0.0741 |

Paired per-seed contrasts (the right statistic here — same test subjects, only
the init and batch order differ):

| contrast | mean | sd | signs | t (df=3) |
|---|---|---|---|---|
| fused − fMRI-only | **+0.0054** | 0.0030 | 4/4 + | +3.57 |
| joint-head − fused | **+0.0034** | 0.0026 | 4/4 + | +2.60 |
| joint-head − fMRI-only | **+0.0088** | 0.0052 | 4/4 + | +3.37 |

### What changed

1. **Multimodal fusion does beat fMRI alone.** +0.0054 R², same sign on every
   seed. The previously reported +0.0014 was seed 42 — the *smallest* of the four
   — read as if it were the estimate. "The joint model is the fMRI model" was
   wrong.
2. **The Shapley gate is the part that does not work.** `joint-head` — the plain
   MLP on concatenated embeddings, sitting inside the model as an incidental
   naive-fusion baseline — beats the gated output on 4/4 seeds. It also beats
   fMRI-only by +0.0088, against the gate's +0.0054. **The cooperative gating
   discards roughly 40% of the available multimodal gain.** The earlier reading
   ("joint-head matches the gated output") was again the single most favourable
   seed.
3. What survives unchanged: proteomics alone scores R² ≈ 0.

### The proteomics branch is amplitude-collapsed, not signal-free

`test_pearson_prot_only` across seeds 1/2/3: **+0.1705, −0.1695, −0.1700.**
Magnitudes agree to within 0.001; the sign flips. That is not a weak signal
behaving randomly — a genuinely noisy branch would scatter in magnitude too.

Two things follow. First, |r| ≈ 0.170 is essentially the **entire** proteomics
signal available in this cohort (LOG 2026-07-30 measured r = 0.171 for
proteomics → instance-2 fluid IQ on the joint cohort). The branch is extracting
all of it. Second, the amplitude is gone: from R² = −0.003 with |r| = 0.17, the
sub-head's output std is ≈ **0.9% of the target's**. It is predicting a constant
with a whisper of correctly-ordered variation on top, and at that amplitude the
MSE gradient cannot identify the sign — hence the flip.

So `prot-only R² ≈ 0` is **not** evidence that proteomics contributes nothing. It
is evidence that the proteomics sub-head is not being read out. This is item 10
manifesting in the branch rather than the main head. Supporting detail: seed 1 is
the one seed with the sign correct, and it has both the only positive prot-only
R² (+0.0011) and the best joint-head of the four (0.0741).

### Caveat on the error bars

All four runs share one joint split, so these sds are **initialisation noise, not
sampling error**. The test set is n=747; the sampling uncertainty on any single
R² is far wider (order ±0.03). Quote the paired contrasts, which are within-test-
set and legitimate; do not quote 0.0677 ± 0.0009 as if the population value were
pinned to three decimals.

n=4 with df=3 makes the t values indicative, not conclusive — p ≈ 0.03–0.08. The
4/4 sign consistency on all three contrasts is what carries the argument.

---

## 2026-08-01 — Item 6 measured: the fit-set leakage is negligible, re-run deferred

`check_scaler_leakage.py` run on the server against
`/shared/tcgnet_data_2/proteomics_tensor.pt`. This is the measurement the previous
entry said to take before paying for a second KNN imputation.

**Exposure.** 1,598 joint val/test EIDs, all present in the proteomics cohort;
**1,134 of them sit in the 36,888-row training split — 3.07% of the fit set.**

**Effect.**

| | median | p99.9 | max |
|---|---|---|---|
| per-feature mean shift | 0.0021 SD | 0.0099 SD | 0.0122 SD |
| per-feature std change | 0.16% | 0.84% | 0.90% |
| encoder input change `\|dz\|`, 4.67M cells | 0.0023 SD | 0.019 SD | 0.245 SD |

1.29% of cells move by more than 0.01 SD.

**Verdict: negligible — do not trigger a preprocessing run for this alone.** A
0.002 SD median shift in the encoder's inputs cannot move a Stage-2 number that
carries ±0.005–0.009 R² of seed noise. The `--exclude-eids` code stays, and the
next preprocessing run (whenever one is needed for another reason) closes the
channel properly. `scaler_fit_leakage_free` will read `false` in the joint bundles
until then, which is correct and now quantified.

> [!warning] The script's first verdict line was wrong and said the opposite
> It thresholded on `max(|dz|)`, which printed "material (2.45e-01 SD)". That
> statistic does not measure leakage. Since
> `dz = -m/s + z·(1/s - 1)`, the tail of `|dz|` is set by cells with extreme
> protein values — |z| ≈ 25 against a 0.9% std change reproduces 0.245 exactly —
> not by how much the statistics moved. The script now judges on the systematic
> offset `|m/s|` and the p99.9, and prints the max only as context. Re-run it if
> you want the corrected verdict line; the numbers above are unaffected.

Still unmeasured: the KNNImputer donor pool. Under 5% of cells are imputed, and
this bound does not cover them.

---

## 2026-07-31 — Item 6 (preprocessing fit-set leakage) and item 12 (attention gradient)

Code only. **Nothing in this entry has been run**; both changes need a run before
they mean anything, and neither may be merged until then.

### Item 6 — the last leakage channel, fixed upstream rather than downstream

`fusion/data/build_joint_dataset.py:266` used to read `_ = scaler`: the scaler was
loaded, logged, and deliberately not used. The plan entry said to refit it on the
joint train fold. **That would have been the wrong fix.** The frozen Stage-1
proteomics encoder was trained in the coordinate system of `scaler_1234`, and the
joint cohort is the imaging subcohort, whose per-protein means and stds are not
the full proteomics cohort's. Refitting at the joint stage would have swapped a
small label-free leakage for a genuine train/inference mismatch on the encoder's
inputs.

The leak is in the fit set, so it is fixed there:

- `proteomics/data/preprocess.py` gains `--exclude-eids`, mirroring the flag
  `train_stage1.py` already has. Listed participants are dropped from the fit set
  of the KNNImputer, the residualizer and the StandardScaler (and from the
  training median/mode used to fill age/sex). They are still imputed, still
  scaled, still in the output tensor, still in their original split — only the
  fitted statistics stop seeing them.
- The tensor now carries `fit_idx`, `fit_excluded_eids` and
  `fit_exclude_eids_file`; `scaler_<seed>.npz` carries the fit-row counts.
- `build_joint_dataset.py` checks it instead of asserting it. It intersects the
  joint val/test EIDs with the proteomics train split, subtracts what the tensor
  says was held out, and logs `Preprocessing provenance OK` or
  `PREPROCESSING LEAKAGE: n of m ...`. The verdict is written into both joint
  dataset bundles as `scaler_fit_leakage_free` and into the build JSON.

This needs two passes to converge (build → preprocess → build); it terminates
after one loop because the joint split depends only on `SPLIT_SEED` and the EID
intersection, which the preprocessing values do not touch. The order is written
out in `FIX_PLAN.md`.

**Measure before paying for it.** Pass 2 is a full KNN imputation over ~37k × 2920.
`fusion/data/check_scaler_leakage.py` (new) answers whether that is worth doing,
in seconds and without touching the raw CSV: `proteomics_tensor.pt` is already
z-scored, so the scaler that *would* have been fit on the clean subset is
recoverable as that subset's mean/std **in the current z coordinates**. It reports
the per-feature shift and the resulting change in encoder input, in SD units, for
the participants Stage 2 evaluates on. It does not cover the KNNImputer's donor
pool — that cannot be measured without re-imputing — but under 5% of cells are
imputed, so the scaler channel is the larger one by cell count.

### Item 12 — `l_sparse` had no gradient

`fmri/Code/train_fmri_stage1.py:504` cached `attn_w.detach()`. Stage 2 reads
`fmri_enc.last_attn[1]` and puts it straight into `l_sparse`
(`fusion/training/losses.py:37`), so the attention-sparsity term has been a
constant. Now only `edge_index` is detached.

Scope of the change: nothing in the fMRI Stage-1 script differentiates
`last_attn`, so locally the only cost is that one batch's graph stays alive until
the next forward overwrites the cache. In Stage 2 it is inert during phase 1 as
well — the fMRI encoder is frozen there — so this only starts acting in phase 2.
`train_fmri_stage1_alt1.py` and `_alt2.py` still detach; left alone deliberately,
they are the colleague's alternates and feed nothing in `fusion/`.

**This is a shared file.** It needs the fMRI colleague's sign-off, and an fMRI
Stage-1 re-run, before merge.

### Verified

`resolve_fit_positions` was exercised on synthetic inputs (no exclusion; overlapping
and non-overlapping EIDs; duplicates; a non-contiguous `train_idx`; a missing file;
and an exclusion that would empty the fit set — the last two raise rather than
silently proceed). All three edited files byte-compile. That is the limit of what
is checkable off the server: `sklearn` and `torch` are not installed locally, and
none of the data is here.

---

## Current state — updated 2026-07-30, end of session

**Branch:** `fix/critical-early-stopping-and-phi-int` — 12 commits, pushed,
**not merged to `main`**.

**What is trustworthy right now**

| Result | Status |
|---|---|
| Proteomics Stage 1 — **age** (test R² 0.868) | ✅ verified clean, safe to report. Not retrained; still instance-0, which is correct for baseline bloods |
| Proteomics Stage 1 — **fluid IQ** (test R² **0.107 ± 0.009**) | ✅ retrained, early-stopped, wd=1e-1, leakage-free |
| Stage 2 — **fluid IQ** (held-out test R² **0.068**, n=747) | ✅ honest number, but see the verdict below |
| Proteomics **CV** summary | ❌ still reports best-epoch. Fluid IQ sign is wrong: **+0.059** reported vs **−0.084** actual (item 8) |
| Stage 2 — **age** | ❌ blocked: no committed script produces an age fMRI encoder (item 15) |
| `fusion/analysis/` figures + CSVs | ❌ still uncommitted on the server, and generated from the dead `phi_int` (items 18/19) |

**The scientific verdict.** The pipeline is now correct; the result is that the
central claim is not supported for fluid IQ:

```
fused 0.0683 | fMRI-only 0.0669 | prot-only -0.0007 | joint-head 0.0689
```

The fusion adds 0.0014 over fMRI alone, and `joint-head` — a plain MLP on
concatenated embeddings, i.e. a naive-fusion baseline already inside the model —
matches the gated output. Proteomics carries r ≈ 0.17 (weak, not absent). See
`PROMPT.md` for the decisions this forces.

**Four root causes behind almost every finding**

1. The proteomics fluid-IQ encoder never trained (no early stopping) — and *both*
   Stage-2 runs loaded it. **Fixed.**
2. `phi_int` was mathematically always zero, killing the Shapley interaction value,
   the `l_shapley` term, and every interpretability output built on them.
   **Fixed and verified live.**
3. Splits drawn fresh at each stage, so Stage-1 training subjects landed in
   Stage-2 validation. **Fixed on the proteomics side** (`--exclude-eids`); the
   fMRI side and the scaler artefact (item 6) are still open.
4. Nothing validated what it loaded. **Fixed** — checkpoint-target assertion,
   seeding, held-out test evaluation, instance-2 labels.

---

## 2026-07-30 — Held-out Stage 2 result, and why the proteomics branch is flat

First genuinely held-out Stage-2 numbers (n=747), plus a decomposition of where
the proteomics signal goes.

### Test result

```
[test] n=747 epoch=10  mae 0.7851 (1.621 IQ) | r2 0.0683 | r 0.2661
[test] branch r2 — fused 0.0683 | fMRI-only 0.0669 | prot-only -0.0007 | joint-head 0.0689
[test] pred_std 0.2194 vs target_std 1.0122
```

**Essentially all of the joint model's performance is the fMRI branch.** Fused
(0.0683) is indistinguishable from fMRI-only (0.0669); the proteomics branch
scores the same as predicting the mean.

> [!important] `w_p` is not a measure of predictive contribution
> The gate reports `w_p ≈ 0.55` while the proteomics branch scores R² ≈ 0. The
> gate is a softmax over Shapley values derived from the sub-head outputs and can
> assign a high weight to a branch that predicts nothing. `Session4_Age.md` read
> this quantity as evidence of proteomic importance. It is not.

Shrinkage is near-optimal rather than pathological: r² = 0.0708 against R² =
0.0683, so the model is calibrated — it simply has little to say.

### Where the proteomics signal goes

Stage-1 encoder scored on the joint test subjects against both label versions:

| Proteomics → fluid IQ | Pearson r |
|---|---|
| Full proteomics cohort, instance-0 (Stage-1 test) | **0.345** |
| Joint/imaging cohort, instance-0 | **0.254** — cohort restriction costs 0.09 |
| Joint/imaging cohort, instance-2 | **0.171** — 9-year gap costs a further 0.08 |

The second step is fully explained by measurement attenuation:
0.254 × 0.636 (instance-0/instance-2 test-retest on these subjects, n=229) =
**0.162**, against 0.171 observed. Nothing unexplained in the target switch.

So proteomics **does** carry signal for instance-2 fluid IQ — r ≈ 0.17 — it is
just weak, and an optimally-scaled r=0.17 predictor caps out at R² = 0.029. The
earlier reading of "contributes nothing" was too strong: it contributes little,
and the fusion extracts even less.

### The tension this exposes

| | instance-0 | instance-2 |
|---|---|---|
| proteomics r | 0.254 | 0.171 |
| fMRI encoder | trained on instance-2 (mismatched) | matched |
| joint cohort | 1,754 | 4,950 |

There is no configuration where both modalities are at their best, because the
blood was drawn ~9 years before the scan. That is a property of UK Biobank, not
of the code, and belongs in the methods.

### Weight decay is inert at the current schedule

The wd sweep (1e-3 → 3e-1, a 300× range) moved val MAE by 0.0001 and every run
stopped at `best_epoch = 10`. AdamW's decoupled decay shrinks weights by
`(1 − lr·wd)` per step; at lr=1e-4 and wd=3e-1 over 10 epochs × 108 batches that
is a 3% total shrinkage, and at wd=1e-3 it is 0.0001%. The flag works — there is
simply no room for it to act. Re-sweep only if the epoch schedule changes.

### Phase 1 is not truncated; Phase 2 degrades

val_r2 plateaus at 0.074–0.078 by **epoch 2–3** and oscillates for the next seven.
`best_epoch = 10` is the low point of a flat noisy curve, not the end of a rising
one. Phase 2 makes it worse immediately: 0.0767 → 0.0710 → 0.0668.

### Correction recorded

Earlier entries converted MAE to R² assuming Gaussian residuals (e.g. "val
R² ≈ 0.17"). That is wrong here — fluid IQ is integer-valued and the shrunk
predictor gives leptokurtic residuals, so MAE runs lower than a Gaussian of the
same variance implies. True val R² is 0.077. **Use the reported R², never a
conversion from MAE.**

---

## 2026-07-30 — Stage 2: test evaluation, real freeze, seeding, argparse

Five fixes to `fusion/training/train_stage2.py`, all needed before any Stage-2
number is reportable.

| Item | Fix |
|---|---|
| 9 | **Held-out test evaluation.** `test_loader` was built and never used, so every Stage-2 number was a validation minimum that had also selected the checkpoint and driven early stopping. Now reloads `best_model.pt` after training and scores the 747 test subjects, writing `test_metrics_fluid_iq_<run_id>.json`. |
| 13 | **Phase 1 actually freezes now.** `freeze()` only clears `requires_grad`; BatchNorm running stats are buffers and update anyway, and dropout stays active. The encoders now go to `.eval()` while `phase == 1`. |
| 14 | **Seeding.** `set_global_seeds(args.seed)` plus a seeded DataLoader generator. Nothing in Stage 2 was seeded before, so no run was reproducible. |
| 20 | **`drop_last=True`** on the train loader — `ProteomicsEncoder` normalises over the batch dimension and a trailing batch of 1 raises inside `BatchNorm1d`. |
| 30 | **`run_id`, `seed` and `weight_decay` in every JSONL row**, so appended runs are separable. Previously two runs shared the file with overlapping epoch numbers. |

Also added: **argparse** (`--seed`, `--weight-decay`, `--lr-head`, `--lr-encoder`,
`--epochs`, `--patience`, `--phase2-epoch`, `--no-test-eval`) so Stage 2 can be
swept the way Stage 1 was, and a shared `evaluate()` used by both the validation
loop and the test pass.

`evaluate()` also fixes two things in passing: metrics are now computed on pooled
predictions rather than as an average of per-batch means, and it reports R² and
Pearson r alongside MAE — with a zero-variance guard, since Stage 1 produced
`pearson_r = nan` at wd=3e0 when the model collapsed to a constant.

> [!warning] `train_stage2_age.py` is now behind
> None of the above was applied to the age trainer, which is a near-duplicate. It
> is blocked anyway (no age fMRI encoder — item 15), but the divergence is real
> and is exactly the duplication hazard flagged in the review. Tracked as item 32.

### Expect the numbers to move

The Phase-1 freeze fix changes what Phase 1 computes: the encoders now use their
pretrained BatchNorm statistics instead of joint-cohort ones, and their dropout is
off. Phase-1 validation was previously measuring a moving target, so the new curve
is not comparable to the run logged above.

---

## 2026-07-30 — First clean Stage 2: `phi_int` fix confirmed live in a real run

First Stage-2 run on the corrected pipeline — instance-2 labels, 4,950 joint
participants, leakage-free proteomics encoder (seed 3, selected on val loss),
fixed `phi_int` and `l_shapley`.

**Best val MAE 0.7289 z = 1.505 IQ points** (epoch 14), early-stopped at 34.

### The `phi_int` fix is confirmed in production

`fusion/training/checkpoints/training_log.jsonl` holds both the old and new runs:

| | `l_shapley` | `w_int` range |
|---|---|---|
| before (30 epochs) | `0.000000` every epoch | 0.2693 – 0.3338 (span 0.065) |
| **after (34 epochs)** | **0.0206 → 0.0643** | **0.1583 – 0.3892 (span 0.231)** |

`w_int` falls monotonically 0.39 → 0.16 through Phase 2 while `w_p` climbs
0.24 → 0.55. The gate now responds to data rather than echoing the other two
logits.

**Scientific consequence:** the model reports proteomics 0.55 / fMRI 0.29 /
interaction 0.16, with the interaction weight *declining* through training. That
is evidence of **limited** cross-modal synergy — the opposite of
`Session4_Age.md:43` ("the interaction term is the second-largest contributor"),
which was read off a structurally constant value.

`l_sparse` remains flat at ~0.0714 as expected — item 12 (detached GAT attention)
is still open.

### Two problems visible in the curve

1. **Phase 2 overfits hard.** From the epoch-11 unfreeze, train MAE falls
   0.79 → 0.53 while val climbs 0.73 → 0.87. Best is epoch 14, three epochs in;
   everything after is damage. Same disease Stage 1 had, same likely fix — Stage 2
   has no weight-decay tuning at all. Worth the same sweep treatment.
2. **Phase 1 barely learns.** val_mae 0.7350 → 0.7328 over ten frozen epochs. Part
   weak signal, but also consistent with item 13: the encoders are not actually
   frozen, so Phase 1 is not doing what it claims.

For scale: predicting the mean gives MAE ≈ 0.798 z, so 0.7289 is ~8.6% better
than guessing.

### New minor issue

`training_log.jsonl` has **no run_id** — both runs are appended with overlapping
epoch numbers and can only be separated by position in the file. Added as item 30.

### Next

- Stage-2 weight-decay sweep (mirrors what fixed Stage 1)
- Item 9: evaluate the 747 held-out test subjects; every number above is val
- Item 6: proteomics scaler refit, still the one open leakage channel
- Item 13: real freeze in Phase 1

---

## 2026-07-30 — Leakage-free Stage 1 done: test R² 0.107 ± 0.009, exclusion costs ~nothing

First Stage-1 checkpoints trained with `--exclude-eids`. **These are the ones that
go to Stage 2.**

### Exclusion applied

```
Exclusion list: 1598 EIDs; 510 present in this dataset.
  train_idx: 11841 -> 11490 (removed 351, 2.96%)
  val_idx:    2507 ->  2431 (removed  76, 3.03%)
```

Only 510 of the 1,598 joint val/test EIDs appear in the proteomics fluid IQ
dataset, because that dataset is instance-0 (32.9% coverage of the joint cohort).
The remainder are harmless no-ops. The joint-level list is still a superset of the
per-target val/test set, so the exclusion is complete.

### Result (wd=1e-1, epochs=100, with exclusion)

| seed | best epoch | test R² | test MAE | test r |
|---|---|---|---|---|
| 42 | 15 | 0.0959 | 1.6588 | 0.3143 |
| 1 | — | 0.1113 | 1.6438 | 0.3377 |
| 2 | — | 0.1035 | 1.6468 | 0.3324 |
| 3 | 10 | 0.1156 | 1.6414 | 0.3451 |

**Mean test R² = 0.1066 ± 0.0087, r = 0.3324.**

### The exclusion is essentially free

| | test R² |
|---|---|
| no exclusion (4 seeds) | 0.1132 ± 0.0052 |
| **with exclusion (4 seeds)** | **0.1066 ± 0.0087** |

Difference 0.0066, SE ≈ 0.005, t ≈ 1.3 on ~6 df — not significant. Removing the
Stage-2 evaluation subjects from Stage-1 training costs nothing measurable, which
settles item 5 empirically as well as in principle.

> [!warning] Methodological note for anyone reading this later
> Seed 42 alone gave 0.0959 and was initially read as a real ~3.3 sd drop caused
> by the exclusion, with plausible mechanisms attached (smaller training set;
> imaging cohort being non-random). Three more seeds dissolved it entirely. This
> is the **second** time in this session a single-seed result produced a confident
> wrong conclusion — the first was wd=1e-2 appearing to beat wd=1e-5. At this
> effect size (differences of ~0.01 R² against a seed sd of ~0.005-0.009),
> **no single run should be used to compare configurations.**

### Cumulative effect on fluid IQ Stage 1

| | original | now |
|---|---|---|
| test R² | 0.067 | **0.107 ± 0.009** |
| test MAE | 1.681 | 1.648 |
| test r | 0.263 | 0.332 |
| best epoch | 1 | 10-15 |
| leakage | Stage-2 eval subjects in train | none |

### Next

Copy the chosen checkpoint to `/shared/tcgnet_data_2/proteomics_stage1_fluid_iq.pt`,
then Stage 2. Item 6 (proteomics scaler refit) is still open and is **not** fixed
by the exclusion — the scaler was fit before any of this and still saw joint
val/test rows.

---

## 2026-07-30 — Joint targets → instance 2; fluid IQ joint cohort 1,754 → 4,950

Measured coverage over the 5,328 joint participants and over the 52,697
proteomics participants:

| column | joint cohort | proteomics Stage-1 cohort |
|---|---|---|
| `21003-0.0` | 5,328 (100.0%) | 52,697 |
| `21003-2.0` | 5,328 (100.0%) | — |
| `20016-0.0` | 1,754 (32.9%) | **16,903** |
| `20016-2.0` | **4,950 (92.9%)** | 6,289 |

### Decision: instance 2 for the joint dataset, instance 0 for proteomics Stage 1

This refines FIX_PLAN item 4, which originally lumped both stages together. That
was wrong — the correct instance differs by stage, because the two modalities were
collected years apart.

**`fusion/data/build_joint_dataset.py` → instance 2.** The connectomes are from
the imaging visit and the fMRI encoder was pretrained on `20016-2.0`. Instance 0
meant supervising on a cognitive test taken ~9 years before the scan, and for age
on recruitment age, which trails the scan by 8.99 ± 1.79 years. It is also a
**2.82× data win** for fluid IQ: Stage-2 training goes 1,244 → ~3,465 and test
249 → ~742.

**`proteomics/data/build_dataset.py` → stays instance 0.** The Olink panel was run
on BASELINE samples (`UKB_Protein_BL_withCov.csv`), so instance-0 cognition is
contemporaneous with the blood draw. Instance 2 would also cut the pretraining
cohort to 6,289 — of which 4,950 are already in the joint cohort, so pretraining
would add almost nothing over Stage 2 itself. After exclusion that is ~3,400
training rows, and this model already overfits badly at 11,800.

Net design: each encoder is pretrained against the measurement matching its own
modality's collection date; the joint model is supervised on the imaging visit.
Both files now carry a comment explaining why they deliberately disagree, since
"aligning" them looks like an obvious cleanup and would silently break one stage
or the other. Worth a sentence in the methods.

Tonight's Stage-1 tuning (wd=1e-1) is unaffected — the proteomics dataset does not
change.

### Consequence

`dataset_joint_age.pt`, `dataset_joint_fluid_iq.pt`, `split_joint_1234.npz` and
`stage1_exclude_eids_1234.npy` must all be rebuilt. The exclusion list **will**
change this time: the joint split is unchanged (5,328, same seed), but the
per-target datasets are not.

---

## 2026-07-30 — Fluid IQ tuning settled: **wd=1e-1, test R² 0.113 ± 0.005**

Second sweep with `--epochs 100`. The optimum is now bracketed on both sides.

### Weight decay curve (seed 42)

| wd | best epoch | best val_loss | test R² | test MAE | test r |
|---|---|---|---|---|---|
| 1e-2 | 4 | 4.5397 | 0.0715 | 1.6760 | 0.3139 |
| 3e-2 | 14 | 4.4212 | 0.0828 | 1.6651 | 0.3081 |
| **1e-1** | 12 | 4.2897 | 0.1077 | 1.6407 | 0.3362 |
| 3e-1 | 24 | 4.2724 | 0.1104 | 1.6379 | 0.3346 |
| 1e0 | 24 | 4.4574 | 0.0697 | 1.6809 | 0.2772 |
| 3e0 | 18 | 5.8737 | −0.2770 | 1.9456 | **nan** |

### Seed stability at wd=1e-1

| seed | best epoch | test R² | test r |
|---|---|---|---|
| 42 | 12 | 0.1077 | 0.3362 |
| 1 | 10 | 0.1152 | 0.3486 |
| 2 | 16 | 0.1103 | 0.3383 |
| 3 | 20 | 0.1195 | 0.3467 |

**Mean test R² = 0.1132 ± 0.0052, r = 0.3425.**

### Findings

1. **wd=1e-1 is the answer.** Its four-seed mean (0.1132) sits ~9 sd above the
   wd=1e-2 distribution (0.0662 ± 0.0055). Not seed luck this time.
2. **1e-1 and 3e-1 are equivalent.** The single 3e-1 result (0.1104) falls inside
   the 1e-1 seed range (0.1077-0.1195). Prefer 1e-1: it converges in 10-20 epochs
   rather than 24.
3. **Correction to the previous entry.** I recorded wd=3e-1 as "truncated, not
   converged" because it ran all 30 epochs with best epoch 24. Re-running it with
   `--epochs 100` gave *bit-identical* results (best epoch 24, val_loss 4.272386,
   R² 0.1104) — it had already peaked at 24 and simply needed 10 more epochs to
   trigger the patience counter. Nothing was lost, and the epoch cap was never the
   binding constraint.
4. **Past 3e-1 the model collapses.** wd=3e0 gives R² = −0.277 and
   `test_pearson_r = nan` — the predictions became constant, so `torch.corrcoef`
   divided by a zero standard deviation. This is the missing zero-variance guard
   in `evaluate()` firing for real (see minor item 25); it should return a defined
   sentinel rather than nan.

### Net result for fluid IQ Stage 1

| | before | after |
|---|---|---|
| test R² | 0.067 | **0.113 ± 0.005** |
| test MAE | 1.681 | 1.635 |
| test r | 0.263 | 0.343 |
| best epoch | 1 | 10-20 |

+69% on R². Still a weak association — that is the honest ceiling for plasma
proteomics predicting fluid IQ — but it is now a real weak signal rather than a
model that quit at epoch 1. **Stage 2 for fluid IQ is worth pursuing.**

`WEIGHT_DECAY` in `config.py` stays at 1e-5: age Stage 1 performs well there
(R² 0.868) and has not been tested at 1e-1. Fluid IQ runs should pass
`--weight-decay 1e-1` explicitly until the two targets get separate defaults.

### Next

Blocked on the shared directory. Needs `corr_matrix.mat` and
`fmri_stage1_encoder.pt` in `/shared/tcgnet_data_2`, then build the joint dataset
to produce the exclusion list, then the final Stage-1 run with `--exclude-eids`
(not yet implemented).

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
