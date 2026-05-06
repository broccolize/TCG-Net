# Proteomics Encoder — Analysis Report

**Scope:** Proteomics encoder only. The fMRI and sMRI branches are out of scope and are not addressed in this report.

---

## 1. File Structure Overview

### `research/` — scripts
| File | Purpose |
|---|---|
| `model_proteomics.py` | Defines `SEBlock` and `ProteomicsBranch` (the proteomics encoder). |
| `process_proteomics.py` | CPU pipeline: drop metadata → KNN imputation (`sklearn`) → z-score (`StandardScaler`) → save tensor. |
| `process_proteomics_gpu.py` | GPU-accelerated batched KNN imputation variant; same downstream z-score and save. Output written to `~/research/proteomics_tensor.pt`. |
| `find_targets.py` | Header scan over `phesantallphenos_*.csv` to locate cognitive field IDs (20016, 20023, 20191, 4282). |
| `find_age.py` | Header scan to locate age-at-recruitment field 21003. |
| `create_final_dataset.py` | Joins processed protein tensor to fluid intelligence (field `20016-0.0`) by `eid`, saves `dataset_final_fluid_iq.pt`. |
| `create_age_dataset.py` | Joins processed protein tensor to age (field `21003-0.0`) by `eid`, saves `dataset_age.pt`. |
| `training_loop_random_targets.py` | Smoke-test training loop on random targets (pipeline sanity check, 10 epochs). |
| `training_loop_final.py` | Real training loop. Despite the print labels, configured to load `dataset_age.pt` (age regression). 80/20 random split, 30 epochs, MSE loss, Adam, weight decay 1e-5, saves `best_model_iq.pth`. |
| `anaylze_iq_training_loop.py` | Trains on the **full** fluid-IQ dataset (no validation split), saves `model_fluid_iq_analysis.pth`, then dumps SE-block gate values to `protein_importance_iq.csv`. |
| `analyze_importance.py` | Loads `best_model_age_clock.pth`, recomputes per-protein average SE-block gate weights, writes `protein_importance_age.csv`. |

### `research/` — data / artefacts
| File | Purpose |
|---|---|
| `proteomics_tensor.pt` | Processed proteomics tensor (`x`, `eid`, `features`). Code comments imply shape (52697, 2920). |
| `dataset_age.pt` | Aligned (X, y, eid) tensor for age regression. |
| `dataset_final_fluid_iq.pt` | Aligned (X, y, eid) tensor for fluid intelligence regression (~16903 rows per script comment). |
| `best_model_age_clock.pth` | Trained checkpoint, used by `analyze_importance.py` for age. |
| `model_fluid_iq_analysis.pth` | Checkpoint from full-data IQ run. |
| `protein_importance_age.csv` | 2920 proteins × average SE gate values (sorted desc). |
| `protein_importance_iq.csv` | 2920 proteins × average SE gate values (sorted desc). |
| `Pre_Processing/` | Empty directory. |
| `__pycache__/model_proteomics.cpython-310.pyc` | Compiled bytecode. |

### `uk_BIOBANK_DATA/` — directory listing only
- `protein/`: `UKB_Protein_BL_withCov.csv` (input used), `UKB_Protein_Info.xlsx`, `UKB_Protein_BL_TimeGap.csv`, `data_ProcessDate.csv`, `olink_assay.dat`, `olink_assay_warning.dat`, `olink_batch_number.dat`, `olink_processing_start_date.dat`, `ukb_protein_batch.csv`, `UKB_Protein_KNNImpute.R`.
- `table/`: `phesantallphenos_01.csv` … `_08.csv`, `UKB_Biomarker.txt`, `UKB_Health_Outcomes_NumCol.csv`, `UKB_ICD10F_date.csv`, `UKB_ICD10_mental_disorders.csv`, `UKB_Outcomes_List.csv`, `UKB_PRS_info.xlsx`.
- `gene/`, `imaging/`: present but not relevant to this branch.

### Files NOT found
- No training logs, loss curves, TensorBoard runs, validation-curve dumps, or saved metric histories.
- No README or design notes.
- No notebook (`.ipynb`).
- No held-out test split or cross-validation artefacts.

---

## 2. Data Pipeline Assessment

| Check | Status | Notes |
|---|---|---|
| Input shape R^(N×2920) | ✅ Matches | Comments and importance CSVs confirm 2920 protein columns; N≈52,697 in the full processed tensor. |
| Z-score normalization | ⚠️ Applied, but on full dataset | `StandardScaler.fit_transform(X)` runs once over all rows in `process_proteomics*.py`. |
| Covariate residualization | ❌ Not done | Specification permits *either* z-score or covariate-residualized + standardized; only raw z-score is implemented. The CSV name (`…_withCov`) suggests covariates exist but they are not used to residualize. |
| Data leakage (normalization) | ❌ Leakage | Scaler and KNN imputer are fit on the entire dataset *before* the train/val split inside the training scripts. |
| Missing value handling | ⚠️ Partial | KNN imputation (k=5) on the full dataset. CPU version uses `sklearn.KNNImputer`; GPU version uses a custom batched cosine-style search initially seeded with column means — also fit on the full dataset. |
| Data loading | ✅ Functional | `eid`-based intersection between protein tensor and target CSV preserves alignment between X and Y. |

### Detailed findings
- **Shape:** Each row is a participant; 2920 columns after dropping seven metadata columns (`eid`, `batch`, `fastingtime_0_0`, `bloodsample_date_0_0`, `ProteinNum_0_0`, `PlateID_0_0`, `WellID_0_0`). Matches spec.
- **Normalization:** `StandardScaler().fit_transform(X_imputed)` is invoked globally, then the resulting tensor is later split 80/20 inside the training loops via `random_split`. Validation samples therefore contributed to the training-time mean and standard deviation. This is a textbook leakage pattern.
- **Covariates:** The source file is `UKB_Protein_BL_withCov.csv` (suggests batch / age / sex / centre covariates are present in the raw export), but the preprocessing scripts drop only seven metadata columns and never regress the protein features against those covariates. Spec allows raw z-score, so this is acceptable but suboptimal — TCG-Net interpretability would benefit from covariate residualization to reduce confounding.
- **Imputation leakage:** Both CPU and GPU KNN imputers are fit on all participants. For TCG-Net this should be re-fit on training rows only, with means/neighbours frozen for validation/test.
- **Data loading:** Looks correct. `eid` is preserved through preprocessing and used to align Y. Dataset sizes line up with reported numbers (52,697 proteomics rows; ~16,903 with fluid IQ; full ~52k with age).

---

## 3. Model Architecture Assessment

Implementation (`model_proteomics.py`):

```
SEBlock(2920, reduction=16)               # gates the RAW input
  Linear(2920, 182, bias=False)
  ReLU
  Linear(182, 2920, bias=False)
  Sigmoid
encoder = Sequential(
  Linear(2920, 512), BatchNorm1d(512), ReLU, Dropout(0.3),
  Linear(512, 256),  BatchNorm1d(256), ReLU,                 # NO dropout here
  Linear(256, 128)                                            # final embedding
)
regression_head = Linear(128, 1)
forward → (embedding, score, x_weighted)   # x_weighted = x * SE-gate
```

| Component | Expected (spec) | Found | Assessment |
|---|---|---|---|
| Stage 1 Linear | 2920 → 1024 | 2920 → 512 | ❌ Wrong width; the entire 1024-d hidden stage is missing. |
| Stage 1 BatchNorm | BN1d(1024) | BN1d(512) | ❌ Wrong dim (because the 1024 layer is absent). |
| Stage 1 activation | GELU | ReLU | ❌ Wrong activation. |
| Stage 1 Dropout | 0.3 | 0.3 | ✅ |
| SE-Block placement | After 1024-d hidden, on h_p | On raw 2920-d input | ❌ Operates on the wrong tensor. |
| SE-Block W1 | 1024 → 256 | 2920 → 182 | ❌ Wrong dims. |
| SE-Block hidden activation | GELU | ReLU (`inplace=True`) | ❌ Wrong activation. |
| SE-Block W2 | 256 → 1024 | 182 → 2920 | ❌ Wrong dims. |
| SE-Block output op | sigmoid → element-wise mul | sigmoid → element-wise mul | ✅ Form is right; tensors are wrong. |
| Stage 2 Linear | 1024 → 512 | 512 → 256 | ❌ Off by one (the spec's 1024→512 was collapsed). |
| Stage 2 BatchNorm | BN1d(512) | BN1d(256) | ❌ |
| Stage 2 activation | GELU | ReLU | ❌ |
| Stage 2 Dropout | 0.3 | (none) | ❌ Missing entirely. |
| Stage 3 Linear | 512 → 256 | (folded into stage 2) | ❌ |
| Stage 3 BatchNorm | BN1d(256) | (folded) | ❌ |
| Stage 3 activation | GELU | (none — ReLU after stage 2 dual-purposed) | ❌ |
| Stage 3 Dropout | 0.2 | (none) | ❌ Missing. |
| Final Linear | 256 → 128 | 256 → 128 | ✅ |
| Output dim z_p | 128 | 128 | ✅ |
| s_p exposure | Returned for interpretability | Not returned by `forward` | ⚠️ Recoverable post-hoc by calling `model.se_block.fc(x)` externally (done in `analyze_importance.py` and `anaylze_iq_training_loop.py`); never exposed as part of the forward contract. |

### Detailed findings
- The implementation is effectively a **3-Linear MLP** (2920 → 512 → 256 → 128) with a single SE gate applied to the *input*. The spec describes a **4-Linear MLP** (2920 → 1024 → 512 → 256 → 128) with SE applied to the 1024-d hidden representation. These are structurally different networks; the implementation is shallower and narrower at the top.
- Activation mismatch is consistent: every activation in the codebase is `ReLU`; the specification asks for `GELU` everywhere (encoder body and SE non-linearity).
- Dropout schedule is incomplete: only the first 0.3 is present. The second 0.3 (after 1024→512 stage) and the 0.2 (after 256→128 pre-final) are absent.
- SE-Block reduction ratio is 16 (so 2920/16 = 182 hidden units), which is the standard SE default; the spec instead fixes hidden size at 256, giving an effective reduction of 4 from the 1024-d hidden. The reduction ratio cannot simply be retuned without first moving the SE-Block to the right location.
- Note that `x_weighted = x * s_p` is the gated *input*, not the gate itself. Downstream code reaches into `model.se_block.fc(...)` to recover s_p — functional, but this couples interpretability code to the encoder's internal layout.

---

## 4. Prediction Model Results

The encoder uses an internal `regression_head = Linear(128, 1)` for both targets. No separate auxiliary heads, no classification head, no Shapley fusion plumbing.

### Age prediction (`training_loop_final.py` → `best_model_age_clock.pth`)
- **Head:** Linear(128, 1), MSE loss, Adam (lr 5e-4, weight decay 1e-5), batch 64, 30 epochs.
- **Split:** 80/20 random split via `torch.utils.data.random_split`. **No test set, no CV, no fixed seed observed.**
- **Metrics computed:** MSE (train + val), R² (val only, averaged across batches).
- **Saved metrics file:** None. No log, no curve, no JSON. Numerical results cannot be quoted because they were never persisted to disk.
- **Reasonableness:** Plasma proteomic age clocks in published UK Biobank work typically reach Pearson r ≈ 0.85–0.93 (MAE 3–5 years) using thousands of participants. We cannot evaluate whether this implementation reaches that benchmark because the runtime metrics were not retained.
- **Methodology concerns:**
  - R² is *averaged across batches* rather than computed once over all validation predictions; this biases the metric (per-batch variance ≠ pooled variance).
  - Best-model selection uses validation loss on data that was already seen by the global StandardScaler/KNN — see §2.
  - No held-out test set; no early stopping.

### Fluid intelligence prediction
Two scripts exist for IQ:
- `training_loop_final.py` *as written* loads `dataset_age.pt`, so it is in fact training the age model. Its print strings still reference “Fluid Intelligence”, which is misleading but does not change behaviour.
- `anaylze_iq_training_loop.py` loads `dataset_final_fluid_iq.pt` and **trains on the full dataset with no validation split**, by explicit comment (`"we know validation fails anyway, we want to see what it memorizes"`). It then dumps SE-gate values for interpretability.
- **Saved metrics file:** None. No held-out estimate of fluid-IQ predictive performance exists in the repository.
- **Reasonableness:** Fluid intelligence is much harder to predict from plasma proteins than chronological age; published estimates are typically modest (R² in the low single digits to ~0.05 from proteomics alone). The author’s own comment that the IQ model overfits and that validation “fails anyway” is consistent with that literature — but no quantitative estimate is logged here.
- **Methodology concerns:** Training without a validation split and saving a checkpoint after 30 full-data epochs guarantees an overfit model. Resulting SE gate rankings (`protein_importance_iq.csv`) should be interpreted as descriptive, not predictive.

### Importance CSV outputs
- `protein_importance_age.csv` and `protein_importance_iq.csv` each contain 2920 rows (Protein, Importance), where Importance is the per-protein mean of the post-sigmoid SE gate across the dataset.
- Top age proteins include MAX, ELN, CDKN2D, TJAP1, CCDC80, PVALB, AGER, ROBO1, COL6A3, NEFL — several (ELN, AGER, NEFL, COL6A3) are well-known ageing markers, which is a reasonable face-validity signal.
- Top IQ proteins include CPQ, IFIT1, NPL, MCEE, LGALS1, MYH4, ALMS1, TPT1 — none of these are dominant in the published cognition literature; combined with the lack of validation, the ranking should be treated as exploratory only.
- Caveat: because the SE gate sits on the **raw input** rather than the 1024-d hidden representation as specified, gate magnitudes here are *per-protein* importance, which actually maps neatly to interpretability — but it is **not** the s_p the spec describes.

---

## 5. Training Behaviour Summary

- **Logs:** None persisted. Both training scripts only `print()` per-epoch metrics; nothing is written to disk. Convergence behaviour cannot be reconstructed from this repository alone.
- **Train + val tracking:** `training_loop_final.py` tracks both train MSE and val MSE/R² each epoch (in stdout only). `anaylze_iq_training_loop.py` tracks neither — it does not even print loss inside the loop.
- **Best checkpoint:** `best_model_age_clock.pth` (≈10.9 MB) and `model_fluid_iq_analysis.pth` (≈10.9 MB). Both file sizes match the parameter count of the implemented architecture and would not match the larger spec architecture (which contains an extra 2920×1024 + 1024×512 worth of weights).
- **Stability / overfitting signals:** Cannot be assessed without logs. The IQ script *intentionally* overfits.
- **Best validation performance:** Not recorded; only `best_val_loss` is held in memory and printed at the end of the run, then discarded.

---

## 6. TCG-Net Compatibility Assessment

### Reusability verdict
**Not directly reusable.** The implemented network is a structurally different (shallower, narrower, ReLU-based) MLP from the SE-MLP that TCG-Net specifies, and the saved checkpoints will not load into the corrected architecture because layer shapes differ (`2920→1024`, `1024→512`, BN(1024) all absent from the current state dict; SE block has wrong widths). The data pipeline also has leakage that will bias any held-out evaluation in TCG-Net.

The 128-d output dimension and the *idea* of an internal SE gate are preserved. Everything between input and output, however, requires structural changes.

### Required changes (numbered)
1. Insert a leading `Linear(2920, 1024) → BatchNorm1d(1024) → GELU → Dropout(0.3)` block before any current layers.
2. Move the `SEBlock` to operate on the 1024-d hidden representation (i.e., between Stage 1 and Stage 2), not on the raw 2920-d input.
3. Reparameterise `SEBlock` so `W1: 1024 → 256` and `W2: 256 → 1024`, replace inner `ReLU` with `GELU`, and keep the final `Sigmoid`.
4. Rebuild the encoder body as `Linear(1024, 512) → BN(512) → GELU → Dropout(0.3) → Linear(512, 256) → BN(256) → GELU → Dropout(0.2) → Linear(256, 128)`.
5. Replace every remaining `ReLU` with `GELU` (specification is uniform on this).
6. Modify `forward` to also return `s_p` (the post-sigmoid SE gate vector) explicitly, e.g. `return z_p, s_p` (or a dict). Decouple interpretability from internal-attribute access.
7. Remove or refactor `regression_head`. In TCG-Net the encoder feeds three downstream heads (y_p, y_fp, Cooperative Shapley Fusion), so the encoder should output `z_p` cleanly and not bind itself to a single MSE head.
8. Refactor preprocessing so KNN imputation and `StandardScaler` are fit on the training split only, with parameters frozen and reapplied to validation/test. Optionally add covariate residualization, since the spec explicitly allows it and the source CSV is `*_withCov.csv`.
9. Recompute fresh checkpoints after the architecture and pipeline are fixed; existing `.pth` files are not state-dict-compatible with the corrected model.
10. Add a held-out test set (and ideally k-fold CV) plus persistent metric logging so TCG-Net evaluation has reproducible baselines.

### SE-block interpretability exposure
- Currently **not exposed** through the public interface. `forward` returns `(embedding, score, x_weighted)`, where `x_weighted` is the *gated input*, not the gate itself.
- Interpretability code recovers the gate by calling `model.se_block.fc(x)` directly — workable but brittle, and it bypasses the canonical forward path (no gradient hooks, no dropout/BN-aware path).
- Cooperative Shapley fusion will need a stable contract for s_p; recommend returning it from forward (e.g. as `z_p, s_p`).

### Auxiliary head compatibility with the Shapley Fusion Module
- The current `regression_head = Linear(128, 1)` can serve as the y_p auxiliary head conceptually, but:
  - It is hard-wired into `forward` and trained jointly with the encoder. For Shapley-style cooperative fusion, the encoder needs to expose `z_p` independently, with the y_p head pluggable.
  - There is no y_fp (joint) head and no scaffolding for the fusion module — those are entirely absent.
  - Loss is plain MSE; a multi-task objective combining y_p, y_fp, and the cooperative Shapley term has not been set up.
- Net assessment: the existing head can be retained as a starting point for y_p, but the encoder must be refactored to return `z_p` (and ideally `s_p`) cleanly, and the y_fp + fusion infrastructure must be added separately.

---

## 7. Overall Task Completion Verdict

### Items completed correctly
- End-to-end data ingestion from UK Biobank protein CSVs to a single tensor file (`proteomics_tensor.pt`) with preserved `eid` keys and 2920 protein columns.
- Target-alignment scripts for age and fluid intelligence by `eid` intersection (`create_age_dataset.py`, `create_final_dataset.py`).
- 128-dimensional output embedding (matches spec).
- Element-wise SE gating mechanism (sigmoid on a two-layer bottleneck MLP, multiplicative gate) — correct *form*, wrong *placement and dimensions*.
- A working training loop with train/val split for age regression (architecture and methodology issues notwithstanding).
- Two interpretability CSVs with per-protein gate weights, ranking 2920 proteins.
- Two saved checkpoints (`best_model_age_clock.pth`, `model_fluid_iq_analysis.pth`) — usable as artefacts but not reloadable into the spec architecture.

### Items partially complete or requiring review
- SE-Block: exists, but on the wrong tensor (raw input instead of 1024-d hidden), with the wrong reduction shape, the wrong inner activation (ReLU vs. GELU), and not exposed through `forward`.
- Normalization: z-score is applied, but with leakage across the train/val split.
- Missing-value handling: KNN imputation is implemented (twice), but fit on the full dataset.
- Age training: runs and saves a checkpoint, but no metrics are persisted, the val R² is computed per-batch (biased estimator), and the model is selected on a leakage-contaminated validation split.
- Auxiliary head: a single `Linear(128, 1)` exists; usable as a seed for y_p but not yet decoupled from the encoder forward and not extended to y_fp / Shapley fusion.

### Items missing or incorrect
- **Architecture (multiple):** missing the entire `Linear(2920,1024) → BN(1024) → GELU → Dropout(0.3)` stage; SE-Block widths wrong; ReLU used instead of GELU throughout; Dropout(0.3) after stage 2 missing; Dropout(0.2) before final layer missing; one Linear stage collapsed.
- **Covariate-residualized standardization** option is not implemented (only raw z-score).
- **Test set:** none. **Cross-validation:** none. **Random-seed control:** none.
- **Training logs / metric persistence:** no log files, no curves, no JSON of best metrics.
- **Held-out fluid-IQ evaluation:** not done; the IQ script trains on the full dataset by design.
- **`s_p` exposed via forward:** not done.
- **y_fp (joint) head and Cooperative Shapley Fusion plumbing:** entirely absent (expected — these are TCG-Net-level additions, not part of the proteomics encoder, but flagged so the researcher knows what is still missing downstream of this branch).
- **Decoupling of regression head from encoder:** not done.

**Overall verdict: partially complete, with significant rework required.** The data ingestion path and the conceptual building blocks (SE gate, encoder MLP, embedding-to-head pattern) are in place, but the actual encoder is structurally divergent from the TCG-Net specification, the training pipeline has leakage, and no quantitative results are logged. The branch needs an architecture rebuild plus a pipeline cleanup before it can serve as TCG-Net's proteomics branch.

---

## 8. Recommendations

1. **Rebuild `ProteomicsBranch` to match the spec exactly.** Replace the current 3-Linear / ReLU / SE-on-input network with the 4-Linear / GELU / SE-on-hidden(1024) network described in §3. Update `SEBlock` to `1024 → 256 → 1024` with GELU + sigmoid. Verify with a forward-pass shape unit test that input `(B, 2920)` produces `z_p` of shape `(B, 128)` and `s_p` of shape `(B, 1024)`.
2. **Expose `s_p` from `forward`.** Change the return signature to `(z_p, s_p)` (and optionally a y_p prediction). Drop external access to `model.se_block.fc(...)` from interpretability code.
3. **Eliminate normalization/imputation leakage.** Refactor preprocessing into a fit-on-train, transform-on-val/test pattern. Use a `Pipeline` with a `KNNImputer` and `StandardScaler` fit on indices belonging to the training fold only, then persist the fitted transformer for reuse.
4. **Introduce a proper split protocol.** Adopt a fixed train / validation / test split (e.g., 70/15/15) with a deterministic seed, plus k-fold CV for both age and fluid intelligence. Persist split indices to disk so TCG-Net evaluations are reproducible.
5. **Persist training metrics.** Log per-epoch train/val loss, R², MAE, and Pearson r to a CSV (or TensorBoard / Weights & Biases run) so model selection and convergence can be audited later. Compute R² on pooled validation predictions, not as a per-batch mean.
6. **Decouple the encoder from its prediction head.** Move `regression_head` out of `ProteomicsBranch` into a separate `ProteomicsAuxHead(in=128, out=1)` module, instantiated alongside the encoder by the training script. This unblocks adding y_fp and the Cooperative Shapley Fusion module without modifying the encoder.
7. **Re-train both targets on the corrected architecture and pipeline**, then regenerate `protein_importance_age.csv` and `protein_importance_iq.csv` from the new s_p (now 1024-d, so importance attribution should aggregate gate values per protein via gradient × input or a learnable input-attention layer — clarify with the design before re-implementing).
8. **Fix the script-name/intent mismatch.** `training_loop_final.py` currently loads `dataset_age.pt` while printing fluid-IQ messages, and `anaylze_iq_training_loop.py` (note the typo) trains on the full dataset by design. Split these into clearly named `train_age.py` / `train_fluid_iq.py` with consistent labelling, and remove the deliberate full-data overfit run.
9. **Consider covariate residualization.** The source CSV is `UKB_Protein_BL_withCov.csv`; loading the covariates (age, sex, batch, assay-warning flags, plate, fasting time) and residualising the proteins against them before z-scoring will reduce confounding and gives the spec’s preferred input form. This is especially important for the age target, where batch effects can leak into apparent signal.
10. **Document the encoder contract.** Add a short README or docstring covering: input shape, output shape (z_p, s_p), preprocessing requirements, expected normalization statistics location, and how the encoder is meant to plug into TCG-Net. This makes downstream integration with the fMRI/sMRI branches and the Shapley fusion module unambiguous.
