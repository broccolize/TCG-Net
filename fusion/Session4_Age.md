# Session 4 — Age Prediction (TCG-Net Stage 2)

Training run on `dataset_joint_age.pt` using `train_stage2_age.py`.
Encoders: `fmri_stage1_age_encoder.pt` + `proteomics_stage1.pt`.
Early stopping at epoch 87. Best checkpoint: epoch 67, val MAE **0.3639 z / 2.822 yrs**.

---

## A. Training Dynamics

- **Phase 1 (epochs 1–10, frozen encoders, T=1.0):**
  - Val MAE: 0.734 z (5.692 yrs) → 0.692 z (5.363 yrs). Steady but modest improvement.
  - Gate near-uniform throughout (w_f≈0.34, w_p≈0.33, w_int≈0.33) — expected with T=1.0.
  - l_aux dominates at epoch 1: 0.3·2.831=0.849 vs l_main=0.419 (~67% vs ~33%), same imbalance as fluid IQ.

- **Phase 2 transition (epoch 11, T→0.1, encoders unfrozen):**
  - Brief regression: val MAE jumps to 0.712 z — normal disruption from encoder unfreezing.
  - Gate shifts immediately on T sharpen: w_f=0.406, w_p=0.295 — fMRI briefly dominant.
  - Then rapid, sustained improvement across 56 epochs: 0.712 → 0.364 z.
  - **Phase 2 delivered a 47% MAE reduction** (0.692 → 0.364 z). Dramatically larger than the ~2% seen for fluid IQ in the same architecture.

- **Best epoch 67:**
  - train_mae=0.270 z (2.096 yrs), val_mae=0.364 z (2.822 yrs). Train/val ratio 1.35 — mild overfit beginning.
  - l_main=0.057, 0.3·l_aux=0.297 — loss composition has inverted; l_main and l_aux are now comparable. Aux heads genuinely learned age.

- **Late overfitting (epochs 68–87):**
  - Train MAE continues falling (0.250 z / 1.939 yrs at e87) while val MAE climbs (0.382 z / 2.963 yrs).
  - Early stopping correctly triggered at patience=20 (epoch 87).

---

## B. Gate Analysis

| Epoch | Phase | w_f | w_p | w_int | Notes |
|---|---|---|---|---|---|
| 1 | 1 | 0.365 | 0.301 | 0.334 | Near-uniform (T=1.0) |
| 10 | 1 | 0.339 | 0.330 | 0.331 | Still near-uniform |
| 11 | 2 | 0.406 | 0.295 | 0.299 | T→0.1 sharpens; fMRI briefly leads |
| 67 (best) | 2 | 0.192 | **0.479** | 0.329 | Proteomics dominant |
| 87 (final) | 2 | 0.203 | **0.471** | 0.326 | Stable proteomics dominance |

- Proteomics ends up dominant (w_p≈0.48), identical qualitative pattern to the fluid IQ run.
- The interaction term (w_int≈0.33) is the second-largest contributor — joint fMRI×proteomics information is meaningful.
- fMRI weight (w_f≈0.19) is the smallest, but non-trivial.

---

## C. Signal Strength vs Fluid IQ

| Metric | Fluid IQ | Age |
|---|---|---|
| Best val MAE (z) | 0.7271 | **0.3639** |
| Best val MAE (units) | 1.484 IQ pts | **2.822 yrs** |
| Approx mean-predictor z-MAE | ~0.734 | ~0.80 |
| Approx reduction vs mean predictor | ~1% | **~54%** |
| Phase 2 MAE improvement | ~2% | **~47%** |
| Best epoch | 1 | 67 |
| l_aux at best epoch | ~2.5 (aux barely learned) | **0.991 (aux genuinely learned)** |

- Age prediction has far more signal in fMRI + proteomics than fluid IQ. The model is learning real structure, not hovering at the mean-predictor floor.
- The aux heads dropping from l_aux≈2.83 (epoch 1) to ≈0.99 (epoch 67) confirms that individual-modality predictions improved substantially — the signal exists in both branches.
- y_std ≈ 7.75 years (derived: 6.289 yrs / 0.811 z at epoch 1). Best MAE of 2.822 yrs = 0.36 standard deviations.

---

## D. Loss Composition

- **l_shapley = 0 at every epoch** — by construction (same as fluid IQ; φ_int := ŷ_fp − y0 − φ_f − φ_p makes the Shapley consistency term identically zero). ✓
- **l_sparse (af_l1) ≈ 0.071 throughout** — stable, negligible at λ=1e-4. ✓
- **sp_l1 ≈ 0.487–0.498** — monitored but not in the active loss (removed in Session 3 fix #5). ✓
- **l_aux collapse**: dropped from 2.831 → 0.991 by best epoch — healthy; aux heads are learning, not stagnant.

---

## E. Result Summary

**Best model:** `fusion/training/checkpoints_age/best_model.pt`
- Val MAE: **0.3639 z / 2.822 years** (epoch 67, phase 2)
- Stops at epoch 87 (patience=20)

Age prediction with TCG-Net is **substantially more successful** than fluid IQ prediction. The ~54% reduction over the mean predictor (vs ~1% for fluid IQ) reflects that biological age is genuinely encoded in both fMRI connectivity patterns and plasma proteomics, whereas fluid IQ was at the signal floor.

The scientific claim of cooperative multimodal fusion is **more supportable here** than for fluid IQ — the model demonstrably learns, the gate differentiates meaningfully between modalities, and the aux heads converge. The ablation (§10: M1 proteomics-only vs M10 full TCG-Net) would still be needed to isolate the fusion contribution specifically, but the headline result is strong.

---

## What's Next

- **§10 Ablation:** Run M1 (proteomics-only) and M2 (fMRI-only) for age to quantify the fusion gain.
- **Test set evaluation:** Report test MAE from the saved best checkpoint.
