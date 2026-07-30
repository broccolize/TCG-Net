# TCG-Net — Fix Plan

Prioritised list of defects found in the code reviews of `proteomics/` and
`fusion/` (2026-07-30). Tick items off as they land; record the change in
[`LOG.md`](LOG.md).

Ordering note: **the order at the bottom matters more than the tiers.** Several
fixes change what a training run produces, so doing them in the wrong sequence
wastes hours of compute.

---

## Critical — a reported number or claim is currently wrong

| # | Fix | Where | Re-run? | Status |
|---|---|---|---|---|
| 1 | `phi_int` → real 2-way interaction index; `l_shapley` redefined to avoid the `phi_int²` trap | `fusion/models/shapley_fusion.py:48`, `fusion/training/losses.py:20` | yes | ✅ **done 2026-07-30** |
| 2 | Retrain proteomics Stage 1 — current artefact is an epoch-1, `val_r2=0.047` fluid-IQ checkpoint feeding **both** Stage-2 runs | `proteomics/training/train_stage1.py` | yes | ⬜ code ready, **retrain pending** |
| 3 | Make `PROT_CKPT` target-specific; assert `ckpt['config']['target'] == target` | `fusion/training/train_stage2_age.py:73,109` | no | ✅ **done 2026-07-30** |
| 4 | Targets → instance 2 (`21003-2.0` / `20016-2.0`); rebuild joint datasets | `fusion/data/build_joint_dataset.py:97,101` | rebuild | ⬜ |
| 5 | Decide the split strategy — re-pretrain both encoders on joint-train only, **or** evaluate on the intersection of both Stage-1 held-out sets | `fusion/data/build_joint_dataset.py:365` | yes | ⬜ **judgement call** |
| 6 | Refit the proteomics scaler/imputer on the joint train fold (line 266 is currently an explicit no-op) | `fusion/data/build_joint_dataset.py:259-266` | yes | ⬜ |
| 7 | Raise `WEIGHT_DECAY` above 1e-5 for fluid IQ | `proteomics/config.py` | yes | ⬜ flag added, value unchanged |
| 8 | Aggregate **final-epoch** CV metrics, not best-epoch (fluid IQ sign flips: +0.059 → −0.084) | `proteomics/training/cv_stage1.py:302,439` | recompute | ⬜ |
| 9 | Actually evaluate `test_loader` — 799 age / 249 fluid IQ held-out subjects unused | `fusion/training/train_stage2.py:97` | no | ⬜ |
| 10 | Fluid IQ Stage 2 predicts a near-constant (`y_hat` std 0.013 vs `y_true` std 1.0); val MAE 0.727 is ~9% better than guessing the mean | `fusion/models/prediction_head.py` + item 2 | yes | ⬜ |
| 11 | Softmax `temperature=0.1` saturates the gate — age weights are binary (0/1, std 0.46), fluid IQ frozen near ⅓ (std 0.02). Neither is adaptive soft fusion | `fusion/models/shapley_fusion.py:52` | yes | ⬜ |

## Major — a mechanism silently isn't working, or you can't reproduce it

| # | Fix | Where | Status |
|---|---|---|---|
| 12 | Stop detaching GAT attention so `l_sparse` has a gradient: `self.last_attn = (attn_ei.detach(), attn_w)` | `fmri/Code/train_fmri_stage1.py:500` | ⬜ |
| 13 | `.eval()` on frozen encoders while `phase == 1` — `requires_grad_(False)` does not stop BatchNorm buffer updates or dropout | `fusion/training/train_stage2.py:165` | ⬜ |
| 14 | Seed Stage 2 — no `set_global_seeds`, no `DataLoader(generator=...)` anywhere | `fusion/training/train_stage2*.py` | ⬜ |
| 15 | **Write and commit an age variant of `train_fmri_stage1.py`.** Now blocking: the colleague's re-run produced only fluid-IQ encoders, and the old `fmri_stage1_age_encoder.pt` was trained on pre-fix graph features. Age Stage 2 cannot run until this exists | `fmri/Code/` | ⬜ **blocking** |
| 16 | Anchor the covariate matcher to exact column names — currently binds `AGER`/`MAGEA3` as the age covariate. **Do not run `--residualize` until fixed** | `proteomics/data/preprocess.py:184` | ⬜ |
| 17 | Correct `Session3_Diagnosis.md:83` and `:156`, which mark the freeze and the scaler leakage as resolved. Both are false | `fusion/Session3_Diagnosis.md` | ⬜ |
| 18 | Commit `fusion/analysis/*.py` — four untracked files that produced your figures, existing only on one server | `fusion/analysis/` | ⬜ |
| 19 | Review `fusion/analysis/interpret_*.py` — never covered by the review; check the subtype rule (counts are near-exact thirds, which looks like tertile binning) | `fusion/analysis/` | ⬜ |

## Minor — hygiene and latent hazards

| # | Fix | Where | Status |
|---|---|---|---|
| 20 | `drop_last=True` on the Stage-2 train loader (inert at 18/28 today, re-arms after item 4) | `fusion/training/train_stage2.py:95` | ⬜ |
| 21 | Apply `BAD_FC_EIDS` in the joint build (empty intersection today — protective by accident) | `fusion/data/build_joint_dataset.py:291` | ⬜ |
| 22 | Key the graph-feature cache by split **and feature definition**; delete both caches before any re-run — see the ⚠️ note below, this is now urgent | `fusion/training/train_stage2.py:90` | ⬜ **live** |
| 23 | Make the alignment check touch the fMRI side; assert `fc_all.shape[0] == fc_ids.size` and `fc_all.shape[1] == 66` | `fusion/data/build_joint_dataset.py:339,351` | ⬜ |
| 24 | `dtype=str` not `dtype=object` in `np.savez` — artefacts currently unreadable by default `np.load` | `proteomics/data/preprocess.py:582,619` | ⬜ |
| 25 | Reconcile the three `sp_l1_norm` definitions; the logged one is a biased mean of per-batch means | `proteomics/training/train_stage1.py:248` | ⬜ |
| 26 | Import `train_one_epoch` instead of retyping it — the copy has already drifted | `proteomics/training/cv_stage1.py:259` | ⬜ |
| 27 | Add an EID-uniqueness assert (passes today: 52697 rows, 52697 unique) | `proteomics/data/preprocess.py:386` | ⬜ |
| 28 | `HIDDEN_DIM_1` instead of hardcoded `1024` | `proteomics/analysis/protein_importance.py:156` | ⬜ |
| 29 | Reconcile the `.pt` vs `.npz` index spaces, or delete the unread `split_joint_*.npz` | `fusion/data/build_joint_dataset.py:479` | ⬜ |

---

## ⚠️ Upstream change (2026-07-30) — the fMRI feature fix will NOT reach fusion

`origin/main` gained two commits (`dc0bfa4`, `28bb8b9`) that correct the
**within-module degree z-score** in `compute_graph_features`
(`fmri/Code/train_fmri_stage1.py`). It previously z-scored *total* strength; it
now z-scores *within-module* strength per Guimerà & Amaral. **This changes the
node features.**

Two consequences that will bite silently:

1. **The graph-feature caches on the server are now stale in a new way.** The
   cache guard only checks subject count (`cached.shape[0] == len(fc_matrices)`),
   so re-running Stage 2 reloads the OLD features and the fix has no effect.
   Delete `graph_feats_joint.npz` and `graph_feats_joint_age.npz` before any run.
2. **Both `fmri_stage1_*.pt` encoders were trained on the old features.** They are
   now stale relative to the code. Loading them while computing new features is a
   train/inference mismatch. They need re-training whenever item 5 is settled —
   which is convenient, since item 5 likely requires re-pretraining anyway.

Confirmed unaffected by the upstream change: `FLUID_IQ_COL = '20016-2.0'`
(item 4 stands), `BAD_FC_EIDS` (item 21 stands), and the detached `last_attn`
(item 12 stands — it moved from line 500 to **line 504**).

---

## Recommended order

1. **Code-only batch** — 1 ✅, 3, 7, 8, 9, 12, 13, 14, 16. Cheap, and several change
   what a re-run produces.
2. **Item 5** — the split-strategy decision. Everything downstream depends on it,
   and it is the only genuine judgement call in the list.
3. **Item 2** — retrain proteomics Stage 1 with early stopping in place.
4. **Items 4 + 6** — rebuild the joint datasets with instance-2 labels and a
   refit scaler.
5. **Delete the graph-feature caches** (item 22) — otherwise the stale cache is
   silently reloaded and the leakage survives the fix invisibly.
6. **Re-run Stage 2**, then regenerate everything under `fusion/analysis/results/`.

Doing 2 or 4 before 5 wastes the run.

---

## Item 5 — the decision you need to make

The joint split is currently drawn independently of both Stage-1 splits, so ~85%
of Stage-2 val/test subjects were seen by the encoders during Stage-1 training.

| Option | Test set size | Cost |
|---|---|---|
| Evaluate only on subjects held out by **both** Stage-1 runs | ~160 (age), ~50-70 (fluid IQ) | free, but confidence intervals are wide |
| **Re-pretrain both encoders on joint-train only** | 799 (age), 249 (fluid IQ) | two extra Stage-1 runs; methodologically standard for two-stage transfer |
| Grow the joint cohort | — | data-pipeline question; age is already 5,328 |

Recommended: re-pretrain. Age has 5,328 joint participants, which makes it
considerably more viable than the fluid IQ figure alone suggests.
