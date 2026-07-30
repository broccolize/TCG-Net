# `/shared/tcgnet_data_2` — contents manifest

What belongs in the new shared exchange directory, who produces it, and what is
deliberately left out. Derived by tracing every path constant in `fusion/`.

Code now resolves this directory from a single constant,
`proteomics/config.py::SHARED_DATA_DIR`, which defaults to
`/shared/tcgnet_data_2`. To point at the old generation without editing code:

```bash
export TCGNET_SHARED_DATA=/shared/tcgnet_data
```

---

## 1. From the fMRI branch — copy from `/home/celery/Project/TCG-Net/Data/`

| File | Required by | Why |
|---|---|---|
| `corr_matrix.mat` | `build_joint_dataset.py` (`CORR_MATRIX_PATH`) | FC matrices + `Patient_IDs`. Without it the joint dataset cannot be built. |
| `fmri_stage1_encoder.pt` | `train_stage2.py` (`FMRI_CKPT`) | fluid IQ Stage-2 encoder. |
| `fmri_stage1_splits.json` | *not read by code* — needed for the split decision | Tells us which subjects were in the fMRI Stage-1 held-out set. Required to compute the clean intersection for `FIX_PLAN.md` item 5. |
| `fmri_stage1_log.json` | *not read by code* — provenance | Records what val performance the encoder actually reached, and (importantly) that it was trained after the z-score fix. |

> [!danger] Missing and blocking: `fmri_stage1_age_encoder.pt`
> `train_stage2_age.py` loads this, and it is **not in the colleague's Data
> folder** — the listing contains only fluid-IQ encoders. It is also not produced
> by any committed script: all three fMRI Stage-1 scripts hardcode
> `FLUID_IQ_COL = '20016-2.0'` and write `fmri_stage1_encoder{,_alt1,_alt2}.pt`.
>
> **The age Stage-2 run cannot proceed until someone writes an age variant of
> `train_fmri_stage1.py` and commits it** (`FIX_PLAN.md` item 15). The old
> `/shared/tcgnet_data/fmri_stage1_age_encoder.pt` exists but was trained on the
> pre-fix graph features, so copying it forward would silently pair a stale
> encoder with corrected features.

### Do NOT copy

| File pattern | Why not |
|---|---|
| `fmri_graph_features_fold{0-4}.npz` | Stage-1 **CV fold** caches over the full fMRI cohort. Fusion computes its own `graph_feats_joint*.npz` over the joint cohort — different subjects, different file. |
| `fmri_stage1_encoder_fold{0-4}.pt` | Per-fold CV encoders. Stage 2 uses the single final encoder. |
| `*_alt1*`, `*_alt2*` (features, encoders, logs, splits) | Architecture-variant experiments. Nothing in `fusion/` references them. Copy only if you intend a variant comparison. |
| `roi_alignment.csv` | Written by `FC_corr.py` as a QC report; not read by any fusion code. |

## 2. From the proteomics branch — after the Stage-1 retrains

| Destination filename | Copy from | Required by |
|---|---|---|
| `proteomics_tensor.pt` | `proteomics/data/proteomics_tensor.pt` | `build_joint_dataset.py` (`PROCESSED_TENSOR_PATH`) |
| `proteomics_stage1_age.pt` | `proteomics/checkpoints/stage1_age_<run_id>_best.pt` | `train_stage2_age.py` (`PROT_CKPT`) |
| `proteomics_stage1_fluid_iq.pt` | `proteomics/checkpoints/stage1_fluid_iq_<run_id>_best.pt` | `train_stage2.py` (`PROT_CKPT`) |

> [!important] The proteomics checkpoint is now target-specific
> Both trainers previously loaded one shared `proteomics_stage1.pt`, which was
> verified to be the **fluid IQ** encoder — so the age run was warm-starting from
> a model fit to the wrong target. The filenames above are now distinct, and both
> trainers assert `ckpt['config']['target']` matches the run and refuse to start
> otherwise. Do not create a generic `proteomics_stage1.pt` in the new folder.

`proteomics/data/artefacts/scaler_1234.joblib` stays **in the repo** — it is
resolved repo-relative, not from the shared directory.

## 3. Generated in place — do not copy from the old folder

| File | Produced by |
|---|---|
| `dataset_joint_age.pt`, `dataset_joint_fluid_iq.pt` | `build_joint_dataset.py` |
| `split_joint_1234.npz` | `build_joint_dataset.py` |
| `graph_feats_joint.npz`, `graph_feats_joint_age.npz` | first Stage-2 run (cached thereafter) |

> [!tip] Starting a clean directory fixes the stale-cache trap for free
> The graph-feature cache validates only subject count, so copying the old
> `graph_feats_joint*.npz` forward would silently reuse features computed with the
> **pre-fix within-module z-score** and the colleague's correction would have no
> effect. A fresh directory guarantees they are recomputed.

---

## Checklist

```
/shared/tcgnet_data_2/
├── corr_matrix.mat                    ← colleague
├── fmri_stage1_encoder.pt             ← colleague
├── fmri_stage1_splits.json            ← colleague (for the item-5 decision)
├── fmri_stage1_log.json               ← colleague (provenance)
├── fmri_stage1_age_encoder.pt         ← ⚠️ BLOCKED — no script produces it
├── proteomics_tensor.pt               ← proteomics branch
├── proteomics_stage1_age.pt           ← proteomics retrain
├── proteomics_stage1_fluid_iq.pt      ← proteomics retrain
└── (everything else generated in place)
```

Fluid IQ Stage 2 can run once rows 1-2 and 6-8 are present. Age Stage 2 is blocked
on the missing fMRI age encoder.
