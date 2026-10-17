# Cooperative Shapley Fusion - TCG-Net

This directory contains joint-data preparation, cooperative Shapley fusion,
prediction heads, Stage 2 training drivers and analysis utilities. It consumes
the proteomics embedding `z_p` and fMRI embedding `z_f` from the two modality
encoders.

## Implemented components

| Component | Current source |
|---|---|
| Joint dataset construction | [`data/build_joint_dataset.py`](data/build_joint_dataset.py) aligns participants across proteomics and fMRI and builds joint age/fluid-intelligence datasets and split artifacts. |
| Scaler checks | [`data/check_scaler_leakage.py`](data/check_scaler_leakage.py) contains checks concerning the proteomics scaler and joint splits. The script's existence is not a guarantee that every experiment is leakage-free. |
| Cooperative fusion | [`models/shapley_fusion.py`](models/shapley_fusion.py) defines `CooperativeShapleyFusion` and `FusionOutput`, including per-modality/joint predictions, model-based Shapley quantities, an interaction index, gating weights and a fused embedding. |
| Prediction and auxiliary heads | [`models/prediction_head.py`](models/prediction_head.py) maps the fused embedding to a scalar prediction by default; [`models/h_p.py`](models/h_p.py) exposes the proteomics auxiliary-head alias. |
| Stage 2 loss | [`training/losses.py`](training/losses.py) combines the main prediction, auxiliary, Shapley-alignment and sparsity terms as implemented in the current code. |
| Training drivers | [`training/train_stage2.py`](training/train_stage2.py) targets the joint fluid-intelligence dataset; [`training/train_stage2_age.py`](training/train_stage2_age.py) targets the joint age dataset. |
| Analysis | [`analysis/`](analysis/) contains joint/pretrained baseline comparisons, phenotype inspection, experiment summaries and Shapley diagnostics. |

## Model interface

With the default embedding dimension, `CooperativeShapleyFusion` takes
`z_f` and `z_p` with shape `(B, 128)`. `FusionOutput` exposes:

- `y_hat_f`, `y_hat_p`, `y_hat_fp`: modality and joint predictions;
- `phi_f`, `phi_p`, `phi_int`: model-based modality contributions and an
  interaction index;
- `z_fusion`: fused representation, shape `(B, 128)`;
- `w`: three gating weights per sample, shape `(B, 3)`.

`PredictionHead` consumes `z_fusion`. The implementation includes
temperature-scaled gating; the source comments document differences from
the design equations. These diagnostics concern the prediction model and
do not establish biological causality or validated mechanistic explanations.

## Data and checkpoint requirements

Joint-data and training scripts import `SHARED_DATA_DIR` from
[`../proteomics/config.py`](../proteomics/config.py). Its current default is
`/shared/tcgnet_data_2`, overridable with `TCGNET_SHARED_DATA` before starting
the process. This is an external artifact directory, separate from the
repository's `shared/` utility placeholder.

The workflow requires external raw/processed inputs, compatible modality
encoder checkpoints and preprocessing artifacts. Check the path constants,
dataset keys and checkpoint-loading code in the selected driver; the age
and fluid-intelligence drivers use different artifact filenames. Installing
dependencies and cloning the source alone does not supply those inputs.

See the [top-level README](../README.md) and [fMRI guide](../fmri/README.md)
for the other modules. This page documents the implemented research code;
it adds no performance figures, completed-ablation claim or assertion of a
newly reproduced training run.
