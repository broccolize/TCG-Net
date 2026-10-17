# fMRI Connectome Encoder - TCG-Net

This directory contains functional-connectivity preparation and fMRI encoder
training code. The implementation is under [`Code/`](Code/).

## Implemented scripts

| File | Role in the current source |
|---|---|
| [`Code/FC_corr.py`](Code/FC_corr.py) | Builds 66-ROI functional-connectivity matrices from BOLD time series, using an sMRI-derived ROI order and subject alignment; writes connectivity and ROI-alignment outputs. |
| [`Code/train_fmri_stage1.py`](Code/train_fmri_stage1.py) | Defines the GCN-GAT `ConnectomeEncoder`, connectivity graph construction, graph-theoretic node features, an auxiliary regression head and Stage 1 training/evaluation routines. |
| [`Code/train_fmri_stage1_alt1.py`](Code/train_fmri_stage1_alt1.py) | Alternative encoder/training implementation for comparison. |
| [`Code/train_fmri_stage1_alt2.py`](Code/train_fmri_stage1_alt2.py) | Alternative implementation including single- and dual-channel encoder classes. |

The standard `ConnectomeEncoder` uses GCN/GAT layers, mean/max graph pooling
and a projection to a 128-dimensional embedding. The joint training drivers
in [`../fusion/training/`](../fusion/training/) import the encoder and data
helpers from `train_fmri_stage1.py`.

## Inputs and execution requirements

The preparation and training scripts depend on external UK Biobank imaging
and phenotype inputs. The current standard training script uses fluid
intelligence from imaging-visit field `20016-2.0`. Raw inputs and generated
connectivity tensors/checkpoints are not provided by this guide.

Several scripts have absolute Linux path constants for raw inputs, cached
graph features and outputs. Review those constants and the selected script's
cohort/target settings before running it. In particular, `FC_corr.py` also
expects the sMRI CSV used for ROI and participant alignment. The exact
training environment and input artifacts must be prepared separately.

Imports include PyTorch, PyTorch Geometric, NumPy, SciPy, pandas,
scikit-learn, NetworkX and tqdm; `FC_corr.py` additionally uses h5py.

## Scope of this documentation

These are implemented research scripts, including experimental alternatives.
Their presence does not establish comparative performance, completed
ablations or a freshly verified end-to-end run. No new training result or
benchmark claim is introduced here.
