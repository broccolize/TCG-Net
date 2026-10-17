# TCG-Net

**TCG-Net** (Two-branch Cooperative Game-Theoretic Network) is a research
implementation for cognitive assessment using plasma proteomics and
resting-state fMRI functional connectivity. It combines modality embeddings
through cooperative Shapley-guided fusion and exposes model-based modality
contributions and interaction diagnostics.

The `main` branch contains proteomics, fMRI and fusion implementations.
The project remains ongoing: code availability does not establish completed
ablation studies, independently reproduced performance or clinical validity.
The descriptions below inventory the source currently present; they do not
add benchmark results or claim that a fresh end-to-end run has been verified.

## Implemented modules

| Directory | Source currently present |
|---|---|
| [`proteomics/`](proteomics/README.md) | Squeeze-and-excitation MLP encoder, auxiliary regression head, preprocessing, target-dataset construction, Stage 1 training/cross-validation scripts, baseline and importance analysis, and shape checks. |
| [`fmri/`](fmri/README.md) | Functional-connectivity preparation, GCN-GAT connectome encoder, graph-feature construction and Stage 1 training, plus alternative encoder/training scripts. |
| [`fusion/`](fusion/README.md) | Joint proteomics-fMRI dataset construction, cooperative Shapley fusion and prediction heads, Stage 2 losses/training drivers, baseline comparisons and interpretation/summary scripts. |
| [`shared/`](shared/README.md) | Documentation placeholder for a future shared utility package. It currently contains a README, not an implemented common-code package. |

### Code starting points

- **Proteomics encoder:** [`proteomics/models/encoder.py`](proteomics/models/encoder.py)
  and [`se_block.py`](proteomics/models/se_block.py). The configured input has
  2,920 features; the encoder exposes a 128-dimensional embedding and an SE
  gate vector. See [`config.py`](proteomics/config.py) for current defaults.
- **fMRI encoder:** [`fmri/Code/train_fmri_stage1.py`](fmri/Code/train_fmri_stage1.py)
  defines `ConnectomeEncoder` using GCN/GAT layers and graph pooling.
  [`FC_corr.py`](fmri/Code/FC_corr.py) prepares the connectivity inputs.
- **Fusion:** [`fusion/models/shapley_fusion.py`](fusion/models/shapley_fusion.py)
  defines `CooperativeShapleyFusion` and `FusionOutput`;
  [`prediction_head.py`](fusion/models/prediction_head.py) maps the fused
  representation to a prediction.
- **Training:** [`fusion/training/train_stage2.py`](fusion/training/train_stage2.py)
  uses the joint fluid-intelligence dataset; the separate
  [`train_stage2_age.py`](fusion/training/train_stage2_age.py) driver uses the
  joint age dataset. These scripts import both modality encoders.
- **Data and analysis:** [`fusion/data/build_joint_dataset.py`](fusion/data/build_joint_dataset.py)
  aligns participants across modalities. The `fusion/data/` and
  `fusion/analysis/` directories also contain scaler checks, baseline,
  phenotype-inspection and Shapley-analysis utilities.

The Shapley quantities describe the implemented prediction model. They should
not be interpreted as established biological mechanisms or causal effects.
Implementation-specific choices are documented in the source comments.

## Getting oriented

```bash
git clone https://github.com/broccolize/TCG-Net.git
cd TCG-Net

# Optional isolated environment (Linux/macOS shell):
python -m venv .venv
source .venv/bin/activate
```

There is no central pinned dependency manifest in this checkout. Depending on
the script, imports include PyTorch, PyTorch Geometric, NumPy, SciPy, pandas,
scikit-learn, h5py, NetworkX, joblib and tqdm. Check the selected entry point
and use dependencies compatible with your platform; an exact reproducible
training environment is not specified by this README.

### Data, paths and checkpoints

- Raw UK Biobank inputs are not supplied here. Running the data-dependent
  workflow requires access to the relevant inputs and preprocessing artifacts.
- Proteomics configuration derives raw-input paths from a sibling
  `../uk_BIOBANK_DATA/` directory. See [`proteomics/config.py`](proteomics/config.py).
- The same configuration defines `SHARED_DATA_DIR` for cross-modality tensors,
  datasets and checkpoints. Set `TCGNET_SHARED_DATA` before starting a process
  to override its current default, `/shared/tcgnet_data_2`.
- This external shared-data directory is distinct from the repository's
  `shared/` utility placeholder.
- Some fMRI preparation/training scripts still contain absolute Linux paths.
  Review their path constants and required files before running them; cloning
  the repository alone does not configure the full pipeline.
- Stage 2 expects compatible joint datasets, preprocessing artifacts and
  pretrained encoder checkpoints. Review each training driver's filenames
  and checkpoint-loading code before use.

## Module documentation

- [Proteomics guide](proteomics/README.md): encoder contract, preprocessing,
  training commands and recorded experiment notes. Consult the current
  configuration and source for paths and defaults.
- [fMRI guide](fmri/README.md): implemented connectivity and encoder scripts,
  inputs and path requirements.
- [Fusion guide](fusion/README.md): implemented joint-data, model, training
  and analysis components.
- [Shared utilities placeholder](shared/README.md): planned common-code scope.

Keep changes scoped to the relevant module. Historical branch names are not
prerequisites for browsing the implementations already present on `main`.
