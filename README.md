# TCG-Net

**TCG-Net** is a Two-branch Cooperative Game-Theoretic Network for individualised
cognitive assessment using functional connectivity and plasma proteomics. It
integrates resting-state fMRI functional connectomes and plasma proteomic
profiles through a cooperative Shapley fusion mechanism to predict cognitive
outcomes and provide individualised mechanistic interpretation.

This repository is a monorepo. Each branch of the architecture lives in its
own top-level folder so that the proteomics and fMRI sub-projects can be
developed independently, then integrated at the fusion stage.

## Module structure

| Folder        | Contents                                                                                              |
|---------------|-------------------------------------------------------------------------------------------------------|
| `proteomics/` | The plasma-proteomics SE-MLP encoder, its preprocessing pipeline, training loops, and analysis tools. |
| `fmri/`       | Reserved for the GCN-GAT functional-connectome encoder. Currently a placeholder.                      |
| `fusion/`     | Reserved for the Cooperative Shapley Fusion Module. Joint work, populated after both encoders land.   |
| `shared/`     | Reserved for utilities, configs, or preprocessing steps that are shared across the two branches.      |

## Branching conventions

Each module is developed on its own long-lived branch and merged into `main`
once its encoder is stable. Pull requests should keep changes scoped to a
single branch's folder.

| Branch              | Owns               |
|---------------------|--------------------|
| `proteomics-encoder`| `proteomics/`      |
| `fmri-encoder`      | `fmri/`            |
| `shapley-fusion`    | `fusion/` (joint)  |

The proteomics and fMRI branches are developed independently. The fusion
branch is started only after both encoders have been merged into `main`,
since it consumes `z_p` (proteomics embedding) and `z_f` (fMRI embedding)
from the two upstream encoders.

## Setup

```bash
git clone <repo-url> TCG-Net
cd TCG-Net

# Optional: create an isolated environment.
python -m venv .venv
source .venv/bin/activate

# Install dependencies for the branch you are working on; each module
# documents its own requirements in its own README.
```

The raw UK Biobank inputs are not included in the repository. They are
expected to live at `../uk_BIOBANK_DATA/` (sibling to `TCG-Net/`) and are
gitignored.

## Per-branch documentation

- [`proteomics/README.md`](proteomics/README.md) — proteomics encoder contract, preprocessing, Stage 1 training.
- [`fmri/README.md`](fmri/README.md) — fMRI connectome encoder placeholder.
- [`fusion/README.md`](fusion/README.md) — Shapley fusion placeholder.
- [`shared/README.md`](shared/README.md) — shared utilities placeholder.
