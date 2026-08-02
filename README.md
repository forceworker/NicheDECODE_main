# NicheDECODE

<pre align="center">
 _   _ ___ ____ _   _ _____ ____  _____ ____ ___  ____  _____
| \ | |_ _/ ___| | | | ____|  _ \| ____/ ___/ _ \|  _ \| ____|
|  \| || | |   | |_| |  _| | | | |  _|| |  | | | | | | |  _|
| |\  || | |___|  _  | |___| |_| | |__| |__| |_| | |_| | |___
|_| \_|___\____|_| |_|_____|____/|_____\____\___/|____/|_____|
</pre>

### Decoding changes in spatial architecture from tissue-level omics data

[![Tutorial](https://img.shields.io/badge/Tutorial-GitHub%20Pages-146b68)](https://forceworker.github.io/NicheDECODE_main/)
[![Data](https://img.shields.io/badge/Data-Zenodo-4f6f9f)](https://doi.org/10.5281/zenodo.18856556)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

**NicheDECODE** is a deconvolution framework for inferring niche-level spatial architecture abundance from tissue-level omics data. It constructs pseudo-tissue training samples, integrates gene-wise spatial variability features, and incorporates functional similarity through a graph-based gating mechanism to model spatial priors.

The framework is designed to bridge large tissue-level omics cohorts with spatial tissue architecture. Across cross-technology, cross-dataset, and multi-omics benchmarks, NicheDECODE demonstrates strong generalizability, robustness, and scalability. In an Alzheimer's disease bulk RNA-seq cohort, it recovers layer-resolved cortical gray matter atrophy and relative white matter shifts consistent with known neuropathological trajectories.

<p align="center">
  <img width="72%" src="https://github.com/forceworker/NicheDECODE_main/blob/main/fig/workflow.png" alt="NicheDECODE workflow">
</p>

## Highlights

- **Spatially informed deconvolution** from tissue-level omics profiles to niche-level architecture abundance.
- **Pseudo-tissue construction** using spatial sliding-window sampling and random mixed-cell simulation.
- **Graph-guided feature gating** that incorporates functional similarity and spatial variability features.
- **Reproducible examples** for human NSCLC and CODEX datasets, including notebooks, checkpoints, and output files.
- **Step-by-step tutorial website** for installation, data preparation, model training, prediction, and evaluation.

## Repository Structure

```text
NicheDECODE_main/
|-- data/              # Data-processing utilities and dataset-specific notebooks
|-- docs/              # GitHub Pages tutorial website
|-- exp/               # Model training, prediction, and evaluation notebooks
|-- fig/               # Workflow and manuscript figures
|-- model/             # NicheDECODE model implementation and utility functions
|-- res/               # Example prediction and ground-truth outputs
|-- save_models/       # Saved model checkpoints
|-- environment.yml    # Conda environment specification
`-- README.md
```

## Quick Start

Clone the repository and create a conda environment from the provided specification:

```bash
git clone https://github.com/forceworker/NicheDECODE_main.git
cd NicheDECODE_main
conda env create --name nichedecode -f environment.yml
conda activate nichedecode
```

Launch Jupyter Lab from the repository root:

```bash
jupyter lab
```

Then follow the example notebooks:

| Step | Notebook | Purpose |
| --- | --- | --- |
| 1 | `data/human_nsclc/data_process.ipynb` | Construct pseudo-tissue data for the human NSCLC example. |
| 2 | `data/CODEX/data_process.ipynb` | Construct pseudo-tissue data for the CODEX example. |
| 3 | `exp/human_nsclc/model_nicheDeconv.ipynb` | Train, predict, and evaluate NicheDECODE on human NSCLC. |
| 4 | `exp/CODEX/model_nicheDeconv.ipynb` | Train, predict, and evaluate NicheDECODE on CODEX. |

> **Note**
> The example model workflow uses CUDA tensors. A CUDA-enabled PyTorch environment is recommended for direct reproduction.

## Tutorial Website

A step-by-step tutorial is available at:

**https://forceworker.github.io/NicheDECODE_main/**

The tutorial covers environment setup, data download, pseudo-tissue data construction, model training, prediction, evaluation metrics, and expected output files. The website source is maintained in `docs/index.html` and deployed with GitHub Pages.

## Data and Experiment Records

The data used in the NicheDECODE examples can be downloaded from Zenodo:

- **Data archive:** [10.5281/zenodo.18856556](https://doi.org/10.5281/zenodo.18856556)
- **Jupyter experiment records:** [Zenodo record 18857669](https://zenodo.org/records/18857669)

After downloading the required files, place dataset-specific inputs under the corresponding folders in `data/` before running the notebooks.

## Outputs

Successful example runs generate prediction tables, ground-truth tables, and model checkpoints:

| Dataset | Prediction | Ground Truth | Checkpoint |
| --- | --- | --- | --- |
| Human NSCLC | `res/human_nsclc/nicheDeconv.csv` | `res/human_nsclc/real_ncslc.csv` | `save_models/human_nsclc/best_model.pt` |
| CODEX | `res/CODEX/nicheDeconv.csv` | `res/CODEX/real_CODEX.csv` | `save_models/CODEX/best_model.pt` |

## Citation

More details can be found in the accompanying paper. Please cite the NicheDECODE work if you use this repository, model, or tutorial in your research.

## License

This project is released under the [MIT License](LICENSE).
