# NicheDECODE: Decoding changes in spatial architecture from tissue-level omics data

**NicheDECODE** is a deconvolution framework that infers niche-level spatial architecture abundance from tissue omics data, by constructing pseudo-tissue training data, integrating gene-wise spatial variability features, and incorporating functional similarity through a graph-based gating mechanism to model spatial priors. It demonstrates strong generalizability, robustness, and scalability across cross-technology, cross-dataset, and multi-omics benchmarks, consistently outperforming existing methods. Applied to bulk RNA-seq samples from an Alzheimer's disease cohort, NicheDECODE reveals layer-resolved cortical gray matter atrophy and relative white matter shifts consistent with established neuropathological trajectories, thereby bridging bulk-level cohorts with spatial tissue architecture to expand the scope of large-scale omics datasets for spatially informed clinical and population research.
<p align="center">
  <img width="60%" src="https://github.com/forceworker/NicheDECODE_main/blob/main/fig/workflow.png">
</p>
More details can be found in paper.

## Setup

### Dependencies and Installation

Workflow of NicheDECODE are implemented in python.The Python libraries used by NicheDECODE and their specific versions are saved in the environment.yml.

Create a new environment using environment.yml to support running NicheDECODE. The specific steps are as follows:

Step1:Type the directory where environment.yml is located in the terminal:

	> cd ~/NicheDECODE  

Step2:Create the environment with a custom name:

	> conda env create --name env_name -f environment.yml  

Step3:Activate the environment:

	> conda activate env_name 



### Notation
The Jupyter records of the various experiments in the NicheDECODE work can be found at: https://doi.org/10.5281/zenodo.15687743.