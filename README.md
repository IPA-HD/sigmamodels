## Description
Code repository for the paper [Sigma Flows for Image and Data Labeling and Learning Structured Prediction](https://arxiv.org/abs/2408.15946).
Reproduces all the plots and experiments from the paper and contains all necessary data to reproduce optimization experiments.

## Requirements
`python` version 3.11 or greater and a recent version of `pip` (25.2 or greater worked for us).
󱐋󱐋 If these version requirement are not met, the results will differ from the paper. 󱐋󱐋

## Environment
For instance, to create a conda environment with the required dependencies run
```bash
conda create -n "sigmamodels" python=3.11
conda activate sigmamodels
```

## Installation
For a CPU-backed installation run the commands below in a bash shell.
```bash
pip install -e .
```
The installed packages are listed in the [pyproject.toml](./pyproject.toml) file.

For a GPU-backed installation, run
```bash
pip install -U "jax[CUDA12]"
pip install -e .
```

## Reproducing Plots from the Paper
The plots from the paper can be reproduced by running the commands
```bash
python ex1/simplex.py
python ex2/express.py
python ex3/voronoi.py
python ex3/house.py
```
This will recreate the plots seen in the paper. If you wish to save the pictures, add the `-c` flag to the calls to scripts (e.g. `python ex1/simplex.py -c`), which will store the outputs in the folder [artifacts](./artifacts).

## Reproducing Training from the Paper
In order to reproduce our optimization experiments, please run
```bash
python ex2/train.py
python ex3/train.py -m "sigmaflowv" -d "voronoi" -c
python ex3/train.py -m "unetv" -d "voronoi" -c
python ex3/train.py -m "sigmaflowh" -d "house" -c
python ex3/train.py -m "uneth" -d "house" -c
```
Running with the `-c` flag saves the trained weights in the folder artifacts.

## Tested environments
We tested the configuration with JAX version 0.7.0, installable via 
```bash
pip install -e ".[jax70]"
```
which ran on our machines.

## Recreate the training runs from the first version
If you care to recreate the training results from the first version of the paper, please reach out to us, we can provide additional data.
