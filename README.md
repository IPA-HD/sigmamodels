## Desciption
Code repository for the paper [Sigma Flows for Image and Data Labeling and Learning Structured Prediction](https://arxiv.org/abs/2408.15946).
Reproduces all the plots and experiments from the paper and contains all necessary binaries to reproduce optimization experiments.

## Installation
*Requirements*: `python` version 3.9 or greater. 

For a CPU-backed installation run the commands below in a bash shell.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
The installed packages are listed in the [pyproject.toml](./pyproject.toml) file.

For a GPU-backed installation, run
```bash
python3 -m venv .venv
source .venv/bin/activate
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
This will recreate the plots seen in the paper. If you wish to save the pictures,
add the `-c` option to the calls, which will store the outputs in the folder artifacts.

## Reproducing Training from the Paper
In order to reproduce our optimization experiments, please run
```bash
python ex2/train.py
python ex3/train.py -m "sigmaflowv" -d "voronoi"
python ex3/train.py -m "unetv" -d "voronoi"
python ex3/train.py -m "sigmaflowh" -d "house"
python ex3/train.py -m "uneth" -d "house"
```
Running the commands with an additional `-c` flag captures the trained weights in the folder artifacts.

## Tested environments
We tested the configuration with JAX version 0.7.0, installable via 
```bash
pip install -e ".[jax70]"
```
which ran on our machines.

The code was furthermore tested against the environments captured by the versions below
```bash
pip install -e ".[jax53]"
```
## Recreate the training runs from the first version
If you care to recreate the training results from the first version of the paper, please reach out to us, we can provide additional data.
