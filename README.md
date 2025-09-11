## Description
Code repository for the paper [Sigma Flows for Image and Data Labeling and Learning Structured Prediction](https://arxiv.org/abs/2408.15946).
Reproduces all the plots and experiments from the paper and contains all necessary data to reproduce optimization experiments.
It was developed by the [Image \& Pattern Analysis Group](https://ipa.math.uni-heidelberg.de) at Heidelberg University.

## Requirements
`python` version 3.11 or greater and a recent version of `pip` (25.2 or greater worked for us).
If these version requirement are not met, the results will differ from the paper.

For instance, to create a conda environment with the required dependencies run
```bash
conda create -n "sigmamodels" python=3.11
conda activate sigmamodels
```
Or if using uv, you can run
```bash
uv venv --python 3.13
source .venv/bin/activate   
```

## Installation
For a CPU-backed installation run the commands below.
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
python legacy/legacy.py
```
This will recreate the plots seen in the paper.
If you wish to save the pictures, add the `-c` flag to the calls to scripts (e.g. `python ex1/simplex.py -c`), which will store the outputs in the folder [artifacts](./artifacts).

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

Warnings like
```bash
E external/xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems. 
```
can be silenced by appending `2> /dev/null` to the commands.

## Tested environments
We tested the configuration with JAX version 0.7.0, installable via 
```bash
pip install -e ".[jax70]"
```

# Citation
```
@misc{cassel2024sigmaflowsimagedata,
      title={Sigma Flows for Image and Data Labeling and Learning Structured Prediction}, 
      author={Jonas Cassel and Bastian Boll and Stefania Petra and Peter Albers and Christoph Schnörr},
      year={2024},
      eprint={2408.15946},
      archivePrefix={arXiv},
      primaryClass={math.DS},
      url={https://arxiv.org/abs/2408.15946}, 
}
```
