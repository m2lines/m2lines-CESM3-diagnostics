# Diagnostics of ocean models with mom6-tools
We assume that ~60-years runs in forced ocean are compeleted and the output is located in $SCRATCH. Next, we need to analyze the results with [mom6-tools](https://github.com/NCAR/mom6-tools).

## Conda environment
We start with creating a conda environment by cloning one of standard environments in NCAR:
```
module load conda
conda create --name env-from-npl-2024a --clone npl-2024a
```
The new environment can be found here:
```
/glade/work/pavelp/conda-envs/env-from-npl-2024a
```
feel free to use my environment or keep following this section.

Activate the environment and install the local version of mom6-tools:
```
conda activate env-from-npl-2024a
git clone https://github.com/NCAR/mom6-tools.git
cd mom6-tools
pip install -e .
```
