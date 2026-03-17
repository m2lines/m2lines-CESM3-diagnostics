# Diagnostics of ocean models with mom6-tools
We assume that forced ocean simulations are compeleted and the output is located in $SCRATCH. Next, we need to analyze the results with [mom6-tools](https://github.com/NCAR/mom6-tools).

## Conda environment
Feel free to use my environment
```
module load conda
conda activate /glade/work/pavelp/conda-envs/env-from-npl-2024a
```
or keep following this section.

We start with creating a conda environment by cloning one of standard environments in NCAR:
```
module load conda
conda create --name env-from-npl-2024a --clone npl-2024a
```
The new environment can be found here:
```
/glade/work/pavelp/conda-envs/env-from-npl-2024a
```
Activate the environment and install the local version of mom6-tools:
```
conda activate env-from-npl-2024a
git clone https://github.com/NCAR/mom6-tools.git
cd mom6-tools
pip install -e .
```
## Computing diagnostics
Provide the path to the experiment in `experiment.yaml`, provide the conda environment in `run_diagnostics.sh`, and run the script:
```
./run_diagnostics.sh experiment.yaml
```
Note that you can uncomment necessary diagnostics.
See the results in the following folder:
```
ls ncfiles
```
output: `no-GM_mon_ave_global_means.nc`.
