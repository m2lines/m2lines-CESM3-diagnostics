# Diagnostics of ocean models with mom6-tools
We assume that forced ocean simulations are completed and the output is located in $SCRATCH. Next, we need to analyze the results with [mom6-tools](https://github.com/NCAR/mom6-tools).

## Conda environment
Feel free to use my environment:
```
module load conda
conda activate /glade/work/pavelp/conda-envs/env-from-npl-2024a
```
Otherwise, create your own environment as shown below.

We start by creating a conda environment by cloning one of the standard environments in NCAR:
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
Provide the path to the experiment in `experiment.yaml`, and run the script:
```
./run_diagnostics.sh experiment.yaml
```
Feel free to uncomment the necessary diagnostics.

Note: short command to send computation to a compute node interactively:
```
qcmd -- ./run_diagnostics.sh experiment.yaml 
```
or as a standalone job:
```
qsub -v YAML_FILE=experiment.yaml pbs_script.pbs
```

See the results in figures:
```
tree PNG
```

Some diagnostics are stored as netcdf files:
```
ls ncfiles
```
