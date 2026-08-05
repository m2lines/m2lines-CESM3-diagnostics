This repository serves for analysis of CESM climate model runs. 

* We plot figure similarly to notebooks/key_diagnostics.ipynb
* This notebook initializes the instance of a class Experiment defined in notebooks/experiment.py
* Climate data is heavy. We analyze it in two stages:
    * Precompute metrics
    * Read precompute metrics and plot
* Precomputing of metrics is mostly done by mom6-tools package which can be found in root (this) directory
* Script which runs mom6-tools is run_diagnostics.sh
* Results of this script are saved to ncfiles. Later on, manually, I copy these files to data/forced_55years or data/coupled_55years or similar folder depending on the kind of the run
* Currently, we are comparing coupled climate runs (data/coupled_55years) and forced ocean runs (data/forced_55years)
* The path to raw data can be found by passing corresponding .yaml file from yamls thorugh mom6-tools
* Note that name of the yaml file, and its identifier inside and its name in data folder commonly coincides
* Additional diagnostics not included to mom6-tools can be found in Experiment class itself. This commonly include atmospheric, ice, and other variables not processed by ocean package mom6-tools
* Never use worktree directory on the cluster
* Do not write outside of root directory
* You change notebooks. I run cells myself.