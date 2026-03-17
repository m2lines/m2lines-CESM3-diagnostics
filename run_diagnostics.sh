#!/bin/bash

# Usage on login node: ./run_diagnostics.sh experiment.yaml
# Usage on compute node: qcmd -- ./run_diagnostics.sh experiment.yaml 

YAML_FILE="$1"
BASENAME=$(basename "$YAML_FILE" .yaml)
LOG_FILE="${BASENAME}_diagnostics.log"
NUM_WORKERS=0

# Common command parts
PYTHON_CMD="python -u -W ignore -m"
WORKER_ARG="--number_of_workers $NUM_WORKERS"

{
  module load conda
  conda activate /glade/work/pavelp/conda-envs/env-from-npl-2024a

  # Time series
  $PYTHON_CMD mom6_tools.stats "$YAML_FILE" -time_series $WORKER_ARG

  ## Poleward heat transport (global, Atlantic, Pacific)
  #$PYTHON_CMD mom6_tools.poleward_heat_transport "$YAML_FILE" $WORKER_ARG

  ## MOC, AMOC
  #$PYTHON_CMD mom6_tools.moc "$YAML_FILE" $WORKER_ARG

  ## MOC, AMOC in sigma coordinates
  #$PYTHON_CMD mom6_tools.moc_sigma2 "$YAML_FILE" $WORKER_ARG

  ## Equatorial jets and stratification
  #$PYTHON_CMD mom6_tools.equatorial_comparison "$YAML_FILE" $WORKER_ARG

  ## Mixed Layer Depth
  #$PYTHON_CMD mom6_tools.surface "$YAML_FILE" $WORKER_ARG

  ## T&S horizontal biases
  #$PYTHON_CMD mom6_tools.TS_levels "$YAML_FILE" $WORKER_ARG

  ## Sections transports, i.e. Drake Passage, Bering Strait, etc.
  #$PYTHON_CMD mom6_tools.section_transports "$YAML_FILE" $WORKER_ARG -save_ncfile

  ## Temperature drift
  #$PYTHON_CMD mom6_tools.drift "$YAML_FILE" thetao $WORKER_ARG --rms --drift --savefig

  ## Salinity drift
  #$PYTHON_CMD mom6_tools.drift "$YAML_FILE" so $WORKER_ARG --rms --drift --savefig
} &> "$LOG_FILE"