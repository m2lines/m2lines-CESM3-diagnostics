#!/bin/bash

# Usage: ./run_diagnostics.sh experiment.yaml

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

  $PYTHON_CMD mom6_tools.stats "$YAML_FILE" -time_series $WORKER_ARG
  #$PYTHON_CMD mom6_tools.poleward_heat_transport "$YAML_FILE" $WORKER_ARG
  #$PYTHON_CMD mom6_tools.moc "$YAML_FILE" $WORKER_ARG
  #$PYTHON_CMD mom6_tools.moc_sigma2 "$YAML_FILE" $WORKER_ARG

} &> "$LOG_FILE"
