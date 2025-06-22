#!/bin/bash

#=========================================================================================
# Slurm SBATCH Directives for a SINGLE Job (Master script sets these)
#=========================================================================================
# Note: The #SBATCH directives in this file are IGNORED when submitted by sbatch
# from the master script, but are useful for documentation.
#
# #SBATCH --job-name=... (Set by master script)
# #SBATCH --output=...   (Set by master script)
# #SBATCH --error=...    (Set by master script)
#
# #SBATCH --partition=A100-80GB
# #SBATCH --qos=hpgpu
# #SBATCH --gres=gpu:1
# #SBATCH --time=3-00:00:00
#
#=========================================================================================
# This script expects the following environment variables to be set by the master script:
# - TASK_ID
# - CURRENT_AD_NAME
# - CURRENT_SCRIPT
# - CURRENT_DATA_DIR
# - CURRENT_FILE_LIST
# - CURRENT_EXTRA_ARGS
# - CURRENT_AGG_MODE
#=========================================================================================

PROJECT_ROOT="/home/seokjeongeum/TSB-AD"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"

#=========================================================================================
# Execution
#=========================================================================================
echo "=========================================================="
echo "Starting SLURM Job: $SLURM_JOB_ID, Task defined by master script as: $TASK_ID"
echo "Host: $(hostname)"
echo "Running with Parameters: AD_Name=$CURRENT_AD_NAME, Script=$CURRENT_SCRIPT, Aggregation Mode: $CURRENT_AGG_MODE"
echo "=========================================================="

# Load modules and activate environment
module load cuda/11.8 conda-py39
conda activate tsb-ad-env

# Change to project directory
cd "$PROJECT_ROOT"

# Construct and run the python command for the benchmark
COMMAND="python -u \"${PROJECT_ROOT}/benchmark_exp/${CURRENT_SCRIPT}\" \
  --AD_Name=\"${CURRENT_AD_NAME}\" \
  --dataset_dir=\"${CURRENT_DATA_DIR}\" \
  --file_lsit=\"${CURRENT_FILE_LIST}\" \
  --save True \
  ${CURRENT_EXTRA_ARGS}"

echo "Executing benchmark command: $COMMAND"
eval "$COMMAND"

#=========================================================================================
# Score Aggregation
#=========================================================================================
echo "----------------------------------------------------------"
echo "Benchmark finished. Now running the corresponding score aggregation."
echo "----------------------------------------------------------"

AGG_COMMAND="python TSPulse2/read_scores.py \"$CURRENT_AD_NAME\" \"$CURRENT_AGG_MODE\""

echo "Executing aggregation command: $AGG_COMMAND"
eval "$AGG_COMMAND"

echo "=========================================================="
echo "Job $SLURM_JOB_ID fully finished at: $(date)"
echo "=========================================================="
