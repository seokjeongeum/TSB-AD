#!/bin/bash

#=========================================================================================
# Slurm SBATCH Directives
#=========================================================================================
#
#SBATCH --job-name=TSPulse_MultiAsUni
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-5 # One task for each model

#=========================================================================================
# Shared Setup
#=========================================================================================
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"

echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Host: $(hostname)"
echo "----------------------------------"

mkdir -p "${PROJECT_ROOT}/slurm_logs"

# --- Experiment Configuration ---
# Each model is run on the same set of generated univariate data.
AD_NAMES=(
    "TSPulse2"
    "TSPulse_ZS_ensemble"
    "TSPulse_ZS_time"
    "TSPulse_ZS_fft"
    "TSPulse_ZS_future"
)

# --- Static Parameters for all tasks ---
# All jobs will run the univariate detector script on the converted data.
RUN_SCRIPT="Run_Detector_U.py"
DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-M-univariate/"
FILE_LIST="${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M-univariate.csv"

# --- NEW: Output Directory Configuration ---
# Define distinct directories for this specific experiment's outputs.
# This prevents overwriting results from other benchmark runs.
SCORE_DIR="${PROJECT_ROOT}/eval/score/multi_as_uni/"
SAVE_DIR="${PROJECT_ROOT}/eval/metrics/multi_as_uni/"

#=========================================================================================
# Task-Specific Logic
#=========================================================================================
# Get the index for the model name from the SLURM task ID
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Select the model for the current task
CURRENT_AD_NAME=${AD_NAMES[$TASK_INDEX]}

# Load modules and activate environment
module load cuda/11.8 conda-py39
conda activate tsb-ad-env

cd "$PROJECT_ROOT"

#=========================================================================================
# Execution
#=========================================================================================
echo "=========================================================="
echo "Running Task with Parameters:"
echo "AD_Name: $CURRENT_AD_NAME"
echo "Script: $RUN_SCRIPT"
echo "Data Dir: $DATA_DIR"
echo "File List: $FILE_LIST"
echo "Score Dir: $SCORE_DIR"
echo "Save Dir: $SAVE_DIR"
echo "=========================================================="

# Construct and run the python command for the benchmark
# Added --score_dir and --save_dir arguments.
COMMAND="python -u \"${PROJECT_ROOT}/benchmark_exp/${RUN_SCRIPT}\" \
  --AD_Name=\"${CURRENT_AD_NAME}\" \
  --dataset_dir=\"${DATA_DIR}\" \
  --file_lsit=\"${FILE_LIST}\" \
  --score_dir=\"${SCORE_DIR}\" \
  --save_dir=\"${SAVE_DIR}\" \
  --save True"

echo "Executing command: $COMMAND"
eval "$COMMAND"

# Check the exit code of the benchmark
if [ $? -ne 0 ]; then
    echo "!!! Benchmark command failed for task $SLURM_ARRAY_TASK_ID. !!!"
    exit 1
fi

echo "=========================================================="
echo "Job Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "--- SLURM JOB ARRAY TASK END ---"