#!/bin/bash
#=========================================================================================
# Slurm SBATCH Script for a Single Run of Run_Detector_M.py
#
# This script executes a single instance of the 'Run_Detector_M.py' script
# to run the TSPulse2 model on the multivariate dataset.
#=========================================================================================

#-----------------------------------------------------------------------------------------
# SBATCH Directives
#-----------------------------------------------------------------------------------------
# -- Job Details --
#SBATCH --job-name=Run_TSPulse2_M
#SBATCH --output=slurm_logs/%j_%x.out      # Log file: jobname_jobid.out
#SBATCH --error=slurm_logs/%j_%x.err       # Error file: jobname_jobid.err

# -- Resource Allocation --
#SBATCH --partition=A100-80GB              # Specify the partition (e.g., A100-80GB)
#SBATCH --qos=hpgpu                    # Quality of Service (use 'hpgpu' or 'add_hpgpu' as needed)
#SBATCH --time=3-00:00:00                  # Max runtime: 1 day (Adjusted to avoid QOS limit issues)
#SBATCH --gres=gpu:1                       # Request 1 GPU

#=========================================================================================
# Environment Setup
#=========================================================================================
set -e # Exit immediately if a command exits with a non-zero status.

# --- Project and Path Configuration ---
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
# The Python script uses a relative path to find the TSB_AD module,
# so setting PYTHONPATH is good practice but may not be strictly necessary
# if the script is run from the correct directory.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# --- Logging and Diagnostics ---
echo "--- SLURM JOB START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start Time: $(date)"
echo "Project Directory: $PROJECT_ROOT"
echo "--------------------------"

# Create log directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/slurm_logs"

# --- Module and Environment Loading ---
echo "Loading modules..."
module load cuda/11.8 conda-py39

echo "Activating Conda environment..."
conda activate tsb-ad-env

# Navigate to the project root directory
cd "$PROJECT_ROOT"
echo "Current Directory: $(pwd)"

#=========================================================================================
# Python Script Execution
#=========================================================================================

# --- Define Parameters for the Python Script ---
# These arguments will be passed to Run_Detector_M.py.
# Note: These paths assume the script is run from the project root.
RUN_SCRIPT_PATH="benchmark_exp/Run_Detector_M.py" # Assumes the script is in the root
DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-M/"
# The Python script was modified to use 'TSPulse2-M-Eva.csv' as the default
FILE_LIST="${PROJECT_ROOT}/Datasets/File_List/TSPulse2-M-Eva.csv"
AD_MODEL_NAME="TSPulse2"

# Define output directories to keep results organized
SCORE_DIR="${PROJECT_ROOT}/eval/score/multi/"
SAVE_DIR="${PROJECT_ROOT}/eval/metrics/multi/"

echo "=========================================================="
echo "Running Python Script: ${RUN_SCRIPT_PATH}"
echo "Parameters:"
echo "  Anomaly Detector: ${AD_MODEL_NAME}"
echo "  Dataset Directory: ${DATA_DIR}"
echo "  File List: ${FILE_LIST}"
echo "  Score Directory: ${SCORE_DIR}"
echo "  Metrics Directory: ${SAVE_DIR}"
echo "=========================================================="

# --- Construct and Execute the Command ---
# Using '-u' for unbuffered output to see logs in real-time.
# All arguments are explicitly passed to the script.
COMMAND="python -u ${RUN_SCRIPT_PATH} \
  --AD_Name=\"${AD_MODEL_NAME}\" \
  --dataset_dir=\"${DATA_DIR}\" \
  --file_lsit=\"${FILE_LIST}\" \
  --score_dir=\"${SCORE_DIR}\" \
  --save_dir=\"${SAVE_DIR}\" \
  --save True"

echo "Executing command: ${COMMAND}"
eval "$COMMAND"

# --- Final Checks and Cleanup ---
# Check the exit code of the Python script
if [ $? -ne 0 ]; then
    echo "!!! Python script failed with a non-zero exit code. !!!"
    exit 1
fi

echo "=========================================================="
echo "Job finished successfully at: $(date)"
echo "--- SLURM JOB END ---" 