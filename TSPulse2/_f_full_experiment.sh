#!/bin/bash
#=========================================================================================
# Slurm SBATCH Script for Evaluating TSPulse2 on Different Eva Sets
#
# This script uses a Slurm job array to run the TSPulse2 model against
# two different evaluation file lists: TSPulse2-M-Eva.csv and TSB-AD-M-Eva.csv.
#=========================================================================================

#-----------------------------------------------------------------------------------------
# SBATCH Directives
#-----------------------------------------------------------------------------------------
# -- Job Details --
#SBATCH --job-name=TSPulse2_Eva_Runs
#SBATCH --output=slurm_logs/%A_%x_%a.out      # Log file: jobid_jobname_taskid.out
#SBATCH --error=slurm_logs/%A_%x_%a.err       # Error file: jobid_jobname_taskid.err

# -- Resource Allocation --
#SBATCH --partition=A100-80GB              # Specify the partition (e.g., A100-80GB)
#SBATCH --qos=hpgpu                    # Quality of Service (use 'hpgpu' or 'add_hpgpu' as needed)
#SBATCH --time=3-00:00:00                  # Max runtime
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --array=1-16

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
echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
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
# Experiment Configuration
#=========================================================================================

# --- Multivariate Configurations ---
MULTI_FILE_LISTS=(
    "${PROJECT_ROOT}/Datasets/File_List/TSPulse2-M-Eva.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M-Eva_1.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M-Eva_2.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M-Eva_3.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M-Eva_4.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M-Eva_5.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M-Eva_6.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSPulse2-M-Eva.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSPulse2-M-Eva.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSPulse2-M-Eva.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSPulse2-M-Eva.csv"
)
MULTI_MODELS=(
    "TSPulse2" 
    "TSPulse2_dimensionality_reduction_ablated"
    "TSPulse2_dimensionality_reduction_ablated"
    "TSPulse2_dimensionality_reduction_ablated"
    "TSPulse2_dimensionality_reduction_ablated"
    "TSPulse2_dimensionality_reduction_ablated"
    "TSPulse2_dimensionality_reduction_ablated"
    "TSPulse2_llm_selection_ablated_ensemble"
    "TSPulse2_llm_selection_ablated_fft"
    "TSPulse2_llm_selection_ablated_forecast"
    "TSPulse2_llm_selection_ablated_time"
)
NUM_MULTI_JOBS=$((${#MULTI_MODELS[@]} ))

# --- Univariate Configurations ---
# No "channel_selection" model for univariate, as it's not applicable.
UNI_MODELS=(
    "TSPulse2" 
    "TSPulse2_llm_selection_ablated_ensemble"
    "TSPulse2_llm_selection_ablated_fft"
    "TSPulse2_llm_selection_ablated_forecast"
    "TSPulse2_llm_selection_ablated_time"
)
UNI_FILE_LIST="${PROJECT_ROOT}/Datasets/File_List/TSB-AD-U-Eva.csv"

NUM_UNI_JOBS=$((${#UNI_MODELS[@]} ))

# --- Task-specific Parameter Selection ---
TASK_ID=$SLURM_ARRAY_TASK_ID
if [ "$TASK_ID" -le "$NUM_MULTI_JOBS" ]; then
    # This is a MULTIVARIATE task
    TASK_INDEX=$((TASK_ID - 1))
    MODEL_INDEX=$((TASK_INDEX ))
    FILE_LIST_INDEX=$((TASK_INDEX ))

    AD_MODEL_NAME=${MULTI_MODELS[$MODEL_INDEX]}
    CURRENT_FILE_LIST=${MULTI_FILE_LISTS[$FILE_LIST_INDEX]}
    RUN_SCRIPT_PATH="benchmark_exp/Run_Detector_M.py"
    DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-M/"

else
    # This is a UNIVARIATE task
    TASK_INDEX=$((TASK_ID - NUM_MULTI_JOBS - 1))
    MODEL_INDEX=$((TASK_INDEX ))
    FILE_LIST_INDEX=$((TASK_INDEX ))

    AD_MODEL_NAME=${UNI_MODELS[$MODEL_INDEX]}
    CURRENT_FILE_LIST=${UNI_FILE_LIST}
    RUN_SCRIPT_PATH="benchmark_exp/Run_Detector_U.py"
    DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-U/"

fi

#=========================================================================================
# Python Script Execution
#=========================================================================================
echo "=========================================================="
echo "Running Task ${SLURM_ARRAY_TASK_ID} with Parameters:"
echo "  Anomaly Detector: ${AD_MODEL_NAME}"
echo "  Run Script: ${RUN_SCRIPT_PATH}"
echo "  Dataset Directory: ${DATA_DIR}"
echo "  File List: ${CURRENT_FILE_LIST}"
echo "=========================================================="

# --- Construct and Execute the Command ---
# Using '-u' for unbuffered output to see logs in real-time.
# All arguments are explicitly passed to the script.
COMMAND="python -u ${RUN_SCRIPT_PATH} \
  --AD_Name=\"${AD_MODEL_NAME}\" \
  --dataset_dir=\"${DATA_DIR}\" \
  --file_lsit=\"${CURRENT_FILE_LIST}\" \
  --save True"

echo "Executing command: ${COMMAND}"
eval "$COMMAND"

# --- Final Checks and Cleanup ---
# Check the exit code of the Python script
if [ $? -ne 0 ]; then
    echo "!!! Python script failed with a non-zero exit code for task ${SLURM_ARRAY_TASK_ID}. !!!"
    exit 1
fi

echo "=========================================================="
echo "Job Task $SLURM_ARRAY_TASK_ID finished successfully at: $(date)"
echo "--- SLURM JOB END ---" 