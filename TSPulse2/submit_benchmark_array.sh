#!/bin/bash

#=========================================================================================
# Slurm SBATCH Directives for a Simplified Job Array
#=========================================================================================
#
#SBATCH --job-name=TSPulse_Benchmark
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-10 # Total tasks: 5 models x 2 domains (M/U)

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

# --- Simplified Experiment Configuration ---
# Each model is run on Multivariate and Univariate datasets.
AD_NAMES=(
    "TSPulse2" "TSPulse2"
    "TSPulse_ZS_ensemble" "TSPulse_ZS_ensemble"
    "TSPulse_ZS_time" "TSPulse_ZS_time"
    "TSPulse_ZS_fft" "TSPulse_ZS_fft"
    "TSPulse_ZS_future" "TSPulse_ZS_future"
)
RUN_SCRIPTS=(
    "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_U.py"
)
DATASET_DIRS=(
    "${PROJECT_ROOT}/Datasets/TSB-AD-M/" "${PROJECT_ROOT}/Datasets/TSB-AD-U/"
    "${PROJECT_ROOT}/Datasets/TSB-AD-M/" "${PROJECT_ROOT}/Datasets/TSB-AD-U/"
    "${PROJECT_ROOT}/Datasets/TSB-AD-M/" "${PROJECT_ROOT}/Datasets/TSB-AD-U/"
    "${PROJECT_ROOT}/Datasets/TSB-AD-M/" "${PROJECT_ROOT}/Datasets/TSB-AD-U/"
    "${PROJECT_ROOT}/Datasets/TSB-AD-M/" "${PROJECT_ROOT}/Datasets/TSB-AD-U/"
)
FILE_LISTS=(
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv" "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-U.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv" "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-U.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv" "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-U.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv" "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-U.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv" "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-U.csv"
)

#=========================================================================================
# Task-Specific Logic
#=========================================================================================
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

CURRENT_AD_NAME=${AD_NAMES[$TASK_INDEX]}
CURRENT_SCRIPT=${RUN_SCRIPTS[$TASK_INDEX]}
CURRENT_DATA_DIR=${DATASET_DIRS[$TASK_INDEX]}
CURRENT_FILE_LIST=${FILE_LISTS[$TASK_INDEX]}

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
echo "Script: $CURRENT_SCRIPT"
echo "Data Dir: $CURRENT_DATA_DIR"
echo "File List: $CURRENT_FILE_LIST"
echo "=========================================================="

# Construct and run the python command for the benchmark
# No more EXTRA_ARGS needed.
COMMAND="python -u \"${PROJECT_ROOT}/benchmark_exp/${CURRENT_SCRIPT}\" \
  --AD_Name=\"${CURRENT_AD_NAME}\" \
  --dataset_dir=\"${CURRENT_DATA_DIR}\" \
  --file_lsit=\"${CURRENT_FILE_LIST}\" \
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