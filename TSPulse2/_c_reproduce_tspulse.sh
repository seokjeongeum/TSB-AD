#!/bin/bash
#=========================================================================================
# Slurm SBATCH Directives for TSAD Benchmarking
#
# This script runs a specific set of TSPulse ZS algorithms for standard benchmarks
# and for a multi-as-uni evaluation. It uses a Slurm job array to execute
# a job for each algorithm-dataset-type combination.
#=========================================================================================
#
#SBATCH --job-name=TSAD_TSPULSE_COMBO
#SBATCH --output=slurm_logs/%A_%x_%a.out      # Unique log for each task
#SBATCH --error=slurm_logs/%A_%x_%a.err       # Unique error log for each task

# --- Resource Allocation ---
# Adjust partition, QoS, and time as needed for your environment.
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

# --- Job Array Configuration ---
#SBATCH --array=1-12

#=========================================================================================
# HYPERPARAMETER & LOGIC BLOCK
#=========================================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Define the Algorithm and Dataset Lists
STANDARD_ALGOS=(
    "TSPulse_ZS_ensemble"
    "TSPulse_ZS_fft"
    "TSPulse_ZS_forecast"
    "TSPulse_ZS_time"
)
STANDARD_RUN_TYPES=("U" "M")

# 2. Define the Parameter Grid
declare -a PARAMS
# Add standard uni and multi runs
for run_type in "${STANDARD_RUN_TYPES[@]}"; do
    for algo in "${STANDARD_ALGOS[@]}"; do
        PARAMS+=("$run_type $algo")
    done
done
# Add multi-as-uni runs
for algo in "${STANDARD_ALGOS[@]}"; do
    PARAMS+=("MU $algo") # MU for Multi-as-Uni
done


# 3. Task-Specific Setup
# Define Project Root from the submission directory first.
# This script assumes you run `sbatch` from the top-level project directory.
PROJECT_ROOT="$SLURM_SUBMIT_DIR"

# Each task in the array will select a unique parameter set.
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
read -r -a CURRENT_PARAMS <<< "${PARAMS[$TASK_INDEX]}"

CURRENT_RUN_TYPE=${CURRENT_PARAMS[0]}
CURRENT_AD_NAME=${CURRENT_PARAMS[1]}


# Construct paths and script name based on the run type
if [ "$CURRENT_RUN_TYPE" == "U" ]; then
    RUN_SCRIPT="benchmark_exp/Run_Detector_U.py"
    DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-U/"
    EVAL_FILE="${PROJECT_ROOT}/Datasets/File_List/TSB-AD-U.csv"
    SCORE_DIR="${PROJECT_ROOT}/eval/score/uni/"
    SAVE_DIR="${PROJECT_ROOT}/eval/metrics/uni/"
elif [ "$CURRENT_RUN_TYPE" == "M" ]; then
    RUN_SCRIPT="benchmark_exp/Run_Detector_M.py"
    DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-M/"
    EVAL_FILE="${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv"
    SCORE_DIR="${PROJECT_ROOT}/eval/score/multi/"
    SAVE_DIR="${PROJECT_ROOT}/eval/metrics/multi/"
elif [ "$CURRENT_RUN_TYPE" == "MU" ]; then
    # This block is based on _d_generate_ground_truth_multi_as_uni.sh
    RUN_SCRIPT="benchmark_exp/Run_Detector_U.py"
    DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-M/"
    EVAL_FILE="${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M-univariate.csv"
    SCORE_DIR="${PROJECT_ROOT}/eval/score/multi_as_uni/"
    SAVE_DIR="${PROJECT_ROOT}/eval/metrics/multi_as_uni/"
fi

# 4. Environment Setup
echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Host: $(hostname)"
echo "----------------------------------"

# Add the project's root and granite-tsfm to Python's path.
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm:${PROJECT_ROOT}/granite-tsfm/notebooks/hfdemo/tspulse/anomaly_detection:${PYTHONPATH}"

# Create slurm_logs in the project root.
mkdir -p "${PROJECT_ROOT}/slurm_logs"

# Change to the project root directory to run the script.
cd "${PROJECT_ROOT}"

# Activate your python environment.
# Make sure the 'tsb-ad-env' conda environment is properly set up.
module load cuda/11.8 conda-py39
conda activate tsb-ad-env

# 5. Execution
echo "=========================================================="
echo "Running Task ${SLURM_ARRAY_TASK_ID} with parameters:"
echo "  Run Type: $CURRENT_RUN_TYPE"
echo "  Algorithm: $CURRENT_AD_NAME"
echo "  Script: $RUN_SCRIPT"
echo "  Eval File: $EVAL_FILE"
echo "  Score Dir: $SCORE_DIR"
echo "  Save Dir: $SAVE_DIR"
echo "=========================================================="
# Command with explicit paths for all run types
COMMAND="python -u \"${RUN_SCRIPT}\" \
  --AD_Name=\"${CURRENT_AD_NAME}\" \
  --dataset_dir=\"${DATA_DIR}\" \
  --file_lsit=\"${EVAL_FILE}\" \
  --score_dir=\"${SCORE_DIR}\" \
  --save_dir=\"${SAVE_DIR}\" \
  --save True"

echo "Executing command: $COMMAND"
eval "$COMMAND"

# Check the exit code of the script
if [ $? -ne 0 ]; then
    echo "!!! Command failed for task $SLURM_ARRAY_TASK_ID. !!!"
    exit 1
fi

echo "=========================================================="
echo "Job Task $SLURM_ARRAY_TASK_ID finished successfully at: $(date)"
echo "--- SLURM JOB ARRAY TASK END ---" 