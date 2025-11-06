#!/bin/bash
#=========================================================================================
# TSAD Benchmarking Script
#
# This script runs a specific set of TSPulse ZS algorithms for standard benchmarks
# and for a multi-as-uni evaluation.
#
# Usage:
#   export TASK_ID=1  # Optional: specify task ID (1-24), defaults to 1
#   bash _c_reproduce_tspulse.sh
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
FT_ALGOS=(
    "TSPulse_FT_ensemble"
    "TSPulse_FT_fft"
    "TSPulse_FT_forecast"
    "TSPulse_FT_time"
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
for run_type in "${STANDARD_RUN_TYPES[@]}"; do
    for algo in "${FT_ALGOS[@]}"; do
        PARAMS+=("$run_type $algo")
    done
done
# Add multi-as-uni runs
for algo in "${STANDARD_ALGOS[@]}"; do
    PARAMS+=("MU $algo") # MU for Multi-as-Uni
done
for algo in "${FT_ALGOS[@]}"; do
    PARAMS+=("MU $algo") # MU for Multi-as-Uni
done


# 3. Task-Specific Setup
# Define Project Root - use current directory or parent if in MAD subdirectory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$SCRIPT_DIR")" == "MAD" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

# Get task ID from environment variable or argument, default to 1
TASK_ID="${TASK_ID:-${1:-1}}"
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 24 ]; then
    echo "ERROR: TASK_ID must be between 1 and 24 (got: $TASK_ID)"
    exit 1
fi

# Each task will select a unique parameter set.
TASK_INDEX=$((TASK_ID - 1))
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
echo "--- JOB TASK START ---"
echo "Task ID: $TASK_ID / 24"
echo "Host: $(hostname)"
echo "Project Root: $PROJECT_ROOT"
echo "----------------------------------"

# Add the project's root and granite-tsfm to Python's path.
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm:${PROJECT_ROOT}/granite-tsfm/notebooks/hfdemo/tspulse/anomaly_detection:${PYTHONPATH}"

# Create log directory in the project root.
mkdir -p "${PROJECT_ROOT}/logs"

# Change to the project root directory to run the script.
cd "${PROJECT_ROOT}"

# Activate conda environment if available (optional)
# Uncomment and modify if you need to use a specific conda environment:
# conda activate tsb-ad-env
# Or use system python if no conda environment is needed

# 5. Execution
echo "=========================================================="
echo "Running Task ${TASK_ID} with parameters:"
echo "  Run Type: $CURRENT_RUN_TYPE"
echo "  Algorithm: $CURRENT_AD_NAME"
echo "  Script: $RUN_SCRIPT"
echo "  Eval File: $EVAL_FILE"
echo "  Score Dir: $SCORE_DIR"
echo "  Save Dir: $SAVE_DIR"
echo "=========================================================="
# Command with explicit paths for all run types
COMMAND="python3 -u \"${RUN_SCRIPT}\" \
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
    echo "!!! Command failed for task $TASK_ID. !!!"
    exit 1
fi

echo "=========================================================="
echo "Task $TASK_ID finished successfully at: $(date)"
echo "--- JOB TASK END ---" 