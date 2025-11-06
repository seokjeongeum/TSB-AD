#!/bin/bash
#=========================================================================================
# Task Mapping:
# 1-6: MULTI Models
#   1: MAD (Main)
#   2: MAD (Dimensionality Reduction Ablation)
#   3: MAD (LLM Selection Ablated - ensemble)
#   4: MAD (LLM Selection Ablated - fft)
#   5: MAD (LLM Selection Ablated - forecast)
#   6: MAD (LLM Selection Ablated - time)
# 7: MAD (Main)
# 8: MAD (Forecast Biased)
# 9: MAD (Non-Forecast Biased)
# 10: MAD (Dimensionality Reduction Ablated)
# 11: MAD (Forecast Biased - Dimensionality Reduction Ablated)
# 12: MAD (Non-Forecast Biased - Dimensionality Reduction Ablated)
# 13: UNI - MAD (Main)
#
# Usage:
#   export TASK_ID=1  # Optional: specify task ID (1-13), defaults to 1
#   bash _d_full_experiment.sh
#=========================================================================================
#=========================================================================================

#=========================================================================================
# Environment Setup
#=========================================================================================
set -e # Exit immediately if a command exits with a non-zero status.

# --- Project and Path Configuration ---
# Define Project Root - use current directory or parent if in MAD subdirectory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$SCRIPT_DIR")" == "MAD" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

# The Python script uses a relative path to find the TSB_AD module,
# so setting PYTHONPATH is good practice but may not be strictly necessary
# if the script is run from the correct directory.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Get task ID from environment variable or argument, default to 1
TASK_ID="${TASK_ID:-${1:-1}}"
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 13 ]; then
    echo "ERROR: TASK_ID must be between 1 and 13 (got: $TASK_ID)"
    exit 1
fi

# --- Logging and Diagnostics ---
echo "--- JOB TASK START ---"
echo "Task ID: $TASK_ID / 13"
echo "Host: $(hostname)"
echo "Start Time: $(date)"
echo "Project Directory: $PROJECT_ROOT"
echo "--------------------------"

#=========================================================================================
# API Key Configuration
#=========================================================================================
# Add your Gemini API keys to this array. The script will assign them to tasks
# in a round-robin fashion (e.g., Task 1 gets Key 1, Task 2 gets Key 2, etc.).
# If there are more tasks than keys, the keys will be reused.
# If empty, the script will use TSPulse_GEMINI_API_KEY from environment if available.
API_KEYS=(
    ""
)

# --- API Key Selection for Current Task ---
if [ ${#API_KEYS[@]} -eq 0 ] || [ -z "${API_KEYS[0]}" ]; then
    # Use environment variable if available
    if [ -n "$TSPulse_GEMINI_API_KEY" ]; then
        echo "Using TSPulse_GEMINI_API_KEY from environment for Task ${TASK_ID}."
    else
        echo "WARNING: No API keys configured. Using empty key (may fail if API key is required)."
        export TSPulse_GEMINI_API_KEY=""
    fi
else
    # Check if enough keys were provided for all tasks
    if [ "${#API_KEYS[@]}" -lt 13 ]; then
        echo "--- WARNING: Not enough API keys provided for all tasks. Keys will be reused."
        echo "    Provided: ${#API_KEYS[@]}, Required: 13"
    fi
    
    # Select an API key for the current task using modular arithmetic for round-robin assignment
    TASK_INDEX=$(( (TASK_ID - 1) % ${#API_KEYS[@]} ))
    SELECTED_API_KEY=${API_KEYS[$TASK_INDEX]}
    
    # Export the selected key for the Python script.
    # The Python script is configured to look for 'TSPulse_GEMINI_API_KEY'.
    export TSPulse_GEMINI_API_KEY=$SELECTED_API_KEY
    echo "API Key has been selected for Task ${TASK_ID}."
fi


# Create log directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Activate conda environment if available (optional)
# Uncomment and modify if you need to use a specific conda environment:
# conda activate tsb-ad-env
# Or use system python if no conda environment is needed

# Navigate to the project root directory
cd "$PROJECT_ROOT"
echo "Current Directory: $(pwd)"

#=========================================================================================
# Experiment Configuration
#=========================================================================================

# --- Multivariate Configurations ---
MULTI_FILE_LISTS=(
    "${PROJECT_ROOT}/Datasets/File_List/MAD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/MAD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/MAD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/MAD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/MAD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/MAD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/MAD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/MAD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv"
    "${PROJECT_ROOT}/Datasets/File_List/TSB-AD-M.csv"
)
MULTI_MODELS=(
    "MAD"
    "MAD_dimensionality_reduction_ablated"
    "MAD_llm_selection_ablated_ensemble"
    "MAD_llm_selection_ablated_fft"
    "MAD_llm_selection_ablated_forecast"
    "MAD_llm_selection_ablated_time"
    "MAD"
    "MAD_forecast_biased"
    "MAD_non_forecast_biased"
    "MAD_dim_redux_ablated"
    "MAD_forecast_biased_dim_redux_ablated"
    "MAD_non_forecast_biased_dim_redux_ablated"
)
NUM_MULTI_JOBS=$((${#MULTI_MODELS[@]}))

# --- Univariate Configurations ---
# No "channel_selection" model for univariate, as it's not applicable.
UNI_MODELS=(
    "MAD"
)
UNI_FILE_LIST="${PROJECT_ROOT}/Datasets/File_List/TSB-AD-U.csv"

NUM_UNI_JOBS=$((${#UNI_MODELS[@]}))

# --- Task-specific Parameter Selection ---
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
echo "Running Task ${TASK_ID} with Parameters:"
echo "  Anomaly Detector: ${AD_MODEL_NAME}"
echo "  Run Script: ${RUN_SCRIPT_PATH}"
echo "  Dataset Directory: ${DATA_DIR}"
echo "  File List: ${CURRENT_FILE_LIST}"
echo "=========================================================="

# --- Construct and Execute the Command ---
# Using '-u' for unbuffered output to see logs in real-time.
# All arguments are explicitly passed to the script.
python3 -u "${RUN_SCRIPT_PATH}" \
  --AD_Name="${AD_MODEL_NAME}" \
  --dataset_dir="${DATA_DIR}" \
  --file_lsit="${CURRENT_FILE_LIST}" \
  --save True

# Check the exit code of the Python script
if [ $? -ne 0 ]; then
    echo "!!! Python script failed with a non-zero exit code for task ${TASK_ID}. !!!"
    exit 1
fi

echo "=========================================================="
echo "Task $TASK_ID finished successfully at: $(date)"
echo "--- JOB END ---" 
