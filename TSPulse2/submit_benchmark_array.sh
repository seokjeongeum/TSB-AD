#!/bin/bash

#=========================================================================================
# Slurm SBATCH Directives for a Job Array
#
# This single script defines the parameters for all tasks and executes the
# correct task based on the SLURM_ARRAY_TASK_ID.
#=========================================================================================
#
#SBATCH --job-name=TSPulse_Benchmark      # A general name for the array job
#SBATCH --output=slurm_logs/%x_%A_%a.out  # Unique log for each task (%x=job-name, %A=job-id, %a=task-id)
#SBATCH --error=slurm_logs/%x_%A_%a.err   # Unique error log for each task
#
#SBATCH --partition=A100-80GB             # The GPU partition you want to use
#SBATCH --qos=hpgpu                       # Requesting permission for the partition
#SBATCH --gres=gpu:1                      # Request 1 GPU *per task*
#SBATCH --time=3-00:00:00                 # Max wall time (3 days)
#
#SBATCH --array=1-20
#
#=========================================================================================
# Shared Setup (This part runs for every task in the array)
#=========================================================================================
# Load secrets from a secure, non-version-controlled file
if [ -f ~/.secrets ]; then
    echo "Loading secrets..."
    source ~/.secrets
else
    echo "ERROR: Secrets file not found!"
    exit 1
fi

echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "----------------------------------"

# Ensure the log directory exists before any task tries to write to it.
# This part is safe to run for every task.
mkdir -p slurm_logs

# These arrays must have the same number of elements as the --array count.
AD_NAMES=(
    "TSPulse2_0623" "TSPulse2_0623" "TSPulse2_0623" "TSPulse2_0623"
    "TSPulse_ZS_ensemble" "TSPulse_ZS_ensemble" "TSPulse_ZS_ensemble" "TSPulse_ZS_ensemble"
    "TSPulse_ZS_time" "TSPulse_ZS_time" "TSPulse_ZS_time" "TSPulse_ZS_time"
    "TSPulse_ZS_fft" "TSPulse_ZS_fft" "TSPulse_ZS_fft" "TSPulse_ZS_fft"
    "TSPulse_ZS_future" "TSPulse_ZS_future" "TSPulse_ZS_future" "TSPulse_ZS_future"
)
RUN_SCRIPTS=(
    "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_M.py" "Run_Detector_U.py"
)
DATASET_DIRS=(
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/"
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/"
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/"
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/"
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/"
)
FILE_LISTS=(
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
)
EXTRA_ARGS=(
    "" "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
    "" "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
    "" "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
    "" "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
    "" "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
)
AGGREGATION_MODES=(
    "multi" "uni" "multi-tuning" "uni-tuning"
    "multi" "uni" "multi-tuning" "uni-tuning"
    "multi" "uni" "multi-tuning" "uni-tuning"
    "multi" "uni" "multi-tuning" "uni-tuning"
    "multi" "uni" "multi-tuning" "uni-tuning"
)

#=========================================================================================
# Task-Specific Logic
#=========================================================================================
# SLURM_ARRAY_TASK_ID is 1-based, but our arrays are 0-based.
# We subtract 1 to get the correct index for our parameter arrays.
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Get parameters for the current task using the calculated index
CURRENT_AD_NAME=${AD_NAMES[$TASK_INDEX]}
CURRENT_SCRIPT=${RUN_SCRIPTS[$TASK_INDEX]}
CURRENT_DATA_DIR=${DATASET_DIRS[$TASK_INDEX]}
CURRENT_FILE_LIST=${FILE_LISTS[$TASK_INDEX]}
CURRENT_EXTRA_ARGS=${EXTRA_ARGS[$TASK_INDEX]}
CURRENT_AGG_MODE=${AGGREGATION_MODES[$TASK_INDEX]}

# --- Environment Setup (Merged from worker_script.sh) ---
# --- FIX: Use the reliable $SLURM_SUBMIT_DIR instead of a relative path ---
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"

# Load modules and activate environment
module load cuda/11.8 conda-py39
conda activate tsb-ad-env

# Change to project directory
cd "$PROJECT_ROOT"

#=========================================================================================
# Execution (Merged from worker_script.sh)
#=========================================================================================
echo "=========================================================="
echo "Running Task with Parameters:"
echo "AD_Name: $CURRENT_AD_NAME"
echo "Script: $CURRENT_SCRIPT"
echo "Aggregation Mode: $CURRENT_AGG_MODE"
echo "=========================================================="

# Construct and run the python command for the benchmark
# NOTE: The fix on --file_list (from lsit)
COMMAND="python -u \"${PROJECT_ROOT}/benchmark_exp/${CURRENT_SCRIPT}\" \
  --AD_Name=\"${CURRENT_AD_NAME}\" \
  --dataset_dir=\"${CURRENT_DATA_DIR}\" \
  --file_list=\"${CURRENT_FILE_LIST}\" \
  --save True \
  ${CURRENT_EXTRA_ARGS}"

echo "Executing benchmark command: $COMMAND"
eval "$COMMAND"

# Check the exit code of the benchmark
if [ $? -ne 0 ]; then
    echo "!!! Benchmark command failed for task $SLURM_ARRAY_TASK_ID. Halting execution for this task. !!!"
    exit 1
fi

#=========================================================================================
# Score Aggregation (Merged from worker_script.sh)
#=========================================================================================
echo "----------------------------------------------------------"
echo "Benchmark finished. Now running the corresponding score aggregation."
echo "----------------------------------------------------------"

AGG_COMMAND="python TSPulse2/read_scores.py \"$CURRENT_AD_NAME\" \"$CURRENT_AGG_MODE\""

echo "Executing aggregation command: $AGG_COMMAND"
eval "$AGG_COMMAND"

echo "=========================================================="
echo "Job Task $SLURM_ARRAY_TASK_ID fully finished at: $(date)"
echo "--- SLURM JOB ARRAY TASK END ---"