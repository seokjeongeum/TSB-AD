#!/bin/bash

#=========================================================================================
# Slurm SBATCH Directives for a Job Array
#=========================================================================================
#
#SBATCH --job-name=TSPulse_Benchmark      # A general name for the array job
#SBATCH --output=slurm_logs/%x_%A_%a.out  # Unique log for each task
#SBATCH --error=slurm_logs/%x_%A_%a.err   # Unique error log for each task
#
#SBATCH --partition=A100-80GB             # The GPU partition you want to use
#SBATCH --qos=hpgpu                       # Requesting permission for the partition
#SBATCH --gres=gpu:1                      # Request 1 GPU *per task*
#SBATCH --time=3-00:00:00                 # Max wall time (3 days)
#
#SBATCH --array=1-18                      # Creates 18 tasks, numbered 1 through 18
#
#=========================================================================================
# Shared Setup (runs for every task)
#=========================================================================================

# --- Define the parameters for all 18 experiments in bash arrays ---
AD_NAMES=(
    "TSPulse2" "TSPulse2"
    "TSPulse_ZS_ensemble" "TSPulse_ZS_ensemble" "TSPulse_ZS_ensemble" "TSPulse_ZS_ensemble"
    "TSPulse_ZS_fft" "TSPulse_ZS_fft" "TSPulse_ZS_fft" "TSPulse_ZS_fft"
    "TSPulse_ZS_future" "TSPulse_ZS_future" "TSPulse_ZS_future" "TSPulse_ZS_future"
    "TSPulse_ZS_time" "TSPulse_ZS_time" "TSPulse_ZS_time" "TSPulse_ZS_time"
)
RUN_SCRIPTS=(
    "Run_Detector_M.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_U.py"
    "Run_Detector_M.py" "Run_Detector_M.py" "Run_Detector_U.py" "Run_Detector_U.py"
)
DATASET_DIRS=(
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/"
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-U/"
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-U/"
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-U/"
    "Datasets/TSB-AD-M/" "Datasets/TSB-AD-M/" "Datasets/TSB-AD-U/" "Datasets/TSB-AD-U/"
)
FILE_LISTS=(
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-U-Eva.csv"
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
    "Datasets/File_List/TSB-AD-M-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
)
EXTRA_ARGS=(
    "" ""
    "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
    "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
    "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
    "" "--score_dir 'eval/score/multi-tuning/' --save_dir 'eval/metrics/multi-tuning/'" "" "--score_dir 'eval/score/uni-tuning/' --save_dir 'eval/metrics/uni-tuning/'"
)
AGGREGATION_MODES=(
    "multi" "uni"
    "multi" "multi-tuning" "uni" "uni-tuning"
    "multi" "multi-tuning" "uni" "uni-tuning"
    "multi" "multi-tuning" "uni" "uni-tuning"
    "multi" "multi-tuning" "uni" "uni-tuning"
)
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_AD_NAME=${AD_NAMES[$TASK_INDEX]}
CURRENT_SCRIPT=${RUN_SCRIPTS[$TASK_INDEX]}
CURRENT_DATA_DIR=${DATASET_DIRS[$TASK_INDEX]}
CURRENT_FILE_LIST=${FILE_LISTS[$TASK_INDEX]}
CURRENT_EXTRA_ARGS=${EXTRA_ARGS[$TASK_INDEX]}
CURRENT_AGG_MODE=${AGGREGATION_MODES[$TASK_INDEX]}
PROJECT_ROOT="/home/seokjeongeum/TSB-AD"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"

#=========================================================================================
# Execution - Part 1: Main Benchmark
#=========================================================================================
echo "=========================================================="
echo "Starting SLURM Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Running with Parameters: AD_Name=$CURRENT_AD_NAME, Script=$CURRENT_SCRIPT, Aggregation Mode: $CURRENT_AGG_MODE"
echo "=========================================================="

# Load modules and activate environment
# CORRECTED: Using the available 'conda-py39' module.
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
# Execution - Part 2: Corresponding Score Aggregation
#=========================================================================================
echo "----------------------------------------------------------"
echo "Benchmark finished. Now running the corresponding score aggregation."
echo "----------------------------------------------------------"

AGG_COMMAND="python TSPulse2/read_scores.py \"$CURRENT_AD_NAME\" \"$CURRENT_AGG_MODE\""

echo "Executing aggregation command: $AGG_COMMAND"
eval "$AGG_COMMAND"

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID fully finished at: $(date)"
echo "=========================================================="
