#!/bin/bash

echo "Starting submission of 18 benchmark jobs..."
mkdir -p slurm_logs # Ensure the log directory exists

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
    "Datasets/File_List/TSB-M-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
    "Datasets/File_List/TSB-M-Eva.csv" "Datasets/File_List/TSB-AD-M-Tuning.csv" "Datasets/File_List/TSB-AD-U-Eva.csv" "Datasets/File_List/TSB-AD-U-Tuning.csv"
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

# Loop from 0 to 17 (for 18 jobs)
for i in {0..17}
do
    # Get parameters for the current job
    AD_NAME=${AD_NAMES[$i]}
    SCRIPT=${RUN_SCRIPTS[$i]}
    DATA_DIR=${DATASET_DIRS[$i]}
    FILE_LIST=${FILE_LISTS[$i]}
    EXTRA=${EXTRA_ARGS[$i]}
    AGG_MODE=${AGGREGATION_MODES[$i]}

    # --- Create custom job name and log file names ---
    # Sanitize the file list name for use in filenames (e.g., replace '/' with '_')
    SAFE_FILE_LIST_NAME=$(echo "$FILE_LIST" | tr '/' '_')
    JOB_NAME="${AD_NAME}_${AGG_MODE}"
    LOG_FILE="slurm_logs/${JOB_NAME}_%j.log" # %j will be replaced by the job ID

    # Submit the worker script with parameters passed as environment variables
    sbatch \
      --job-name="$JOB_NAME" \
      --output="$LOG_FILE" \
      --error="$LOG_FILE" \
      --export=ALL,TASK_ID=$((i+1)),CURRENT_AD_NAME="$AD_NAME",CURRENT_SCRIPT="$SCRIPT",CURRENT_DATA_DIR="$DATA_DIR",CURRENT_FILE_LIST="$FILE_LIST",CURRENT_EXTRA_ARGS="$EXTRA",CURRENT_AGG_MODE="$AGG_MODE" \
      worker_script.sh
    
    echo "Submitted job $((i+1)): $JOB_NAME"
    sleep 1 # Sleep for a second to avoid overwhelming the scheduler
done

echo "All 18 jobs have been submitted."