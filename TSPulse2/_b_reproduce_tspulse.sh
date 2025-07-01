#!/bin/bash
#=========================================================================================
# Slurm SBATCH Directives for TSAD Benchmarking
#
# This script runs the full benchmark suite described in the README.md.
# It uses a Slurm job array to execute all 16 combinations of datasets,
# evaluation types, and modes.
#=========================================================================================
#
#SBATCH --job-name=TSAD_Benchmark
#SBATCH --output=slurm_logs/%A_%x_%a.out      # Unique log for each task
#SBATCH --error=slurm_logs/%A_%x_%a.err       # Unique error log for each task

# --- Resource Allocation ---
# Adjust partition, QoS, and time as needed for your environment.
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

# --- Job Array Configuration ---
# Total combinations: 2 datasets (U/M) * 2 eval types (Tuning/Eva) * 4 modes = 16 tasks
#SBATCH --array=1-16

#=========================================================================================
# HYPERPARAMETER & LOGIC BLOCK
#=========================================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Define the Parameter Grid
# Using loops to generate all parameter combinations cleanly.
declare -a PARAMS

DATASET_ABBRS=("U" "M")
EVAL_TYPES=("Tuning" "Eva")
MODES=("time" "fft" "forecast" "time+fft+forecast")
MODE_NAMES=("time" "fft" "forecast" "ensemble") # For output filenames

for data_abbr in "${DATASET_ABBRS[@]}"; do
  for eval_type in "${EVAL_TYPES[@]}"; do
    for i in "${!MODES[@]}"; do
      mode=${MODES[$i]}
      mode_name=${MODE_NAMES[$i]}
      PARAMS+=("$data_abbr $eval_type $mode $mode_name")
    done
  done
done

# 2. Task-Specific Setup
# Define Project Root from the submission directory first.
# This script assumes you run `sbatch` from the top-level project directory.
PROJECT_ROOT="$SLURM_SUBMIT_DIR"

# Each task in the array will select a unique parameter set.
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
read -r -a CURRENT_PARAMS <<< "${PARAMS[$TASK_INDEX]}"

CURRENT_DATA_ABBR=${CURRENT_PARAMS[0]}
CURRENT_EVAL_TYPE=${CURRENT_PARAMS[1]}
CURRENT_MODE=${CURRENT_PARAMS[2]}
CURRENT_MODE_NAME=${CURRENT_PARAMS[3]}

# Construct absolute file paths using the Project Root.
# The data directory path is now deeper to point directly to the CSV files.
if [ "$CURRENT_DATA_ABBR" == "U" ]; then
    DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-U"
else
    DATA_DIR="${PROJECT_ROOT}/Datasets/TSB-AD-M"
fi
EVAL_FILE="${PROJECT_ROOT}/Datasets/File_List/TSB-AD-${CURRENT_DATA_ABBR}-${CURRENT_EVAL_TYPE}.csv"
OUT_FILE="benchmarks/TSB-AD-${CURRENT_DATA_ABBR}-${CURRENT_EVAL_TYPE}-${CURRENT_MODE_NAME}.csv"

# 3. Environment Setup
echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Host: $(hostname)"
echo "----------------------------------"

# Define experiment directory.
EXPERIMENT_DIR="${PROJECT_ROOT}/granite-tsfm/notebooks/hfdemo/tspulse/anomaly_detection"

# Add the project's root and the `granite-tsfm` directory to Python's path.
# This ensures that modules like `tsfm_public` can be found.
export PYTHONPATH="${PROJECT_ROOT}/granite-tsfm:${PYTHONPATH}"

# Create slurm_logs in the project root, where Slurm expects to write output.
mkdir -p "${PROJECT_ROOT}/slurm_logs"

# Change to the experiment directory to run the script.
cd "${EXPERIMENT_DIR}"

# Create the benchmarks directory here, relative to the experiment script.
mkdir -p benchmarks

# Activate your python environment.
# Make sure the 'tsb-ad-env' conda environment is properly set up.
# You might need to adjust 'conda-py39' depending on your system's module names.
module load cuda/11.8 conda-py39
conda activate tsb-ad-env

# 4. Execution
echo "=========================================================="
echo "Running Task ${SLURM_ARRAY_TASK_ID} with parameters:"
echo "  Data Directory: $DATA_DIR"
echo "  Evaluation File: $EVAL_FILE"
echo "  Mode: $CURRENT_MODE"
echo "  Output File: $OUT_FILE"
echo "=========================================================="

# Construct and run the final python command
COMMAND="python run_experiment.py \
  --data_direc \"${DATA_DIR}\" \
  --eval_file \"${EVAL_FILE}\" \
  --mode \"${CURRENT_MODE}\" \
  --out_file \"${OUT_FILE}\""

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