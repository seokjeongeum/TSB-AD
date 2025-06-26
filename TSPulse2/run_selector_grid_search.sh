#!/bin/bash
#=========================================================================================
#
# Slurm SBATCH Script for Architecture Search on the Selector Model
#
#=========================================================================================

#SBATCH --job-name=SelectorArchSearch
#SBATCH --output=slurm_logs/%A_%x_%a.out
#SBATCH --error=slurm_logs/%A_%x_%a.err
#SBATCH --partition=A100-80GB
#SBATCH --qos=add_hpgpu
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

# FIX: Moved the --array directive here with all other SBATCH directives.
# The array size must be hardcoded.
#SBATCH --array=1-21

#=========================================================================================
#
# Model Architecture Definition
#
#=========================================================================================
MODELS_TO_TEST=(
    "BestOfBreedMLP" "MLP" "ResMLP" "SkipMLP" "CNN1D" "Encoder" "RandomForest"
    "XGBoost" "CatBoost" "ExtraTrees" "GradientBoosting" "AdaBoost" "SVC" "KNN"
    "LogisticRegression" "GaussianNB" "LDA" "DecisionTree" "QDA" "Bagging" "PassiveAggressive"
)

#=========================================================================================
#
# Shared Setup - All commands must come AFTER the SBATCH directives.
#
#=========================================================================================
set -e # Exit immediately on error

# Check if running as a slurm array job.
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: This script is not running as a Slurm job array." >&2
    echo "This may be because the #SBATCH --array directive is misplaced." >&2
    exit 1
fi

# FIX: Moved the informational echo here. It will now appear in each job's log.
NUM_JOBS=${#MODELS_TO_TEST[@]}
echo "INFO: Starting task for one of ${NUM_JOBS} models in the architecture search."

# Define project root first, so it can be used for all paths
PROJECT_ROOT="$SLURM_SUBMIT_DIR"

echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Host: $(hostname)"
echo "----------------------------------"

mkdir -p "${PROJECT_ROOT}/slurm_logs"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"

module load cuda/11.8 conda-py39
conda activate tsb-ad-env

cd "$PROJECT_ROOT"

#=========================================================================================
#
# Task-Specific Logic & Execution
#
#=========================================================================================
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_MODEL=${MODELS_TO_TEST[$TASK_INDEX]}

CURRENT_OUTPUT_DIR="${PROJECT_ROOT}/TSPulse2/selector_arch_search_combined/${CURRENT_MODEL}"
mkdir -p "$CURRENT_OUTPUT_DIR"

echo "=========================================================="
echo "Running Task ${SLURM_ARRAY_TASK_ID} for Selector Architecture:"
echo "Python Script: TSPulse2/train_selector_with_embeddings.py"
echo "Model to Use: $CURRENT_MODEL"
echo "Augmentation: Enabled"
echo "Output Dir: $CURRENT_OUTPUT_DIR"
echo "=========================================================="

# MODIFIED: Updated the command to use the new required arguments
# and added the flag for the detailed report.
COMMAND="python -u TSPulse2/train_selector_with_embeddings.py \
    --output_model_dir=\"${CURRENT_OUTPUT_DIR}\" \
    --model_to_use=${CURRENT_MODEL} \
    --detailed_report"

echo "Executing command: ${COMMAND}"
eval "${COMMAND}"

if [ $? -ne 0 ]; then
    echo "!!! Selector training command failed for task ${SLURM_ARRAY_TASK_ID} (${CURRENT_MODEL}). !!!"
    touch "${CURRENT_OUTPUT_DIR}/_FAILED"
    exit 1
fi

touch "${CURRENT_OUTPUT_DIR}/_SUCCESS"

echo "=========================================================="
echo "Job Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "--- SLURM JOB ARRAY TASK END ---" 