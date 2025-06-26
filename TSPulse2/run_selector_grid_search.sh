#!/bin/bash
#=========================================================================================
# Slurm SBATCH Script for Architecture Search on the Selector Model
#
# This script systematically tests a list of model architectures for the
# train_selector_with_embeddings.py script. It uses a Slurm job array to
# execute each architecture as a separate training task.
#=========================================================================================
#
#SBATCH --job-name=SelectorArchSearch
#SBATCH --output=slurm_logs/%x_%A_%a.out # Unique log for each task
#SBATCH --error=slurm_logs/%x_%A_%a.err  # Unique error log for each task
#
#SBATCH --partition=A100-80GB            # The GPU partition to use
#SBATCH --qos=hpgpu                      # Requesting permission for the partition
#SBATCH --gres=gpu:1                     # Request 1 GPU per task
#SBATCH --time=1-00:00:00                # Max wall time (1 day)

#=========================================================================================
# Model Architecture Definition
#
# This list should match the keys in the 'architectures' dictionary in the
# train_selector_with_embeddings.py script.
#=========================================================================================
MODELS_TO_TEST=(
    "BestOfBreedMLP"
    "MLP"
    "ResMLP"
    "SkipMLP"
    "CNN1D"
    "Encoder"
    "RandomForest"
    "XGBoost"
    "CatBoost"
    "ExtraTrees"
    "GradientBoosting"
    "AdaBoost"
    "SVC"
    "KNN"
    "LogisticRegression"
    "GaussianNB"
    "LDA"
    "DecisionTree"
    "QDA"
    "Bagging"
    "PassiveAggressive"
)

#=========================================================================================
# Slurm Array Configuration
#=========================================================================================
NUM_JOBS=${#MODELS_TO_TEST[@]}
echo "INFO: Submitting a job array with ${NUM_JOBS} tasks for selector architecture search."
#SBATCH --array=1-${NUM_JOBS}

#=========================================================================================
# Shared Setup
#=========================================================================================
set -e # Exit immediately on error

echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / ${NUM_JOBS}"
echo "Host: $(hostname)"
echo "----------------------------------"

mkdir -p slurm_logs

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"

module load cuda/11.8 conda-py39
conda activate tsb-ad-env

cd "$PROJECT_ROOT"

#=========================================================================================
# Task-Specific Logic & Execution
#=========================================================================================
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_MODEL=${MODELS_TO_TEST[$TASK_INDEX]}

# Create a unique output directory for this specific model run
CURRENT_OUTPUT_DIR="TSPulse2/selector_arch_search_combined/${CURRENT_MODEL}"
mkdir -p "$CURRENT_OUTPUT_DIR"

echo "=========================================================="
echo "Running Task ${SLURM_ARRAY_TASK_ID} for Selector Architecture:"
echo "Python Script: TSPulse2/train_selector_with_embeddings.py"
echo "Model to Use: $CURRENT_MODEL"
echo "Augmentation: Enabled"
echo "Output Dir: $CURRENT_OUTPUT_DIR"
echo "=========================================================="

# Construct and run the final python command
# Augmentation is always on, as requested.
COMMAND="python -u TSPulse2/train_selector_with_embeddings.py \
  --output_model_dir=\"${CURRENT_OUTPUT_DIR}\" \
  --model_to_use=${CURRENT_MODEL}"

echo "Executing command: $COMMAND"
eval "$COMMAND"

if [ $? -ne 0 ]; then
    echo "!!! Selector training command failed for task $SLURM_ARRAY_TASK_ID ($CURRENT_MODEL). !!!"
    touch "${CURRENT_OUTPUT_DIR}/_FAILED"
    exit 1
fi

touch "${CURRENT_OUTPUT_DIR}/_SUCCESS"

echo "=========================================================="
echo "Job Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "--- SLURM JOB ARRAY TASK END ---" 