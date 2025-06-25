#!/bin/bash
#=========================================================================================
# Slurm SBATCH Directives for a Hyperparameter Grid Search
#
# This script systematically tests all combinations of specified hyperparameters
# for the train_classifier.py script. It generates a parameter list and uses a
# Slurm job array to execute each unique configuration as a separate task.
#
# PREREQUISITE: The python script 'train_classifier.py' MUST be modified to accept
# command-line arguments for the hyperparameters being tested. See comments above.
#=========================================================================================
#
#SBATCH --job-name=TSPulse_GridSearch    # A general name for the array job
#SBATCH --output=slurm_logs/%x_%A_%a.out # Unique log for each task (%x=job-name, %A=job-id, %a=task-id)
#SBATCH --error=slurm_logs/%x_%A_%a.err  # Unique error log for each task
#
#SBATCH --partition=A100-80GB            # The GPU partition you want to use
#SBATCH --qos=hpgpu                      # Requesting permission for the partition
#SBATCH --gres=gpu:1                     # Request 1 GPU *per task*
#SBATCH --time=3-00:00:00                 # Max wall time (3 days)

#=========================================================================================
# Hyperparameter Grid Definition
#
# Define the sets of hyperparameters to test. The script will generate all combinations.
# The notebook suggests these are good candidates for tuning:
# head_reduce_d_model, decoder_mode, head_gated_attention_activation,
# mask_ratio, channel_virtual_expand_scale
#=========================================================================================
HEAD_REDUCE_D_MODELS=(1 2)
DECODER_MODES=("mix_channel" "common_channel")
MASK_RATIOS=(0.0 0.3)
HEAD_GATED_ATTENTION_ACTIVATIONS=("softmax" "sigmoid")
CHANNEL_VIRTUAL_EXPAND_SCALES=(1 2)


#=========================================================================================
# Combination Generation
#
# This block creates an array 'PARAMS' where each element is a string containing
# a unique combination of the hyperparameters defined above.
#=========================================================================================
declare -a PARAMS
for reduce_d in "${HEAD_REDUCE_D_MODELS[@]}"; do
  for decoder in "${DECODER_MODES[@]}"; do
    for mask in "${MASK_RATIOS[@]}"; do
      for head_act in "${HEAD_GATED_ATTENTION_ACTIVATIONS[@]}"; do
        for expand_scale in "${CHANNEL_VIRTUAL_EXPAND_SCALES[@]}"; do
          # Each combination is a single space-separated string
          PARAMS+=("$reduce_d $decoder $mask $head_act $expand_scale")
        done
      done
    done
  done
done

#=========================================================================================
# Slurm Array Configuration
#
# Set the array size to the total number of combinations we generated.
#=========================================================================================
NUM_JOBS=${#PARAMS[@]}
echo "INFO: Submitting a job array with ${NUM_JOBS} tasks."
#SBATCH --array=1-${NUM_JOBS}

#=========================================================================================
# Shared Setup (runs for every task)
#=========================================================================================
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / ${NUM_JOBS}"
echo "Host: $(hostname)"
echo "----------------------------------"

# Ensure the log directory exists before any task tries to write to it.
mkdir -p slurm_logs

# Use the reliable $SLURM_SUBMIT_DIR to define the project root
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"

# Load modules and activate Conda environment
module load cuda/11.8 conda-py39
conda activate tsb-ad-env

# Change to the project directory
cd "$PROJECT_ROOT"

#=========================================================================================
# Task-Specific Logic & Execution
#=========================================================================================
# SLURM_ARRAY_TASK_ID is 1-based, but our array is 0-based. Subtract 1 for the index.
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Retrieve the space-separated parameter string for the current task
CURRENT_PARAMS_STR=${PARAMS[$TASK_INDEX]}

# Read the parameters from the string into an array
read -r -a CURRENT_PARAMS <<< "$CURRENT_PARAMS_STR"

# Assign parameters to named variables for clarity
CURRENT_REDUCE_D=${CURRENT_PARAMS[0]}
CURRENT_DECODER=${CURRENT_PARAMS[1]}
CURRENT_MASK=${CURRENT_PARAMS[2]}
CURRENT_HEAD_ACT=${CURRENT_PARAMS[3]}
CURRENT_EXPAND_SCALE=${CURRENT_PARAMS[4]}


# Create a unique output directory for this specific run to avoid conflicts
RUN_ID="reduce_${CURRENT_REDUCE_D}_decoder_${CURRENT_DECODER}_mask_${CURRENT_MASK}_headact_${CURRENT_HEAD_ACT}_expand_${CURRENT_EXPAND_SCALE}"
CURRENT_OUTPUT_DIR="${PROJECT_ROOT}/TSPulse2/grid_search_results/${RUN_ID}"
mkdir -p "$CURRENT_OUTPUT_DIR"

echo "=========================================================="
echo "Running Task ${SLURM_ARRAY_TASK_ID} with Parameters:"
echo "Python Script: TSPulse2/train_classifier.py"
echo "Head Reduce D Model: $CURRENT_REDUCE_D"
echo "Decoder Mode: $CURRENT_DECODER"
echo "Mask Ratio: $CURRENT_MASK"
echo "Head Gated Attention Activation: $CURRENT_HEAD_ACT"
echo "Channel Virtual Expand Scale: $CURRENT_EXPAND_SCALE"
echo "Output Dir: $CURRENT_OUTPUT_DIR"
echo "=========================================================="

# Construct and run the final python command
COMMAND="python -u TSPulse2/train_classifier.py \
  --output_dir=\"${CURRENT_OUTPUT_DIR}\" \
  --head_reduce_d_model=${CURRENT_REDUCE_D} \
  --decoder_mode=\"${CURRENT_DECODER}\" \
  --mask_ratio=${CURRENT_MASK} \
  --head_gated_attention_activation=\"${CURRENT_HEAD_ACT}\" \
  --channel_virtual_expand_scale=${CURRENT_EXPAND_SCALE}"

echo "Executing command: $COMMAND"
eval "$COMMAND"

# Check the exit code of the training script
if [ $? -ne 0 ]; then
    echo "!!! Training command failed for task $SLURM_ARRAY_TASK_ID. !!!"
    # Create a failure token file for easy identification of failed runs
    touch "${CURRENT_OUTPUT_DIR}/_FAILED"
    exit 1
fi

# Create a success token file
touch "${CURRENT_OUTPUT_DIR}/_SUCCESS"

echo "=========================================================="
echo "Job Task $SLURM_ARRAY_TASK_ID fully finished at: $(date)"
echo "--- SLURM JOB ARRAY TASK END ---" 