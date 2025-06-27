#!/bin/bash
#=========================================================================================
# Slurm SBATCH Directives for a Hyperparameter Grid Search on RTX2080Ti GPUs
#
# This script systematically tests all combinations of specified hyperparameters
# for the train_classifier.py script. It generates a parameter list and uses a
# Slurm job array to execute each unique configuration as a separate task.
#
# This version is specifically corrected to run on the RTX2080Ti partition.
#=========================================================================================
#
#SBATCH --job-name=TSPulse_GridSearch_2080Ti # A specific name for this grid search
#SBATCH --output=slurm_logs/%A_%x_%a.out      # Unique log for each task (%x=job-name, %A=job-id, %a=task-id)
#SBATCH --error=slurm_logs/%A_%x_%a.err       # Unique error log for each task
#
# === CORRECTED RESOURCE REQUESTS for RTX2080Ti ===
# 1. Specify the correct partition for RTX 2080 Ti GPUs.
#    Based on the cluster summary table, these are on nodes n[1-6].
#    The partition is likely named after the GPU type.
#SBATCH --partition=2080ti
#
# 2. Specify the Quality of Service (QOS). For older/standard GPUs,
#    sometimes no QOS is needed, or a general one like 'hpgpu' is used.
#    We will start by not specifying a QOS, letting Slurm use the partition's default.
#    If the job fails to schedule, you can try adding '#SBATCH --qos=hpgpu'.
#
# 3. Set a reasonable time limit.
#SBATCH --time=3-00:00:00
#
# 4. Request 1 GPU per task. The --gres flag will select an RTX2080Ti
#    within the specified partition.
#SBATCH --gres=gpu:1
# =================================================

# Set the array size. The total number of combinations is 2*2*2*2*2 = 32.
#SBATCH --array=1-32

#=========================================================================================
# HYPERPARAMETER & LOGIC BLOCK
# This entire block will be executed by EACH task in the job array.
#=========================================================================================

# Exit immediately if a command exits with a non-zero status.
set -e 

# 1. Define the Hyperparameter Grid
HEAD_REDUCE_D_MODELS=(1 2)
DECODER_MODES=("mix_channel" "common_channel")
MASK_RATIOS=(0.0 0.3)
HEAD_GATED_ATTENTION_ACTIVATIONS=("softmax" "sigmoid")
CHANNEL_VIRTUAL_EXPAND_SCALES=(1 2)

# Set a batch size for all runs in this grid search.
# The default in train_classifier.py is now 1, which can be slow for a grid search.
# Adjust this value based on your available GPU memory.
BATCH_SIZE=16384

# 2. Generate All Combinations
declare -a PARAMS
for reduce_d in "${HEAD_REDUCE_D_MODELS[@]}"; do
  for decoder in "${DECODER_MODES[@]}"; do
    for mask in "${MASK_RATIOS[@]}"; do
      for head_act in "${HEAD_GATED_ATTENTION_ACTIVATIONS[@]}"; do
        for expand_scale in "${CHANNEL_VIRTUAL_EXPAND_SCALES[@]}"; do
          PARAMS+=("$reduce_d $decoder $mask $head_act $expand_scale")
        done
      done
    done
  done
done

# 3. Task-Specific Setup
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_PARAMS_STR=${PARAMS[$TASK_INDEX]}
read -r -a CURRENT_PARAMS <<< "$CURRENT_PARAMS_STR"

CURRENT_REDUCE_D=${CURRENT_PARAMS[0]}
CURRENT_DECODER=${CURRENT_PARAMS[1]}
CURRENT_MASK=${CURRENT_PARAMS[2]}
CURRENT_HEAD_ACT=${CURRENT_PARAMS[3]}
CURRENT_EXPAND_SCALE=${CURRENT_PARAMS[4]}

# 4. Environment Setup
echo "--- SLURM JOB ARRAY TASK START ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Host: $(hostname)"
echo "----------------------------------"

mkdir -p slurm_logs

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/granite-tsfm"

module load cuda/11.8 conda-py39
conda activate tsb-ad-env
cd "$PROJECT_ROOT"

# 5. Execution
RUN_ID="reduce_${CURRENT_REDUCE_D}_decoder_${CURRENT_DECODER}_mask_${CURRENT_MASK}_headact_${CURRENT_HEAD_ACT}_expand_${CURRENT_EXPAND_SCALE}"
CURRENT_OUTPUT_DIR="${PROJECT_ROOT}/TSPulse2/grid_search_results_2080ti/${RUN_ID}"
mkdir -p "$CURRENT_OUTPUT_DIR"

echo "=========================================================="
echo "Running Task ${SLURM_ARRAY_TASK_ID} on Partition RTX2080ti"
echo "Parameters:"
echo "  Head Reduce D Model: $CURRENT_REDUCE_D"
echo "  Decoder Mode: $CURRENT_DECODER"
echo "  Mask Ratio: $CURRENT_MASK"
echo "  Head Gated Attention Activation: $CURRENT_HEAD_ACT"
echo "  Channel Virtual Expand Scale: $CURRENT_EXPAND_SCALE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output Dir: $CURRENT_OUTPUT_DIR"
echo "=========================================================="

# Construct and run the final python command
COMMAND="python -u TSPulse2/train_classifier.py \
  --output_dir=\"${CURRENT_OUTPUT_DIR}\" \
  --batch_size=${BATCH_SIZE} \
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
    touch "${CURRENT_OUTPUT_DIR}/_FAILED"
    exit 1
fi

touch "${CURRENT_OUTPUT_DIR}/_SUCCESS"

echo "=========================================================="
echo "Job Task $SLURM_ARRAY_TASK_ID fully finished at: $(date)"
echo "--- SLURM JOB ARRAY TASK END ---" 