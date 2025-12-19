#!/bin/bash

# Check for training-specific environment variables
# if [ -z "${CHECKPOINT_DIR}" ]; then
#     echo "Error: CHECKPOINT_DIR environment variable required"
#     exit 1
# fi

# Source common setup
source "${DIRPATH_PROJECT}/scripts/damodel/shared_setup.sh"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


# Export training-specific variables
# echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
# export MODEL_NAME="qwen-${MODEL_SIZE}"
# echo "MODEL_NAME: ${MODEL_NAME}"

echo "Starting training..."

# Run training
bash "${DIRPATH_PROJECT}/examples/train/AgentGym-RL/sciworld_train.sh" 2>&1
TRAIN_EXIT_CODE=$?

# Cleanup: kill the server
echo "Training finished with exit code: $TRAIN_EXIT_CODE"
echo "Killing server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null

echo "End time: $(date)"
exit $TRAIN_EXIT_CODE
