#!/bin/bash

# Check for training-specific environment variables
if [ -z "${MODEL_SIZE}" ]; then
    echo "Error: MODEL_SIZE environment variable required"
    exit 1
fi

if [ -z "${USE_MEMORY}" ]; then
    echo "Error: USE_MEMORY environment variable required"
    exit 1
fi

# Source common setup
source "${HOME}/agentgym_rl/scripts/slurm/shared_setup.sh"

echo "=========================================="
echo "ScienceWorld Training with Memory (SLURM)"
echo "Model Size: ${MODEL_SIZE}"
echo "Memory Enabled: ${USE_MEMORY}"
echo "=========================================="

# Run training with memory configuration
cd "${HOME}/agentgym_rl/AgentGym-RL"

if [ "${USE_MEMORY}" = true ]; then
    echo "Memory bank: ENABLED"
    HYDRA_FULL_ERROR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_MODE=online \
    bash "${HOME}/agentgym_rl/examples/train/AgentGym-RL/sciworld_train.sh" \
        actor_rollout_ref.rollout.memory.enabled=true \
        actor_rollout_ref.rollout.memory.k=3 \
        actor_rollout_ref.rollout.memory.min_reward=0.5 \
        actor_rollout_ref.rollout.memory.save_path="outputs/memory_bank/sciworld_${MODEL_SIZE}" 2>&1
else
    echo "Memory bank: DISABLED"
    HYDRA_FULL_ERROR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_MODE=online \
    bash "${HOME}/agentgym_rl/examples/train/AgentGym-RL/sciworld_train.sh" \
        actor_rollout_ref.rollout.memory.enabled=false 2>&1
fi

TRAIN_EXIT_CODE=$?

# Cleanup: kill the server
echo "Training finished with exit code: $TRAIN_EXIT_CODE"
echo "Killing server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null

echo "End time: $(date)"
exit $TRAIN_EXIT_CODE
