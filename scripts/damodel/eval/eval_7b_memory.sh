#!/bin/bash
# Local Evaluation Script for ScienceWorld
# Usage: bash scripts/local/train_3b_memory.sh

# Tee logging wrapper - redirect stdout and stderr to log file
set -euo pipefail
set -x

export DIRPATH_PROJECT="${HOME}/workspace/agentgym_rl"
export MODEL_SIZE="7b"
export EXP_NAME="eval_${MODEL_SIZE}_memory"

source "${DIRPATH_PROJECT}/scripts/damodel/logging.sh"


# ============================================================================
# Configuration
# ============================================================================
export MEMORY_ENABLED=true  # Set to false to disable memory
export MEMORY_K=5  # Set to false to disable memory
export MEMORY_MIN_REWARD=0.5
export MEMORY_DISTANCE_METRIC='l2'
export STEPNUM_CHECKPOINT=125

export DIRNAME_CHECKPOINT="train_7b_memory_20260103134836_20260103_134841"
# ln -sf "${HOME}/shared-storage/agentgym_rl_AgentGym-RL_saves/${DIRNAME_CHECKPOINT}" "AgentGym-RL/saves/" && ll "AgentGym-RL/saves/${DIRNAME_CHECKPOINT}"


export ckpt_path="${DIRPATH_PROJECT}/AgentGym-RL/saves/${DIRNAME_CHECKPOINT}/global_step_${STEPNUM_CHECKPOINT}/actor"
source "${DIRPATH_PROJECT}/scripts/damodel/eval/eval_shared.sh"
echo "eval_7b_memory.sh: evaluated ckpt_path=${ckpt_path} with MEMORY_ENABLED=${MEMORY_ENABLED}, MEMORY_K=${MEMORY_K}, MEMORY_MIN_REWARD=${MEMORY_MIN_REWARD}, MEMORY_DISTANCE_METRIC=${MEMORY_DISTANCE_METRIC}."
