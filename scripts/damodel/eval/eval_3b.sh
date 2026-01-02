#!/bin/bash
# Local Evaluation Script for ScienceWorld
# Usage: bash scripts/local/train_3b_memory.sh

# Tee logging wrapper - redirect stdout and stderr to log file
set -euo pipefail
set -x

export DIRPATH_PROJECT="${HOME}/workspace/agentgym_rl"
export MODEL_SIZE="3b"
export EXP_NAME="eval_${MODEL_SIZE}"

source "${DIRPATH_PROJECT}/scripts/damodel/logging.sh"


# ============================================================================
# Configuration
# ============================================================================
export MEMORY_ENABLED=false  # Set to false to disable memory
export MEMORY_K=3  # Set to false to disable memory
export MEMORY_MIN_REWARD=0.5

export DIRNAME_CHECKPOINT="${MODEL_SIZE}_n8_riv"
export STEPNUM_CHECKPOINT=125

export ckpt_path="${DIRPATH_PROJECT}/AgentGym-RL/saves/${DIRNAME_CHECKPOINT}/global_step_${STEPNUM_CHECKPOINT}/actor"
# /home/zhanwen/agentgym_rl/AgentGym-RL/saves/0.5b_n8/global_step_25/actor
source "${DIRPATH_PROJECT}/scripts/damodel/eval/eval_shared.sh"
echo "eval_3b.sh: evaluated ckpt_path=${ckpt_path} with MEMORY_ENABLED=${MEMORY_ENABLED}, MEMORY_K=${MEMORY_K}, MEMORY_MIN_REWARD=${MEMORY_MIN_REWARD}"
