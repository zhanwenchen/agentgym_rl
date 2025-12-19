#!/bin/bash
# Local Evaluation Script for ScienceWorld
# Usage: bash scripts/local/train_3b_memory.sh

# Tee logging wrapper - redirect stdout and stderr to log file
set -euo pipefail
set -x

export DIRPATH_PROJECT="${HOME}/workspace/agentgym_rl"
export EXP_NAME="eval_3b"

source "${DIRPATH_PROJECT}/scripts/damodel/logging.sh"


# ============================================================================
# Configuration
# ============================================================================
export MODEL_SIZE="3b"
export MEMORY_ENABLED=true  # Set to false to disable memory
export MEMORY_K=3  # Set to false to disable memory
export MEMORY_MIN_REWARD=0.5

export ckpt_path="${DIRPATH_PROJECT}/AgentGym-RL/saves/${MODEL_SIZE}_n8/global_step_75/actor"
# /home/zhanwen/agentgym_rl/AgentGym-RL/saves/0.5b_n8/global_step_25/actor
source "${DIRPATH_PROJECT}/scripts/damodel/eval/eval_shared.sh"
