#!/bin/bash
# Local Evaluation Script for ScienceWorld
# Usage: bash scripts/local/train_3b_memory.sh

# Tee logging wrapper - redirect stdout and stderr to log file
if [ -z "${LOGGING_ENABLED:-}" ]; then
    export LOGGING_ENABLED=1
    LOG_DIR="${HOME}/agentgym_rl/logs"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/eval_0.5b_$(date +%Y%m%d%H%M%S).log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "Logging to: ${LOG_FILE}"
fi

set -euo pipefail
set -x

# ============================================================================
# Configuration
# ============================================================================
export MODEL_SIZE="0.5b"
export USE_MEMORY=false  # Set to false to disable memory


export ckpt_path="${HOME}/agentgym_rl/AgentGym-RL/saves/${MODEL_SIZE}_n8/global_step_25/actor"
# /home/zhanwen/agentgym_rl/AgentGym-RL/saves/0.5b_n8/global_step_25/actor
source "${HOME}/agentgym_rl/scripts/local/eval/eval_shared.sh"
