if [ -z "${LOGGING_ENABLED:-}" ]; then
    export LOGGING_ENABLED=1
    LOG_DIR="${HOME}/agentgym_rl/logs"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/eval_3b_memory_$(date +%Y%m%d%H%M%S).log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "Logging to: ${LOG_FILE}"
fi
