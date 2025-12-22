#!/bin/bash


# ============================================================================
# Start ScienceWorld Server
# ============================================================================
# Find available port
export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo "Starting sciworld server on port ${PORT}..."
sciworld --host localhost --port "${PORT}" > /dev/null 2>&1 &
export SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"


export TMPDIR="${HOME}/tmp"
export RAY_TMPDIR="${HOME}/tmp"
export TRITON_HOME="${HOME}"

# Check for required environment variables
if [ -z "${MODEL_SIZE}" ]; then
    echo "Error: MODEL_SIZE environment variable required"
    exit 1
fi

# Print job info
# echo "Job ID: $SLURM_JOB_ID"
# echo "Node: $SLURM_NODELIST"
echo "Model Size: ${MODEL_SIZE}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# module load gcc/14
# source "${HOME}"

export DIRPATH_PROJECT="${HOME}/agentgym_rl"
export FPATH_SCRIPT_TRAIN="${DIRPATH_PROJECT}/examples/train/AgentGym-RL/sciworld_train.sh"

if [[ -f "${FPATH_SCRIPT_TRAIN}" ]]; then
    echo "Assertion passed: ${FPATH_SCRIPT_TRAIN} exists and is a regular file."
else
    echo "Assertion failed: ${FPATH_SCRIPT_TRAIN} does not exist or is not a regular file."
    exit 1 # Exit with a non-zero status to indicate failure
fi

# cd "${DIRPATH_PROJECT}"

# Activate conda environment
# conda activate agog

echo "CONDA_PREFIX=${CONDA_PREFIX}"


export PATH="${HOME}/miniforge3/bin/:${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export TMPDIR="${HOME}/tmp"


echo "ll PATH: $(ls -alt ${PATH})"
echo "which python: $(which python)"

# Start sciworld env server in background
# echo "Starting sciworld server on port ${PORT}..."
# sciworld --host localhost --port "${PORT}" &
# export SERVER_PID=$!
# echo "Server PID: ${SERVER_PID}"

# Wait for server to be ready
sleep 5
echo "Server should be ready..."
