#!/bin/bash

# Check for eval-specific environment variables
if [ -z "${ckpt_path}" ]; then
    echo "Error: ckpt_path environment variable required"
    exit 1
fi

# Source common setup
source "${DIRPATH_PROJECT}/scripts/damodel/shared_setup.sh"

# Export eval-specific variables
echo "Checkpoint Path: ${ckpt_path}"

echo "Starting eval..."

# Run eval
bash examples/eval/sciworld_eval.sh 2>&1
EVAL_EXIT_CODE=$?

# Cleanup: kill the server
echo "Eval finished with exit code: $EVAL_EXIT_CODE"
echo "Killing server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null

echo "End time: $(date)"
# exit $EVAL_EXIT_CODE
