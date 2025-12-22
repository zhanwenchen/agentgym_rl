#!/bin/bash
# AgentGym-RL Setup Script (Minimal Pipeline)
# This version has verl and vllm dependencies removed

# Note: This script is for historical reference only.
# The full training pipeline with verl and vllm has been removed.
# Only core utilities remain.

echo "=================================================="
echo "AgentGym-RL Minimal Pipeline"
echo "=================================================="
echo ""
echo "The verl training framework and vllm dependencies"
echo "have been removed from this repository."
echo ""
echo "What remains:"
echo "  - Core utilities in agentgym_rl_utils/"
echo "  - AgentGym environment integration"
echo "  - Documentation for reference"
echo ""
echo "For the full training pipeline, please refer to:"
echo "  - Original verl: https://github.com/volcengine/verl"
echo "  - Historical versions of this repository"
echo ""
echo "=================================================="

# Basic setup for the minimal utilities
conda create -n agentgym-rl-minimal python=3.10 numpy pandas -y
conda activate agentgym-rl-minimal

# Install minimal dependencies
pip install -e AgentGym-RL

# Optional: Install AgentGym if needed
# pip install -e AgentGym/agentenv
# pip install -e AgentGym/agentenv-sciworld

echo ""
echo "Minimal setup complete!"
echo "You can now use the standalone utilities in agentgym_rl_utils/"
