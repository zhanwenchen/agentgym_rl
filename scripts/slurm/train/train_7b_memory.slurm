#!/bin/bash
#SBATCH --job-name=train_7b_n8_memory
#SBATCH --account=hartvigsen_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/train/log_%x_%j.log
#SBATCH --error=logs/train/log_%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pct4et@virginia.edu
#SBATCH --exclude=udc-an28-1,udc-an28-7,udc-an40-1


export MODEL_SIZE="7b"
export MEMORY_ENABLED=true
export MEMORY_K=3
export MEMORY_MIN_REWARD=0.5


source "${HOME}/agentgym_rl/scripts/slurm/train/train_shared.sh"
