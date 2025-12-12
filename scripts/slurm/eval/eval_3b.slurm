#!/bin/bash
#SBATCH --job-name=eval_3b_n8
#SBATCH --account=hartvigsen_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/eval/log_%x_%j.log
#SBATCH --error=logs/eval/log_%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pct4et@virginia.edu
#SBATCH --exclude=udc-an28-1,udc-an28-7,udc-an40-1


export MODEL_SIZE="3b"
export ckpt_path="${HOME}/agentgym_rl/AgentGym-RL/saves/${MODEL_SIZE}_n8/global_step_125/actor"

source "${HOME}/agentgym_rl/scripts/slurm/eval/eval_shared.sh"
