# Training and Evaluation Scripts - Removed

The following scripts have been removed as they depended on the verl training framework:

## Training Scripts (examples/train/)
- `AgentGym-RL/babyai_train.sh` - Used `verl.agent_trainer.main_ppo`
- `AgentGym-RL/sciworld_train.sh` - Used `verl.agent_trainer.main_ppo`
- `AgentGym-RL/searchqa_train.sh` - Used `verl.agent_trainer.main_ppo`
- `AgentGym-RL/textcraft_train.sh` - Used `verl.agent_trainer.main_ppo`
- `AgentGym-RL/webarena_train.sh` - Used `verl.agent_trainer.main_ppo`
- `ScalingInter-RL/*_train.sh` - All used `verl.agent_trainer.main_ppo`

## Evaluation Scripts (examples/eval/)
- `babyai_eval.sh` - Used `verl.agent_trainer.main_generation`
- `sciworld_eval.sh` - Used `verl.agent_trainer.main_generation`
- `searchqa_eval.sh` - Used `verl.agent_trainer.main_generation`
- `textcraft_eval.sh` - Used `verl.agent_trainer.main_generation`
- `webarena_eval.sh` - Used `verl.agent_trainer.main_generation`

## Reason for Removal

All these scripts invoked Python modules from the verl framework:
- `python3 -m verl.agent_trainer.main_ppo` for training
- `python3 -m verl.agent_trainer.main_generation` for evaluation

Since the entire verl directory and its dependencies (Ray, vllm, Hydra, etc.) have been removed 
to minimize the pipeline, these scripts are no longer functional.

## Alternative

For users needing the full RL training pipeline, please refer to:
1. The original verl repository at https://github.com/volcengine/verl
2. Historical versions of this repository before the refactoring

The core AgentGym environment remains available in the `AgentGym` directory.
