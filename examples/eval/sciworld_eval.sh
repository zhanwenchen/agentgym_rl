set -euo pipefail
set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="sciworld"

# cd AgentGym-RL
# source activate
# conda activate agentgym-rl

env_server_url="http://localhost:${PORT}"

sample_num=1
max_rounds=30
export batch_size=1
export max_response_length=8192 # orginally 8192
export gpu_memory_utilization=0.70 # originally 0.95 # 0.5 works on most. 0.70 works on a800_80gb.

# ckpt_path="global_step_150/actor"
model_path=${ckpt_path}/huggingface

# cd AgentGym-RL/scripts

if [[ "${ckpt_path}" == *"global_step_"* ]]; then
    echo "Using finetuned model ${ckpt_path}, merging with base model."
    "${PYTHON_BIN}" "${DIRPATH_PROJECT}/AgentGym-RL/scripts/model_merger.py" --local_dir "${ckpt_path}"
    echo "Using finetuned model ${ckpt_path}, merging complete."
else
    echo "Using pretrained model, no need to merge."
fi


    # data.path=AgentEval/${task_name} \
HYDRA_FULL_ERROR=1 "${PYTHON_BIN}" -m verl.agent_trainer.main_generation  \
    data.path="${DIRPATH_PROJECT}/AgentGym-RL/AgentItemId/eval" \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    data.n_samples=${sample_num} \
    data.batch_size=${batch_size} \
    agentgym.task_name=${task_name} \
    agentgym.env_addr=${env_server_url} \
    agentgym.max_rounds=${max_rounds} \
    agentgym.timeout=500 \
    model.path=${model_path} \
    rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    rollout.temperature=1 \
    rollout.max_model_len=32768 \
    rollout.max_tokens=200 \
    rollout.tensor_model_parallel_size=1 \
    rollout.memory.enabled="${MEMORY_ENABLED}" \
    rollout.memory.k="${MEMORY_K}" \
    rollout.memory.min_reward="${MEMORY_MIN_REWARD}" \
    rollout.memory.distance_metric="${MEMORY_DISTANCE_METRIC}" \
    rollout.memory.save_path="${ckpt_path}/memory_bank" \
    rollout.rollout_log_dir="${ckpt_path}/executor_logs/$(date +%Y%m%d%H%M%S)" \
    "$@"

status=$?
exit $status

# bash examples/eval/AgentGym-RL/sciworld_eval.sh |& tee log_eval_512gb_80gb_3b_n8.log
