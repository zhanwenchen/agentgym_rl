# run.sh

conda create -n ag python numpy pandas ninja cmake setuptools_scm # triton # triton messes up the nvcc bin to 12.8
# conda create -n ag python=13 numpy pandas ninja psutil gcc=14.* gxx=14.* cmake setuptools_scm # triton # triton messes up the nvcc bin to 12.8
conda create -n ag python=13 numpy pandas ninja psutil cmake setuptools_scm # triton # triton messes up the nvcc bin to 12.8
conda activate ag
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install torch torchvision # for 12.8

screen -S flash_attn
conda activate ag
export TMPDIR="/home/pct4et/tmp"
module load gcc/14
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache
pip install transformers datasets wandb safetensors tensordict ray codetiming omegaconf hydra-core
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install -e AgentGym-RL
pip install -e AgentGym/agentenv
pip install -e AgentGym/agentenv-sciworld

screen -S sw
# sciworld --host 0.0.0.0 --port 36001
sciworld --host localhost --port 36001
# exit screen with Ctrl+A D


screen -S ag
RAY_DEBUG=legacy bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_train.log
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install --no-build-isolation .

export HF_ENDPOINT="https://hf-mirror.com"

HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct
RAY_DEBUG=legacy bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_train.log



# og:
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/timezone
sudo apt update
sudo apt full-upgrade
sudo apt install screen openjdk-17-jdk iproute2
ssh-keygen
cat ~/.ssh/id_rsa.pub  # add to github
git clone git@github.com:zhanwenchen/agentgym_rl.git
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh
echo "Preparing environment for agentgym-rl..."
conda create -n agog "python<3.11" -y
conda activate agog
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# install flash-atten
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME
# for RL
cd agentgym_rl
pip install -e AgentGym-RL
git clone git@github.com:zhanwenchen/AgentGym.git
pip install -e AgentGym/agentenv
pip install -e AgentGym/agentenv-sciworld
# pip install vllm==0.6.3 transformers==4.51.3 tokenizers huggingface_hub peft==0.17.1 faiss-gpu-cu12 sentence_transformers click==8.0.1 weave --no-build-isolation
pip install vllm==0.6.3 transformers==4.51.3 tokenizers huggingface_hub peft==0.17.1 faiss-gpu-cu12 sentence_transformers click==8.0.1 --no-build-isolation
# No longer install/import weave because of # ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /root/miniforge3/envs/agog/lib/python3.10/lib-dynload/../.././libicui18n.so.78) weave
wandb login # https://wandb.ai/authorize
ll "${CONDA_PREFIX}/lib/python3.10/site-packages/vllm/"
ll "${CONDA_PREFIX}/lib/python3.10/site-packages/vllm/version.py"


# try:
#     # from ._version import __version__, __version_tuple__
#     __version__ = '0.6.3'
#     __version_tuple__ = (0, 6, 3)
# except Exception as e:


cd "${HOME}/agentgym_rl"
export HF_ENDPOINT="https://hf-mirror.com"
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

export CUDA_HOME="${CONDA_PREFIX}"

# Use scratch for models

# export DIRPATH_SAVES_SCRATCH="/scratch/${USER}/agentgym_rl_AgentGym-RL_saves"
# mkdir -p "${DIRPATH_SAVES_SCRATCH}"
# ln -sf "${DIRPATH_SAVES_SCRATCH}" "${HOME}/agentgym_rl/AgentGym-RL/saves"
export DIRPATH_SAVES="${HOME}/shared-storage/agentgym_rl_AgentGym-RL_saves"
mkdir -p "${DIRPATH_SAVES}"
ln -sf "${DIRPATH_SAVES}" "${HOME}/workspace/agentgym_rl/AgentGym-RL/saves"


# # Use scratch for saves
# export DIRPATH_MODELS_SCRATCH="/scratch/${USER}/agentgym_rl_models"
# mkdir -p "${DIRPATH_MODELS_SCRATCH}"
# ln -sf "${DIRPATH_MODELS_SCRATCH}" "${HOME}/agentgym_rl/models"

# bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb.log
# bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb.log
# RAY_DEBUG=legacy bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb_3b_n8.log
# bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb_14b_n8.log

scp -r AgentGym-RL/AgentItemId riv:~/agentgym_rl/AgentGym-RL

# Lastly, don't forget to modify
# vim /home/pct4et/envs/agog/lib/python3.10/site-packages/vllm/version.py
# to be
# try:
#     __version__ = '0.6.3'
#     __version_tuple__ = (0, 6, 3)
# except Exception as e:
#     import warnings

#     warnings.warn(f"Failed to read commit hash:\n{e}",
#                   RuntimeWarning,
#                   stacklevel=2)

#     __version__ = "dev"
#     __version_tuple__ = (0, 0, __version__)


# conda remove gcc gxx libstdcxx-ng # nope. Need to module load gcc/14
pip install click==8.0.1 weave # click 8.3* breaks ray
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.13.1
bash examples/eval/sciworld_eval.sh |& tee log_eval_512gb_80gb_3b_n8.log
pip install faiss-gpu-cu12 sentence_transformers
bash scripts/local/train/train_3b_memory.sh
