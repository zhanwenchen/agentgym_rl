# run.sh

# zip
# zip -r agentgym_rl.zip agentgym_rl/ -x '*.git*' -x '*.safetensors'

# System
sudo apt update
sudo apt full-upgrade
sudo apt install -y screen openjdk-17-jdk iproute2 rsync tk tcllib # type 2 when asked about sshd config.

# CUDA
export DIRNAME_MOUNT="d524mtfhri0c73avm3ig"
cd "${HOME}/workspace/${DIRNAME_MOUNT}/downloads/cuda"

# 1. compilers
sudo apt install -y ./cuda-cuobjdump-12-4_12.4.99-1_amd64.deb ./cuda-cuxxfilt-12-4_12.4.99-1_amd64.deb ./cuda-toolkit-config-common_12.4.99-1_all.deb ./cuda-toolkit-12-config-common_12.4.99-1_all.deb ./cuda-toolkit-12-4-config-common_12.4.99-1_all.deb ./cuda-cudart-12-4_12.4.99-1_amd64.deb ./cuda-cccl-12-4_12.4.99-1_amd64.deb ./cuda-driver-dev-12-4_12.4.99-1_amd64.deb ./cuda-cudart-dev-12-4_12.4.99-1_amd64.deb ./cuda-nvvm-12-4_12.4.99-1_amd64.deb ./cuda-crt-12-4_12.4.99-1_amd64.deb ./cuda-nvcc-12-4_12.4.99-1_amd64.deb ./cuda-nvprune-12-4_12.4.99-1_amd64.deb ./cuda-compiler-12-4_12.4.0-1_amd64.deb
# 2. cuda-libraries-12-4
sudo apt install -y ./cuda-nvrtc-12-4_12.4.99-1_amd64.deb ./cuda-opencl-12-4_12.4.99-1_amd64.deb ./libcublas-12-4_12.4.2.65-1_amd64.deb ./libcufft-12-4_11.2.0.44-1_amd64.deb ./libcufile-12-4_1.9.0.20-1_amd64.deb ./libcurand-12-4_10.3.5.119-1_amd64.deb ./libcusolver-12-4_11.6.0.99-1_amd64.deb ./libcusparse-12-4_12.3.0.142-1_amd64.deb ./libnpp-12-4_12.2.5.2-1_amd64.deb ./libnvjitlink-12-4_12.4.99-1_amd64.deb ./libnvfatbin-12-4_12.4.99-1_amd64.deb ./libnvjpeg-12-4_12.3.1.89-1_amd64.deb ./cuda-libraries-12-4_12.4.0-1_amd64.deb
# 3. ./cuda-libraries-dev-12-4_12.4.0-1_amd64.deb
sudo apt install -y ./cuda-profiler-api-12-4_12.4.99-1_amd64.deb ./cuda-nvrtc-dev-12-4_12.4.99-1_amd64.deb ./cuda-opencl-dev-12-4_12.4.99-1_amd64.deb ./libcublas-dev-12-4_12.4.2.65-1_amd64.deb ./libcufft-dev-12-4_11.2.0.44-1_amd64.deb ./libcufile-dev-12-4_1.9.0.20-1_amd64.deb ./libcurand-dev-12-4_10.3.5.119-1_amd64.deb ./libcusolver-dev-12-4_11.6.0.99-1_amd64.deb ./libcusparse-dev-12-4_12.3.0.142-1_amd64.deb ./libnpp-dev-12-4_12.2.5.2-1_amd64.deb ./libnvjitlink-dev-12-4_12.4.99-1_amd64.deb ./libnvfatbin-dev-12-4_12.4.99-1_amd64.deb ./libnvjpeg-dev-12-4_12.3.1.89-1_amd64.deb ./cuda-libraries-dev-12-4_12.4.0-1_amd64.deb
# 4. ./cuda-tools-12-4_12.4.0-1_amd64.deb
sudo apt install -y ./cuda-cupti-12-4_12.4.99-1_amd64.deb ./cuda-cupti-dev-12-4_12.4.99-1_amd64.deb ./cuda-nvdisasm-12-4_12.4.99-1_amd64.deb ./cuda-gdb-12-4_12.4.99-1_amd64.deb ./cuda-nvprof-12-4_12.4.99-1_amd64.deb ./cuda-nvtx-12-4_12.4.99-1_amd64.deb ./cuda-sanitizer-12-4_12.4.99-1_amd64.deb ./cuda-command-line-tools-12-4_12.4.0-1_amd64.deb ./nsight-compute-2024.1.0_2024.1.0.13-1_amd64.deb ./cuda-nsight-compute-12-4_12.4.0-1_amd64.deb ./nsight-systems-2023.4.4_2023.4.4.54-1_amd64.deb ./cuda-nsight-systems-12-4_12.4.0-1_amd64.deb ./cuda-nsight-12-4_12.4.99-1_amd64.deb ./cuda-nvml-dev-12-4_12.4.99-1_amd64.deb ./cuda-nvvp-12-4_12.4.99-1_amd64.deb ./cuda-visual-tools-12-4_12.4.0-1_amd64.deb ./gds-tools-12-4_1.9.0.20-1_amd64.deb ./cuda-tools-12-4_12.4.0-1_amd64.deb
# 5. ./cuda-documentation-12-4_12.4.99-1_amd64.deb
sudo apt install -y ./cuda-documentation-12-4_12.4.99-1_amd64.deb
# 6. ./cuda-toolkit-12-4_12.4.0-1_amd64.deb. FINALLY
sudo apt install -y ./cuda-toolkit-12-4_12.4.0-1_amd64.deb


# og:
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/timezone


# Globus Personal Connect requires non-root user account
sudo useradd -m -s $(which bash) -G sudo zhanwen
sudo bash -c "echo 'zhanwen ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/zhanwen"
sudo chmod 0440 /etc/sudoers.d/zhanwen
# sudo update-alternatives --set editor /usr/bin/vim.basic
# sudo groupadd admin
# sudo usermod zhanwen -g admin
# sudo su - zhanwen
# sudo visudo
# sudo echo >> /etc/sudoers.d/zhanwen


sudo su - zhanwen # need to download as 
wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf globusconnectpersonal-latest.tgz
cd globusconnectpersonal-3.2.8/
# ./globusconnectpersonal
./globusconnectpersonal -start &
echo '/root,0,1' >> ${HOME}/.globusonline/lta/config-paths
sudo chmod o+x,o+r /root
sudo chmod o+x,o+r,o+w -R /root/workspace/agentgym_rl
exit
# sudo mkdir -p "/home/zhanwen/.globusonline/lta/"
# sudo chown zhanwen /home/zhanwen/.globusonline/lta/config-paths
# ~/.globusonline/lta/config-paths # add line "/root,0,1"
# ./globusconnectpersonal -stop
# sudo -H -u zhanwen bash -c './globusconnectpersonal -start &' 

ssh-keygen
cat ~/.ssh/id_rsa.pub  # add to github
git clone git@github.com:zhanwenchen/agentgym_rl.git
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh
# echo "Preparing environment for agentgym-rl..."


conda create -n agog "python<3.11" -y
conda activate agog

export DIRNAME_MOUNT="d524mtfhri0c73avm3ig"
cd "${HOME}/workspace/${DIRNAME_MOUNT}/downloads"
pip install torch-2.4.0+cu124-cp310-cp310-linux_x86_64.whl flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
# pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# pip install torch-2.4.0+cu124-cp310-cp310-linux_x86_64.whl --no-build-isolation
# pip install flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
# install flash-atten
# export FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# export FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# wget "${FLASH_ATTENTION_URL}" -O "${FLASH_ATTENTION_NAME}"
# pip install $FLASH_ATTENTION_NAME
# rm -f $FLASH_ATTENTION_NAME
# for RL
cd ${HOME}/workspace/agentgym_rl
# pip install -e AgentGym-RL --no-build-isolation
git clone git@github.com:zhanwenchen/AgentGym.git

pip install -e AgentGym-RL -e AgentGym/agentenv -e AgentGym/agentenv-sciworld --no-build-isolation
# pip install -e AgentGym/agentenv-sciworld --no-build-isolation
# pip install vllm==0.6.3 transformers==4.51.3 tokenizers huggingface_hub peft==0.17.1 faiss-gpu-cu12 sentence_transformers click==8.0.1 weave --no-build-isolation
pip install vllm==0.6.3 transformers==4.51.3 tokenizers huggingface_hub peft==0.17.1 faiss-gpu-cu12 sentence_transformers click==8.0.1 --no-build-isolation
# No longer install/import weave because of # ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /root/miniforge3/envs/agog/lib/python3.10/lib-dynload/../.././libicui18n.so.78) weave
wandb login # https://wandb.ai/authorize
ll "${CONDA_PREFIX}/lib/python3.10/site-packages/vllm/version.py"


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



cd "${HOME}/workspace/agentgym_rl"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_XET_HIGH_PERFORMANCE=1
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

# export CUDA_HOME="${CONDA_PREFIX}"

# # use shared-storage for models
# export DIRPATH_SAVES="${HOME}/shared-storage/agentgym_rl_AgentGym-RL_saves"
# mkdir -p "${DIRPATH_SAVES}"
# ln -sf "${DIRPATH_SAVES}" "${HOME}/workspace/agentgym_rl/AgentGym-RL/saves"

# Use scratch for models

# export DIRPATH_SAVES_SCRATCH="/scratch/${USER}/agentgym_rl_AgentGym-RL_saves"
# mkdir -p "${DIRPATH_SAVES_SCRATCH}"
# ln -sf "${DIRPATH_SAVES_SCRATCH}" "${HOME}/agentgym_rl/AgentGym-RL/saves"


# # Use scratch for saves
# export DIRPATH_MODELS_SCRATCH="/scratch/${USER}/agentgym_rl_models"
# mkdir -p "${DIRPATH_MODELS_SCRATCH}"
# ln -sf "${DIRPATH_MODELS_SCRATCH}" "${HOME}/agentgym_rl/models"

# bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb.log
# bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb.log
# RAY_DEBUG=legacy bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb_3b_n8.log
# bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb_14b_n8.log

# scp -r AgentGym-RL/AgentItemId riv:~/agentgym_rl/AgentGym-RL

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
# pip install click==8.0.1 weave # click 8.3* breaks ray
# conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.13.1
# bash examples/eval/sciworld_eval.sh |& tee log_eval_512gb_80gb_3b_n8.log
# pip install faiss-gpu-cu12 sentence_transformers
# bash scripts/local/train/train_3b_memory.sh
