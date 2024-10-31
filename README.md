<div id="top" align="center">

# MedSora: Optical Flow Representation Alignment Mamba Diffusion Model for Medical Video Generation
  
  [Zhenbin Wang](https://github.com/wongzbb), Lei Zhang<sup>✉</sup>, [Lituan Wang](https://github.com/LTWangSCU), Minjuan Zhu, [Zhenwei Zhang](https://github.com/Zhangzw-99) </br>
  
  [![arXiv](https://img.shields.io/badge/arXiv-2406.15910-b31b1b.svg)](https://arxiv.org/abs/2406.15910)
 </br>
  


</div>

<!-- ## News🚀
(2024.xx.xx) ***The first edition of our paper has been uploaded to arXiv***🔥🔥

(2024.xx.xx) ***The project code has been uploaded to Github*** 🔥🔥 -->

## 🛠 Setup

```bash
git clone https://github.com/wongzbb/MedSora.git
cd MedSora
conda create -n MedSora python=3.11.0
conda activate MedSora

conda install cudatoolkit==11.7 -c nvidia
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc

pip install open_clip_torch loguru wandb diffusers einops omegaconf torchmetrics local_attention pyAV decord accelerate imageio-ffmpeg imageio pytest fvcore chardet yacs termcolor submitit tensorboardX seaborn lpips

# for official mamba
mkdir whl && cd whl
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && cd ..

# # for Chinese, you may need add
# export HF_ENDPOINT=https://hf-mirror.com
# ```
# ## 📚 Data Preparation
# **Cholec Triplet**:  You can directly use the [processed data](https://huggingface.co/datasets/ZhenbinWang/CholecTriplet_processed) without further data processing.
# ```
# huggingface-cli download --repo-type dataset --resume-download ZhenbinWang/CholecTriplet_processed --local-dir ./datasets/CholecTriplet_processed/
# ```
# **Colonoscopic**:   You can directly use the [processed data](https://huggingface.co/datasets/ZhenbinWang/Colonoscopic_processed) without further data processing.
# ```
# huggingface-cli download --repo-type dataset --resume-download ZhenbinWang/Colonoscopic_processed --local-dir ./datasets/Colonoscopic_processed/
# ```
# **Kvasir Capsule**:   You can directly use the [processed data](https://huggingface.co/datasets/ZhenbinWang/Kvasir_Capsule_processed) without further data processing.
# ```
# huggingface-cli download --repo-type dataset --resume-download ZhenbinWang/Kvasir_Capsule_processed --local-dir ./datasets/Kvasir_Capsule_processed/
# ```

# ## 🎇 Sampling
# You can directly sample the video from the checkpoint model. Here is an example for quick usage for using our **pre-trained models**:
# 1. Download the pre-trained weights from [here]()(coming soon).
# 2. Run [`sample.py`](sample.py) by the following scripts to customize the various arguments.
# ```
# #CUDA_VISIBLE_DEVICES=0 torchrun --master_port=12345 --nnodes=1 --nproc_per_node=1 sample.py \
#     --model DiM-B/2 \
#     --sample-global-batch-size 10 \
#     --config config/cho/cho_train.yaml \
#     --what2video N2V \
#     --ckpt results/cho/139-DiM-B-2/checkpoints/0240000.pt
# ```

# ## ⏳ Training
# The weight of pretrained MedSora can be found [here]()(coming soon), and in our implementation we use DiM-B/2 during training MedSora.
# Train MedSora with `N` GPUs.
# ```
# CUDA_VISIBLE_DEVICES=your_gpu_id torchrun --master_port=12345 --nnodes=1 --nproc_per_node=N train.py \
#   --model DiM-B/2 \
#   --epoch 300 \
#   --global-batch-size 2 \
#   --config config/cho/cho_train.yaml \
#   --what2video N2V \
#   --wandb \
#   --autocast
# ```
# - `--autocast`: This option enables half-precision training for the model. We recommend disabling it, as it is prone to causing NaN errors. To do so, simply remove this option from the command line.