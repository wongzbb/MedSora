<div id="top" align="center">

# MedSora: Optical Flow Representation Alignment Mamba Diffusion Model for Medical Video Generation
  [Zhenbin Wang](https://github.com/wongzbb), Lei Zhang<sup>✉</sup>, [Lituan Wang](https://github.com/LTWangSCU), Minjuan Zhu, [Zhenwei Zhang](https://github.com/Zhangzw-99) </br>
  [![arXiv]()](https://arxiv.org/abs/2406.15910)
  </br>
</div>



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

# for Chinese, you may need
export HF_ENDPOINT=https://hf-mirror.com
```
## 📚 Data Preparation
**Cholec Triplet**:  You can directly use the [processed data](https://huggingface.co/datasets/ZhenbinWang/CholecTriplet_processed) without further data processing.
```
huggingface-cli download --repo-type dataset --resume-download ZhenbinWang/CholecTriplet_processed --local-dir ./datasets/CholecTriplet_processed/
```
**Colonoscopic**:   You can directly use the [processed data](https://huggingface.co/datasets/ZhenbinWang/Colonoscopic_processed) without further data processing.
```
huggingface-cli download --repo-type dataset --resume-download ZhenbinWang/Colonoscopic_processed --local-dir ./datasets/Colonoscopic_processed/
```
**Kvasir Capsule**:   You can directly use the [processed data](https://huggingface.co/datasets/ZhenbinWang/Kvasir_Capsule_processed) without further data processing.
```
huggingface-cli download --repo-type dataset --resume-download ZhenbinWang/Kvasir_Capsule_processed --local-dir ./datasets/Kvasir_Capsule_processed/
```

## 🎇 Sampling


## ⏳ Training
