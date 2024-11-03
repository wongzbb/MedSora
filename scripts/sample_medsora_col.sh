#!/bin/bash
cudaID=3
port=12749
nnodes=1
nproc_per_node=1
config=./configs/col/col_train.yaml
ptxasPath=/opt/conda/envs/test_sora/bin/ptxas  # run `which ptxas` to get the path

model=MedSora-B

ckpt=/root/code/MedSora/results/col/019-MedSora-B-2/checkpoints/0740000.pt # path to the checkpoint


CUDA_VISIBLE_DEVICES=$cudaID \
TRITON_PTXAS_PATH=$ptxasPath \
torchrun \
    --master_port=$port \
    --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    sample.py \
    --model $model \
    --config $config \
    --use-local-cov \
    --ckpt $ckpt