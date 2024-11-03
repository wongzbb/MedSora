#!/bin/bash
cudaID=3
port=12749
nnodes=1
nproc_per_node=1
config=./configs/kva/kva_train.yaml
ptxasPath=/opt/conda/envs/test_sora/bin/ptxas  # run `which ptxas` to get the path

model=MedSora-B

ckpt=... # path to the checkpoint


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