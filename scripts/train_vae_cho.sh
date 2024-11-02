#!/bin/bash
cudaID=0
port=12749
nnodes=1
nproc_per_node=1
config=./configs/cho/cho_train.yaml

CUDA_VISIBLE_DEVICES=$cudaID torchrun \
    --master_port=$port \
    --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    train_vae.py \
    --config $config
