#!/bin/bash
cpuNum=4
cudaID=3
port=12749
nnodes=1
nproc_per_node=1
config=./configs/col/col_train.yaml
ptxasPath=/opt/conda/envs/test_sora/bin/ptxas  # run `which ptxas` to get the path

model=MedSora-B
epoch=300
batchSize=2


OMP_NUM_THREADS=$cpuNum \
CUDA_VISIBLE_DEVICES=$cudaID \
TRITON_PTXAS_PATH=$ptxasPath \
torchrun \
    --master_port=$port \
    --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    train.py \
    --model $model \
    --epoch $epoch \
    --global-batch-size $batchSize \
    --config $config \
    --use-local-cov