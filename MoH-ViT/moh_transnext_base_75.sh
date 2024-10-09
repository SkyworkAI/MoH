#!/usr/bin/env bash

export NCCL_LL_THRESHOLD=0
DATA=$1

python -m torch.distributed.launch --nproc_per_node=8 --master_port=2013 \
    --use_env main.py --config ./configs/moh_transnext_base_75.py \
    --data-path $DATA --batch-size 128 \
    --output_dir results/moh_transnext_base_75 --num_workers 32