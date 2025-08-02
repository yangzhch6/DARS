#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.

export WANDB_API_KEY="004ba186f7e1f9bd08fe620ddeaaf98ef356c95f"
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/weka/home/yongxin.wang/workspace/yangzhch6/models/Qwen/Qwen2.5-1.5B"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Train over a single node, 1 A100-80GB GPUs.
torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/weka/home/yongxin.wang/workspace/yangzhch6/RLVR-Verify-Data/sft_critique/Qwen2.5-1.5B-openr1/0-100.parquet \
    data.val_files=/mnt/weka/home/yongxin.wang/workspace/yangzhch6/RLVR-Verify-Data/sft_critique/Qwen2.5-1.5B-openr1/0-100_val.parquet \
    data.prompt_key=input \
    data.response_key=output \
    +data.use_template=False \
    data.max_length=8192 \
    data.train_batch_size=1024 \
    data.micro_batch_size=4 \
    optim.lr=1e-6 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.project_name=sft-debug \
    trainer.experiment_name=Qwen2.5-1.5B-sft-100 \
    trainer.total_epochs=1 \
    trainer.logger='["console","wandb"]'