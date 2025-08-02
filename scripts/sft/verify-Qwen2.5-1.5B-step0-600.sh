#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.

export WANDB_API_KEY="004ba186f7e1f9bd08fe620ddeaaf98ef356c95f"
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/weka/home/yongxin.wang/workspace/yangzhch6/models/Qwen/Qwen2.5-1.5B"
export SAVE_PATH="/mnt/weka/home/yongxin.wang/workspace/yangzhch6/rvlr-baseline/checkpoints/verify/Qwen2.5-1.5B-step-0-600"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Train over a single node, 1 A100-80GB GPUs.
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/weka/home/yongxin.wang/workspace/yangzhch6/RLVR-Verify-Data/sft_critique/Qwen2.5-1.5B-openr1/0-600.parquet \
    data.val_files=/mnt/weka/home/yongxin.wang/workspace/yangzhch6/RLVR-Verify-Data/sft_critique/Qwen2.5-1.5B-openr1/0-600_val.parquet \
    data.prompt_key=input \
    data.response_key=output \
    +data.use_template=False \
    data.max_length=8192 \
    data.train_batch_size=1024 \
    data.micro_batch_size=2 \
    optim.lr=1e-6 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=verify \
    trainer.experiment_name=Qwen2.5-1.5B-step-0-600 \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true