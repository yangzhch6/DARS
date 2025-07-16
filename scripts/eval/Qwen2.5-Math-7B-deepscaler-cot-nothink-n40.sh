#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/hpc2hdd/home/zyang398/yangzhch6/models/Qwen/Qwen2.5-Math-7B"
export CUDA_VISIBLE_DEVICES=0,1

# Train over a single node, 1 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/hpc2hdd/home/zyang398/yangzhch6/projs/RLVR-Data/nothink/deepscaler_cot.parquet \
    data.val_files=/hpc2hdd/home/zyang398/yangzhch6/projs/RLVR-Data/nothink/deepscaler_cot.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    +data.reward_impl_version=2 \
    +actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=40 \
    actor_rollout_ref.rollout.n_val=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    +algorithm.grpo_use_std=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='eval' \
    trainer.experiment_name='Qwen2.5-Math-7B-deepscaler_cot-nothink-3k-n40' \
    +trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}"