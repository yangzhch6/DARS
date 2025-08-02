#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export WANDB_API_KEY="004ba186f7e1f9bd08fe620ddeaaf98ef356c95f"
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/weka/home/yongxin.wang/workspace/yangzhch6/rvlr-baseline/checkpoints/sft_critique/Qwen2.5-7B-openr1-step-100/global_step_41"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Train over a single node, 1 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/weka/home/yongxin.wang/workspace/yangzhch6/RLVR-Data/no_template/nothink/openr1.parquet \
    data.val_files=/mnt/weka/home/yongxin.wang/workspace/yangzhch6/RLVR-Data/no_template/nothink/valid.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=2560 \
    data.max_response_length=3072 \
    +data.use_template=False \
    +data.reward_impl_version=3 \
    +actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    +algorithm.clip_adv_value=1.0 \
    +algorithm.grpo_use_std=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rl-critique' \
    trainer.experiment_name='Qwen2.5-7B-openr1-nothink-3k' \
    +trainer.val_before_train=True \
    +trainer.init_global_steps=100 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=105 \
    trainer.total_epochs=30 "${@:1}"