> The official implemention of [Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration](https://arxiv.org/abs/2508.13755v1)

## Prepare Data
...

## Install

```
pip install -e ./verl
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
pip install -e .
```


You may need to install flash-attention through:
```
https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
```

## Run DARS-B
```
#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export WANDB_API_KEY="004ba186f7e1f9bd08fe620ddeaaf98ef356c95f"
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="$PATH/models/Qwen/Qwen2.5-Math-1.5B"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Train over a single node, 1 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo_dars \
    algorithm.adv_estimator=grpo \
    data.train_files=$PATH/RLVR-Data/nothink/openr1.parquet \
    data.val_files=$PATH/RLVR-Data/nothink/capacity_train_and_val.parquet \
    data.train_batch_size=3072 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.shuffle=False \
    +data.resampling_func=1 \
    +data.use_template=True \
    +data.reward_impl_version=2 \
    +actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=3072 \
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
    actor_rollout_ref.rollout.val_temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    +algorithm.grpo_use_std=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='DARS' \
    trainer.experiment_name='Qwen2.5-Math-1.5B-openr1-nothink-3k-func1-bs3k-pp2' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=14 \
    trainer.test_freq=14 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=142 \
    trainer.total_epochs=30 "${@:1}"
```


## Run grpo baseline
refer to [https://github.com/yangzhch6/rlvr-baseline](https://github.com/yangzhch6/rlvr-baseline)