# 针对Openpangu的环境安装
文档用于介绍针对Openpangu模型的环境配置
## Openpangu下载
如果希望使用openPangu-Embedded-7B-V1.1，请参考[文档](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1)。
## 镜像使用
如果希望使用基于 `Openpangu` 构建的镜像，请下载 [网盘链接](https://github.com/volcengine/verl/tree/main/docs/ascend_tutorial/dockerfile_build_guidance.rst) 。

## 安装基础环境
基础环境涉及以下软件包，请参考 [文档](https://gitcode.com/Ascend/pytorch) 安装。

| 软件包 (Software) | 版本 (Version) |
| :--- | :--- |
| Python | >= 3.10, <3.12 |
| CANN | == 8.1.RC1 |
| torch | == 2.5.1 |
| torch_npu | == 2.5.1.post1 |

## 安装其他软件包

基础环境准备完毕后，需要通过指令安装以下软件包：
| 软件包 (Software) | 版本 (Version) |
| :--- | :--- |
| torchvision | == 0.20.1|
| triton-ascend | == 3.2.0rc4 |
| transformers  | == 4.53.2 |

安装指令：

    pip install torchvision==0.20.1
    pip install transformers==4.53.2

    # 安装triton-ascend，不需要单独安装triton
    pip install triton-ascend==3.2.0rc4

## 安装 vllm & vllm-ascend
### vllm 源码安装指令：

    git clone --depth 1 --branch v0.9.2 https://github.com/vllm-project/vllm.git
    cd vllm && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..

### vllm-ascend 源码安装指令：
    git clone --depth 1 --branch v0.9.2rc1 https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend && pip install -v -e . && cd ..
### 参考 [文档](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1/blob/main/inference/vllm_ascend_for_openpangu_embedded_7b.zh.md) 更改 vllm-ascend 适配 Openpangu
下载 [链接](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1/tree/main/inference) 中适配盘古模型的vllm-ascend代码，替换原本的vllm_ascend中的部分代码

    yes | cp -r ~/inference/vllm_ascend/* ~/vllm_ascend/vllm_ascend/
## verl安装
### 从项目代码中下载DARS Openpangu版本的verl文件

    cd dars
    pip install -e ./verl

### 其余软件包
   根据Torch版本[主文档](https://github.com/yangzhch6/DARS)安装deepscaler

# 训练与测试
## 模型训练
    #!/bin/bash
    set -x
    # bash /data1/lhh/expriments/OP-7B-ET-Breadth.sh
    export HF_HOME="/root/.cache/huggingface"
    export PYTHONPATH=/root/.cache/huggingface/modules:/usr/local/Ascend/ascend-toolkit/8.1.RC1/python/site-packages:$PYTHONPATH
    DATA_PATH="/data1/lhh/data/dars"
    MODEL_PATH="/opt/pangu/openPangu-Embedded-7B-V1.1"

    python3 -m verl.trainer.main_ppo_dars \
        algorithm.adv_estimator=grpo \
        data.train_files=$DATA_PATH/openr1.parquet \
        data.val_files=$DATA_PATH/capacity_val.parquet \
        data.train_batch_size=1024 \
        data.val_batch_size=64 \
        data.max_prompt_length=1024 \
        data.filter_overlong_prompts=True\
        data.max_response_length=6144 \
        data.shuffle=False \
        data.resampling_func=1 \
        data.reward_impl_version=2 \
        actor_rollout_ref.actor.ppo_epochs=2 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=5e-6 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=14336 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.000 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
        actor_rollout_ref.rollout.data_parallel_size=1\
        actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.000 \
        +algorithm.n_max=32 \
        +algorithm.grpo_use_std=False \
        trainer.critic_warmup=0 \
        +trainer.del_last_ckpt=False \
        +trainer.log_train=True \
        trainer.project_name='DARS' \
        trainer.experiment_name='DARS-7B-ET-Breadth' \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=44 \
        trainer.test_freq=-1 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=6 \
        trainer.logger=console \
        reward_model.model.trust_remote_code=True\
        critic.model.trust_remote_code=True\
        data.trust_remote_code=True\
        actor_rollout_ref.model.trust_remote_code=True\
        trainer.device=npu 
        "${@:1}"
## 模型测试
    #!/bin/bash
    set -x
    # bash /data1/lhh/expriments/Pangu_test.sh
    export HF_HOME="/root/.cache/huggingface"
    export PYTHONPATH=/root/.cache/huggingface/modules:/usr/local/Ascend/ascend-toolkit/8.1.RC1/python/site-packages:$PYTHONPATH

    DATA_PATH="/data1/lhh/data/dars"
    MODEL_PATH="/data1/lhh/save_op_model"
    OUTPUT_PATH="/data1/lhh/result/OP_7B_Dars/Pass32"

    python3 -m verl.trainer.main_test \
        algorithm.adv_estimator=grpo \
        data.train_files="${DATA_PATH}/train.parquet" \
        data.val_files="${DATA_PATH}/test.parquet" \
        data.train_batch_size=12 \
        data.val_batch_size=512 \
        data.max_prompt_length=1024 \
        data.filter_overlong_prompts=True\
        data.max_response_length=8192 \
        data.shuffle=False \
        data.validation_shuffle=False \
        data.resampling_func=1 \
        data.reward_impl_version=2 \
        actor_rollout_ref.actor.ppo_epochs=2 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=5e-6 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=512 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64\
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.000 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=9216\
        actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True\
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95\
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0\
        actor_rollout_ref.rollout.val_kwargs.top_k=-1\
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.000 \
        +algorithm.n_max=32 \
        +algorithm.grpo_use_std=False \
        trainer.critic_warmup=0 \
        +trainer.del_last_ckpt=False \
        +trainer.log_train=True \
        +trainer.validation_data_dir=$OUTPUT_PATH \
        trainer.project_name='DARS' \
        trainer.experiment_name='DARS-1B-ET-Generation' \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=14 \
        trainer.test_freq=14 \
        trainer.default_hdfs_dir=null \
        trainer.total_training_steps=86 \
        trainer.total_epochs=30 \
        trainer.logger=console \
        reward_model.model.trust_remote_code=True\
        critic.model.trust_remote_code=True\
        data.trust_remote_code=True\
        actor_rollout_ref.model.trust_remote_code=True\
        +data.use_fast=True \
        trainer.device=npu 
        "${@:1}"
