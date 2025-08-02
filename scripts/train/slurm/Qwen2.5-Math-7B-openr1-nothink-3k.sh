#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=/mnt/weka/home/yongxin.wang/workspace/yangzhch6/rvlr-baseline/slurm/train/Qwen2.5-Math-7B-openr1-nothink-3k.out

source /mnt/weka/home/yongxin.wang/miniconda3/bin/activate rlvr-baseline

bash /mnt/weka/home/yongxin.wang/workspace/yangzhch6/rvlr-baseline/scripts/train/Qwen2.5-Math-7B-openr1-nothink-3k.sh
