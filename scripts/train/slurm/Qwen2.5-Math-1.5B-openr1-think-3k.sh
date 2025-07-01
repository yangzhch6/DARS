#!/bin/bash
#SBATCH -o slurm/Qwen2.5-Math-1.5B-openr1-think-3k-%j.out
#SBATCH --partition=i64m1tga800u
#SBATCH -J 1.5bOR1
#SBATCH -n 4
#SBATCH --gres=gpu:2

module load cuda/12.4

source /hpc2hdd/home/zyang398/yangzhch6/anaconda3/bin/activate rllm-baseline

bash /hpc2hdd/home/zyang398/yangzhch6/projs/reasoning_baselines/rllm/scripts/train/Qwen2.5-Math-1.5B-openr1-think-3k.sh