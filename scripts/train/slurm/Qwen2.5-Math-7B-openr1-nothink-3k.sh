#!/bin/bash
#SBATCH -o slurm/Qwen2.5-Math-7B-openr1-nothink-3k-%j.out
#SBATCH --partition=i64m1tga800u
#SBATCH -J 7bOR1
#SBATCH -n 8
#SBATCH --gres=gpu:4

module load cuda/12.4

source /hpc2hdd/home/zyang398/yangzhch6/anaconda3/bin/activate rllm-baseline

bash /hpc2hdd/home/zyang398/yangzhch6/projs/reasoning_baselines/rllm/scripts/train/Qwen2.5-Math-7B-openr1-nothink-3k.sh