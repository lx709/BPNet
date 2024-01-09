#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J bpnet
#SBATCH --mail-user=xiang.li.1@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=210:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:6
#SBATCH --cpus-per-task=48
#SBATCH --account conf-eccv-2024.03.14-elhosemh

cd /ibex/ai/home/lix0i/3DCoMPaT/BPNet
# export TOKENIZERS_PARALLELISM=false
source ~/.bashrc
conda init bash

conda activate BPNet2
# module load cuda/11.7
export OMP_NUM_THREADS=48

sh ./tool/train.sh com10_coarse_v2 config/compat/bpnet_10_coarse.yaml 48

# sh ./tool/train.sh com10_fine_v2 config/compat/bpnet_10_fine.yaml 48