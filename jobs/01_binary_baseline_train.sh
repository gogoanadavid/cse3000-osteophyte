#!/bin/bash
#SBATCH --job-name=osteo_bin_train
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/osteo_bin_train_%j.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/osteo_bin_train_%j.err

set -euo pipefail

cd /home/dgogoana/osteophytes_project

module purge
module load 2025
module load python/3.11.9
module load py-numpy/1.26.4
module load py-pandas/2.2.3
module load py-h5py/3.12.1
module load py-pillow/11.0.0
module load py-torch/2.5.1

export PYTHONPATH="/scratch/dgogoana/osteophytes_project/venvs/torch_h5env/lib/python3.11/site-packages:${PYTHONPATH:-}"
export TORCH_HOME=/scratch/dgogoana/osteophytes_project/torch_cache

mkdir -p /home/dgogoana/osteophytes_project/logs
mkdir -p "$TORCH_HOME"
mkdir -p /scratch/dgogoana/osteophytes_project/runs/01_binary_baseline

python scripts/05_train_binary_baseline.py \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 2 \
  --selection-metric mean_auroc \
  --h5-path /scratch/dgogoana/osteophytes_project/data/all-for-hip-prediction-20260420-0.4mm-224x224.h5
