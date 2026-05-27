#!/bin/bash
#SBATCH --job-name=osteo_binary
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu:1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-2
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/osteo_binary_%A_%a.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/osteo_binary_%A_%a.err

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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /home/dgogoana/osteophytes_project/logs "$TORCH_HOME" outputs/checkpoints outputs/predictions

SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

python -m src.train_binary \
  --data-config configs/data.json \
  --train-config configs/binary_pretrain_template.json \
  --seed "$SEED" \
  --out-dir "outputs/checkpoints/binary_seed${SEED}" \
  --batch-size 32 \
  --num-workers 2

python -m src.predict \
  --data-config configs/data.json \
  --checkpoint "outputs/checkpoints/binary_seed${SEED}/best.pt" \
  --split train \
  --out "outputs/predictions/binary_train_scores_seed${SEED}.csv" \
  --batch-size 32 \
  --num-workers 2
