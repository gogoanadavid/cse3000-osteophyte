#!/bin/bash
#SBATCH --job-name=mixed_sanity
#SBATCH --partition=gpu-a100-small
#SBATCH --gpus-per-task=1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/mixed_sanity_%j.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/mixed_sanity_%j.err

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

mkdir -p /home/dgogoana/osteophytes_project/logs "$TORCH_HOME" outputs/checkpoints/ordinal outputs/metrics

# Override these at submission time if needed, for example:
#   sbatch --export=ALL,SEED=1,BUDGET=2048 slurm/one_mixed_sanity.sbatch
export SEED="${SEED:-0}"
export STRATEGY="${STRATEGY:-score_stratified}"
export BUDGET="${BUDGET:-1024}"
export BATCH_ALL="${BATCH_ALL:-32}"
export BATCH_GRADED="${BATCH_GRADED:-16}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export DATA_CONFIG="${DATA_CONFIG:-configs/data.json}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/ordinal_template.json}"

bash scripts/run_one_mixed_sanity.sh
