#!/bin/bash
#SBATCH --job-name=osteo_mix025_g2_noinit
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/osteo_mix025_g2_noinit_%j.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/osteo_mix025_g2_noinit_%j.err

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
mkdir -p /home/dgogoana/osteophytes_project/logs "$TORCH_HOME" /scratch/dgogoana/osteophytes_project/runs/02_mixed_supervision
python scripts/07_train_mixed_supervision.py \
  --supervision-mode mixed \
  --strong-fraction 0.25 \
  --strong-sampling-strategy grade2_targeted \
  --weak-label-mode location_binary \
  --model-head threshold_independent \
  --selection-metric mean_spearman \
  --output-root /scratch/dgogoana/osteophytes_project/runs/02_mixed_supervision/ \
  --weak-loss-weight 0.1 \
  --threshold-weights 1.0,3.0,2.0 \
  --run-tag location_binary_mixed_025_grade2_targeted_no_init_threshold_independent \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 2 \
  --h5-path /scratch/dgogoana/osteophytes_project/data/all-for-hip-prediction-20260420-0.4mm-224x224.h5
