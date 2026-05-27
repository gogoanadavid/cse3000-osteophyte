#!/bin/bash
#SBATCH --job-name=osteo_mix025_init
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/osteo_mix025_init_%j.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/osteo_mix025_init_%j.err

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

if [[ -z "${BINARY_CHECKPOINT:-}" ]]; then
  BINARY_CHECKPOINT="$(
    find /scratch/dgogoana/osteophytes_project/runs/01_binary_baseline \
      -mindepth 2 \
      -maxdepth 2 \
      -name best_model.pt \
      -type f \
      -print 2>/dev/null \
      | sort \
      | tail -n 1
  )"
fi

if [[ -z "${BINARY_CHECKPOINT:-}" || ! -f "$BINARY_CHECKPOINT" ]]; then
  echo "Could not find a binary baseline best_model.pt."
  echo "Run jobs/01_binary_baseline_train.sh first or submit with:"
  echo "  BINARY_CHECKPOINT=/scratch/dgogoana/osteophytes_project/runs/01_binary_baseline/<timestamp>/best_model.pt sbatch jobs/02_mixed_supervision_025_binary_init.sh"
  exit 1
fi

echo "Using binary checkpoint: $BINARY_CHECKPOINT"

python scripts/07_train_mixed_supervision.py --supervision-mode mixed --strong-fraction 0.25 --weak-label-mode location_binary --strong-sampling-strategy random --model-head threshold_independent --init-from-binary-checkpoint "$BINARY_CHECKPOINT" --loss-balance-mode proportional --selection-metric mean_spearman --epochs 10 --batch-size 32 --num-workers 2 --h5-path /scratch/dgogoana/osteophytes_project/data/all-for-hip-prediction-20260420-0.4mm-224x224.h5
