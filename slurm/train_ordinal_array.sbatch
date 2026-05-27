#!/bin/bash
#SBATCH --job-name=osteo_ordinal
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu:1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-0
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/osteo_ordinal_%A_%a.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/osteo_ordinal_%A_%a.err

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

mkdir -p /home/dgogoana/osteophytes_project/logs "$TORCH_HOME" outputs/checkpoints/ordinal

JOB_LIST="${JOB_LIST:-outputs/job_lists/ordinal_jobs.csv}"
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$JOB_LIST")"
IFS=, read -r seed strategy budget_name mode budget_file binary_checkpoint out_dir <<< "$LINE"

python -m src.train_ordinal \
  --data-config configs/data.json \
  --train-config configs/ordinal_template.json \
  --seed "$seed" \
  --budget-file "$budget_file" \
  --budget-name "$budget_name" \
  --strategy "$strategy" \
  --mode "$mode" \
  --binary-checkpoint "$binary_checkpoint" \
  --out-dir "$out_dir" \
  --batch-size-all 32 \
  --batch-size-graded 16 \
  --num-workers 2
