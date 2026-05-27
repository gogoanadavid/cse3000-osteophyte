#!/bin/bash
#SBATCH --job-name=osteo_budgets
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu:1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-2
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/osteo_budgets_%A_%a.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/osteo_budgets_%A_%a.err

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

mkdir -p /home/dgogoana/osteophytes_project/logs "$TORCH_HOME" budgets outputs/job_lists

SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
BUDGETS="0,64,128,256,512,1024,2048,4096,8192,full"

python -m src.sampling --index outputs/index.csv --strategy random --seed "$SEED" --budgets "$BUDGETS" --out-dir "budgets/random_seed${SEED}"
python -m src.sampling --index outputs/index.csv --strategy binary_positive_enriched --seed "$SEED" --budgets "$BUDGETS" --out-dir "budgets/binary_positive_enriched_seed${SEED}"
python -m src.sampling --index outputs/index.csv --strategy oracle_grade_stratified --seed "$SEED" --budgets "$BUDGETS" --out-dir "budgets/oracle_grade_stratified_seed${SEED}"
python -m src.sampling \
  --index outputs/index.csv \
  --strategy score_stratified \
  --scores "outputs/predictions/binary_train_scores_seed${SEED}.csv" \
  --score-column severity_proxy \
  --seed "$SEED" \
  --budgets "$BUDGETS" \
  --out-dir "budgets/score_stratified_seed${SEED}"

python -m src.make_job_list \
  --experiment-grid configs/experiment_grid.json \
  --out outputs/job_lists/ordinal_jobs.csv \
  --eval-out outputs/job_lists/eval_jobs.csv
