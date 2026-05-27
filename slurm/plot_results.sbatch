#!/bin/bash
#SBATCH --job-name=osteo_plots
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu:1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/osteo_plots_%j.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/osteo_plots_%j.err

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

mkdir -p /home/dgogoana/osteophytes_project/logs "$TORCH_HOME" outputs/figures

python -m src.collect_results \
  --metrics-root outputs/metrics \
  --checkpoints-root outputs/checkpoints \
  --out outputs/results_all.csv
python -m src.plot_curves --results outputs/results_all.csv --out-dir outputs/figures
python -m src.plateau \
  --results outputs/results_all.csv \
  --full-budget-name full \
  --primary-metric quality_mean \
  --higher-is-better true \
  --out outputs/plateau_analysis.json
