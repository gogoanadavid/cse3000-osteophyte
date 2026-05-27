#!/bin/bash
set -euo pipefail

PROJECT_DIR=/home/dgogoana/osteophytes_project
LOG_DIR=/home/dgogoana/osteophytes_project/logs
RUNS_DIR=/scratch/dgogoana/osteophytes_project/runs/02_mixed_supervision
BINARY_CHECKPOINT=/scratch/dgogoana/osteophytes_project/runs/01_binary_baseline/20260527_145251/best_model.pt
H5_PATH=/scratch/dgogoana/osteophytes_project/data/all-for-hip-prediction-20260420-0.4mm-224x224.h5

mkdir -p "$LOG_DIR" "$RUNS_DIR"

submit_budget_job() {
  local fraction="$1"
  local percent="$2"
  local tag="location_binary_mixed_${percent}_grade2_targeted_threshold_independent"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=osteo_mix${percent}_g2
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=${LOG_DIR}/osteo_mix${percent}_g2_%j.out
#SBATCH --error=${LOG_DIR}/osteo_mix${percent}_g2_%j.err

set -euo pipefail
cd ${PROJECT_DIR}
module purge
module load 2025
module load python/3.11.9
module load py-numpy/1.26.4
module load py-pandas/2.2.3
module load py-h5py/3.12.1
module load py-pillow/11.0.0
module load py-torch/2.5.1
export PYTHONPATH="/scratch/dgogoana/osteophytes_project/venvs/torch_h5env/lib/python3.11/site-packages:\${PYTHONPATH:-}"
export TORCH_HOME=/scratch/dgogoana/osteophytes_project/torch_cache
mkdir -p ${LOG_DIR} "\$TORCH_HOME" ${RUNS_DIR}
python scripts/07_train_mixed_supervision.py \
  --supervision-mode mixed \
  --strong-fraction ${fraction} \
  --strong-sampling-strategy grade2_targeted \
  --weak-label-mode location_binary \
  --model-head threshold_independent \
  --selection-metric mean_spearman \
  --output-root ${RUNS_DIR}/ \
  --weak-loss-weight 0.1 \
  --threshold-weights 1.0,3.0,2.0 \
  --binary-checkpoint ${BINARY_CHECKPOINT} \
  --run-tag ${tag} \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 2 \
  --h5-path ${H5_PATH}
EOF
}

submit_budget_job 0.05 005
submit_budget_job 0.10 010
submit_budget_job 0.25 025
submit_budget_job 0.50 050
submit_budget_job 0.75 075
submit_budget_job 1.0 100
