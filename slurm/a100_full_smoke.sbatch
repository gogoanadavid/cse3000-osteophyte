#!/bin/bash
#SBATCH --job-name=osteo_a100_smoke
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home/dgogoana/osteophytes_project/logs/osteo_a100_smoke_%j.out
#SBATCH --error=/home/dgogoana/osteophytes_project/logs/osteo_a100_smoke_%j.err

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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /home/dgogoana/osteophytes_project/logs

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || true

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

device = torch.device("cuda")
print("device_name", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device=device)
y = x @ x.T
torch.cuda.synchronize()
print("matmul_mean", float(y.mean().detach().cpu()))
print("ok")
PY
