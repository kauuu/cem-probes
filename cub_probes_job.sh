#!/bin/bash
#SBATCH --job-name=cub_probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# Trains the CUB CEM of experiments/configs/probes/cub_cem.yaml and then runs
# the inter-concept leakage probes on it (experiments/run_probes.py).
#
# Usage (from the repository root):
#   sbatch cub_probes_job.sh /path/to/CUB200            → train + probe
#   sbatch cub_probes_job.sh /path/to/CUB200 train      → train only
#   sbatch cub_probes_job.sh /path/to/CUB200 probe      → probe only (needs a trained model)
# stdout/stderr go to logs/slurm/<job name>-<job id>.{out,err}. SLURM does not
# create that folder, so it must exist before submitting (it is kept in the
# repo; otherwise run `mkdir -p logs/slurm`).
# Any #SBATCH value can be overridden on the command line, e.g.
#   sbatch --time 08:00:00 cub_probes_job.sh /path/to/CUB200 probe
#
# /path/to/CUB200 (the dataset_config.root_dir of the config) must contain
#   class_attr_data_10/{train,val,test}.pkl   (copy of cem/data/CUB200/class_attr_data_10/)
#   CUB_200_2011/images/<class>/<image>.jpg   (the extracted CUB-200-2011 download)

set -euo pipefail

CUB_DIR=${1:?"Usage: sbatch cub_probes_job.sh /path/to/CUB200 [all|train|probe]"}
# The CUB loader builds image paths by string replacement, so it needs the
# trailing slash
CUB_DIR="${CUB_DIR%/}/"
STAGE=${2:-all}
CONFIG=experiments/configs/probes/cub_cem.yaml
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

source /cluster/apps/biomed/vogtlab/users/kkarthikeyan/software/miniconda3/etc/profile.d/conda.sh
conda activate cem-probes

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Node: $(hostname) | Job: ${SLURM_JOB_ID:-local} | Stage: ${STAGE}"
echo "CUB root: ${CUB_DIR}"
for required in "${CUB_DIR}class_attr_data_10/train.pkl" "${CUB_DIR}CUB_200_2011/images"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: ${required} not found (see the CUB root layout in this script's header)"
    exit 1
  fi
done
# Compute nodes have no internet, so the ImageNet weights of the ResNet-18
# backbone must already be in the torch hub cache (download them once on the
# login node: python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')")
if ! python -c "import glob, os, sys, torch; sys.exit(not glob.glob(os.path.join(torch.hub.get_dir(), 'checkpoints', 'resnet18-*.pth')))"; then
  echo "ERROR: pretrained ResNet-18 weights not found in the torch hub cache; download them on the login node first (see this script)"
  exit 1
fi
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

if [[ "${STAGE}" == "all" || "${STAGE}" == "train" ]]; then
  echo "=== Training CEM ($(date)) ==="
  python experiments/run_experiments.py \
    -c "${CONFIG}" \
    --num_workers "${NUM_WORKERS}" \
    -p dataset_config.root_dir "${CUB_DIR}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "probe" ]]; then
  echo "=== Running leakage probes ($(date)) ==="
  python experiments/run_probes.py \
    -c "${CONFIG}" \
    -p dataset_config.root_dir "${CUB_DIR}" \
    -p num_workers "${NUM_WORKERS}"
fi

echo "=== Done ($(date)) ==="
