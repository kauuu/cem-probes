#!/bin/bash
#SBATCH --job-name=cub_probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Trains the CUB CEM of experiments/configs/probes/cub_cem.yaml and then runs
# the inter-concept leakage probes on it (experiments/run_probes.py).
#
# Usage (from the repository root):
#   sbatch cub_probes_job.sh /path/to/CUB200            → train + probe
#   sbatch cub_probes_job.sh /path/to/CUB200 train      → train only
#   sbatch cub_probes_job.sh /path/to/CUB200 probe      → probe only (needs a trained model)
# Any #SBATCH value can be overridden on the command line, e.g.
#   sbatch --time 08:00:00 cub_probes_job.sh /path/to/CUB200 probe
#
# /path/to/CUB200 is the directory containing class_attr_data_10/ and the
# CUB images (the dataset_config.root_dir of the config).

set -euo pipefail

CUB_DIR=${1:?"Usage: sbatch cub_probes_job.sh /path/to/CUB200 [all|train|probe]"}
STAGE=${2:-all}
CONFIG=experiments/configs/probes/cub_cem.yaml
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

source /cluster/apps/biomed/vogtlab/users/kkarthikeyan/software/miniconda3/etc/profile.d/conda.sh
conda activate cem-probes

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Node: $(hostname) | Job: ${SLURM_JOB_ID:-local} | Stage: ${STAGE}"
echo "CUB root: ${CUB_DIR}"
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
