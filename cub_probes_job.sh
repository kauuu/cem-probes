#!/bin/bash
#SBATCH --job-name=cub_probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# stdout/stderr go to logs/slurm/<job name>-<job id>.{out,err}. SLURM does not
# create that folder, so it must exist before submitting (it is kept in the
# repo via logs/slurm/.gitkeep; otherwise run `mkdir -p logs/slurm`).
#
# Trains the CUB CEM of experiments/configs/probes/cub_cem.yaml and then runs
# the inter-concept leakage probes on it (experiments/run_probes.py).
#
# Usage (from the repository root):
#   sbatch cub_probes_job.sh                            → train + probe
#   sbatch cub_probes_job.sh train                      → train only
#   sbatch cub_probes_job.sh probe                      → probe only (needs a trained model)
#   sbatch cub_probes_job.sh /path/to/CUB200 [stage]    → same, with another CUB root
# Any #SBATCH value can be overridden on the command line, e.g.
#   sbatch --time 08:00:00 cub_probes_job.sh /path/to/CUB200 probe
#
# The CUB root (the dataset_config.root_dir of the config) is the directory
# containing class_attr_data_10/ and the CUB images. It defaults to
# DEFAULT_CUB_DIR below.

set -euo pipefail

DEFAULT_CUB_DIR=/cluster/home/kkarthikeyan/thesis/cub200/
# An optional first argument that is not a stage overrides the CUB root
if [[ $# -gt 0 && "$1" != "all" && "$1" != "train" && "$1" != "probe" ]]; then
  CUB_DIR=$1
  shift
else
  CUB_DIR=${DEFAULT_CUB_DIR}
fi
if [[ ! -d "${CUB_DIR}" ]]; then
  echo "ERROR: CUB root '${CUB_DIR}' is not a directory (a mistyped stage is read as a CUB root)"
  exit 1
fi
# The CUB loader builds image paths by string replacement, so it needs the
# trailing slash
CUB_DIR="${CUB_DIR%/}/"
STAGE=${1:-all}
if [[ "${STAGE}" != "all" && "${STAGE}" != "train" && "${STAGE}" != "probe" ]]; then
  echo "ERROR: unknown stage '${STAGE}' (expected all, train or probe)"
  exit 1
fi
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
