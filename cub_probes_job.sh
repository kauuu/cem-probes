#!/bin/bash
#SBATCH --job-name=cub_probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# Trains runs of experiments/configs/probes/cub_cem.yaml (CEM, IntCEM) and then
# runs the inter-concept leakage probes on them (experiments/run_probes.py).
#
# Usage (from the repository root):
#   sbatch cub_probes_job.sh [/path/to/CUB200] [all|train|probe] [RUN_NAME ...]
#   sbatch cub_probes_job.sh                            → train + probe all runs
#   sbatch cub_probes_job.sh train                      → train all runs
#   sbatch cub_probes_job.sh train IntCEM               → train IntCEM only
#   sbatch cub_probes_job.sh probe IntCEM               → probe IntCEM only (needs a trained model)
#   sbatch cub_probes_job.sh probe CEM IntCEM           → probe both
#   sbatch cub_probes_job.sh /path/to/CUB200 [stage] [RUN_NAME ...]  → another CUB root
# RUN_NAMEs are run_name values of the config; without any, all its runs are
# used. Already trained runs are loaded instead of retrained, and runs without
# a trained model are skipped by the probe stage.
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
shift $(( $# > 0 ? 1 : 0 ))
CONFIG=experiments/configs/probes/cub_cem.yaml
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

source /cluster/apps/biomed/vogtlab/users/kkarthikeyan/software/miniconda3/etc/profile.d/conda.sh
conda activate cem-probes

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Run names and results directory of the config
CONFIG_RUNS=($(python -c "import yaml; print(' '.join(r['run_name'] for r in yaml.safe_load(open('${CONFIG}'))['runs']))"))
RESULTS_DIR=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}')).get('results_dir', 'results'))")
RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  RUNS=("${CONFIG_RUNS[@]}")
fi
for RUN in "${RUNS[@]}"; do
  if [[ ! " ${CONFIG_RUNS[*]} " == *" ${RUN} "* ]]; then
    echo "ERROR: unknown run '${RUN}' (runs in ${CONFIG}: ${CONFIG_RUNS[*]})"
    exit 1
  fi
done

echo "Node: $(hostname) | Job: ${SLURM_JOB_ID:-local} | Stage: ${STAGE} | Runs: ${RUNS[*]}"
echo "CUB root: ${CUB_DIR}"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

if [[ "${STAGE}" == "all" || "${STAGE}" == "train" ]]; then
  # Only train the requested runs (run_experiments matches {run_name}_split_{n})
  RUN_REGEX="^($(IFS='|'; echo "${RUNS[*]}"))_split_"
  echo "=== Training ${RUNS[*]} ($(date)) ==="
  python experiments/run_experiments.py \
    -c "${CONFIG}" \
    --num_workers "${NUM_WORKERS}" \
    --filter_in "${RUN_REGEX}" \
    -p dataset_config.root_dir "${CUB_DIR}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "probe" ]]; then
  for RUN in "${RUNS[@]}"; do
    if [[ ! -f "${RESULTS_DIR%/}/${RUN}_fold_1.pt" ]]; then
      echo "WARNING: skipping probes on ${RUN}: no trained model at ${RESULTS_DIR%/}/${RUN}_fold_1.pt"
      continue
    fi
    echo "=== Running leakage probes on ${RUN} ($(date)) ==="
    python experiments/run_probes.py \
      -c "${CONFIG}" \
      --run_name "${RUN}" \
      -p dataset_config.root_dir "${CUB_DIR}" \
      -p num_workers "${NUM_WORKERS}"
  done
fi

echo "=== Done ($(date)) ==="
