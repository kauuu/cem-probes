#!/bin/bash
# Usage:
#   ./jupyter_job.sh                                   → 1h, CPU
#   ./jupyter_job.sh 02:00:00                          → 2h, CPU
#   ./jupyter_job.sh 02:00:00 -p gpu --gres gpu:1      → 2h, 1 GPU
#   ./jupyter_job.sh -p gpu --gres gpu:1 --mem 32G     → 1h, 1 GPU, 32G RAM

TIME=01:00:00
if [ -n "$1" ] && [[ "$1" != -* ]]; then
  TIME=$1
  shift
fi

srun --job-name jupyter --time "$TIME" "$@" bash -c \
  'source /cluster/apps/biomed/vogtlab/users/kkarthikeyan/software/miniconda3/etc/profile.d/conda.sh && \
   conda activate cem-probes && \
   echo "Node: $(hostname)" && \
   jupyter notebook --ip $(hostname -i) --no-browser'
