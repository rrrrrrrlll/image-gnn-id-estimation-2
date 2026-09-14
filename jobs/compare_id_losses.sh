#!/usr/bin/env bash
# Four arms: CE, CE+MLE, CE+TwoNN, CE+smooth correlation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u src/scripts/compare_id_losses.py "$@"
