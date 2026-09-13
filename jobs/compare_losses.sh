#!/usr/bin/env bash
# Run inside an already activated cluster Python environment.
# Scheduler independent: bash jobs/compare_losses.sh [Python arguments...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u src/scripts/compare_losses.py "$@"
