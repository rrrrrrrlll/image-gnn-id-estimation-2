#!/usr/bin/env bash
# Submit from repository root, after activating the cluster environment:
# sbatch jobs/compare_losses.slurm
# Supply account/partition via sbatch options as required by your cluster.
#SBATCH --job-name=gnn-loss-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=loss-comparison-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
bash jobs/compare_losses.sh --device cuda "$@"
