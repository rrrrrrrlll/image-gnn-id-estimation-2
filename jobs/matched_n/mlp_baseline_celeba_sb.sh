#!/bin/bash
# Matched-training-size MLP baseline for celeba_sb (Gap 2).
#
# Retrains the non-graph MLP baseline at the SAME matched training size
# (n_target=28709, 5 restarts) used by the fraction=1.0 point of the
# matched-n gap-curve sweep, so GNN-vs-MLP accuracy comparisons are not
# confounded by different training-set sizes.
#
# No SBATCH header -- this project moved off Slurm. Wrap this script
# with whatever the new cluster's job submission command is, or run it
# directly (bash mlp_baseline_celeba_sb.sh) / in the background (nohup ... &).
#
# Adjust the conda env name / project path below if they differ on the
# new cluster.
set -e
conda activate image_gnn
cd "$(dirname "$0")/../.."

mkdir -p log
exec > >(tee -a "log/mlp_celeba_sb.log") 2>&1

echo "Starting matched-n MLP baseline training (celeba_sb)..."
python src/scripts/train_mlp_baseline.py \
    -d config/mlp/celeba_sb.yaml \
    -m config/models/mlp.yaml \
    -t config/gnn_training_config.yaml \
    --n-target 28709 \
    --num-seeds 5 \
    --results-dir results/mlp_baseline/mlp_baseline_matched_n
