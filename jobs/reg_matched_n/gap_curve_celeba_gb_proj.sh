#!/bin/bash
# Matched-training-size gap curve for celeba_gb (Gap 1) using GNNProjModel, with
# ID regularization -- CE + lambda_id * id_penalty(embeddings, "mle") via
# config/gnn_training_config_reg.yaml (see training/id_regularizers.py).
# Otherwise identical to jobs/matched_n/gap_curve_celeba_gb_proj.sh. Results land
# in results/proj_reg_matched_n, a dedicated directory for this
# jobs/reg_matched_n/ schedule -- separate from both the unregularized
# proj matched-n results and the earlier
# jobs/matched_n/gap_curve_celeba_gb_proj_reg.sh run (different fraction/restart
# schedule; keeping them apart avoids mixing differently-scheduled rows in
# one CSV).
#
# Every dataset in this sweep is trained at fractions of a SHARED reference
# size (reference_n=28709, fer2013's own full training size -- the smallest
# of the 7 in-scope datasets), instead of each dataset's own eligible-
# training-node count. This makes the fraction axis comparable across
# datasets: fraction=0.25 always means "7,177 training points," not "25%
# of whatever this dataset happens to have."
#
# Restart schedule: 6 restarts for the three smallest sizes, 3 for the
# three largest.
#   fractions:  0.025  0.05  0.1  0.25  0.5  1.0
#   restarts:     6      6    6    3    3    3
#
# No SBATCH header -- this project moved off Slurm. Wrap this script
# with whatever the new cluster's job submission command is, or run it
# directly (bash gap_curve_celeba_gb_proj.sh) / in the background (nohup ... &).
#
# Adjust the conda env name / project path below if they differ on the
# new cluster.
set -e
conda activate image_gnn
cd "$(dirname "$0")/../.."

mkdir -p log
exec > >(tee -a "log/gap_celeba_gb_proj_reg.log") 2>&1

echo "Starting matched-n gap-curve training (ID-regularized GNNProjModel, celeba_gb)..."
python src/scripts/train_gap_curve.py \
    -d config/gnn/celeba_gb.yaml \
    -m config/models/gnn_proj.yaml \
    -t config/gnn_training_config_reg.yaml \
    --size-fractions 0.025,0.05,0.1,0.25,0.5,1.0 \
    --num-seeds 6,6,6,3,3,3 \
    --reference-n 28709 \
    --results-dir results/gap_curve/proj_reg_matched_n
