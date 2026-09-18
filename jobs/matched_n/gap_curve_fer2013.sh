#!/bin/bash
# Matched-training-size gap curve for fer2013 (Gap 1).
#
# Every dataset in this matched-n sweep is trained at fractions of a
# SHARED reference size (reference_n=28709, fer2013's own full training
# size -- the smallest of the 7 in-scope datasets), instead of each
# dataset's own eligible-training-node count. This makes the fraction
# axis comparable across datasets: fraction=0.25 always means "7,177
# training points," not "25% of whatever this dataset happens to have."
#
# Restart schedule: 10 restarts for the four smallest sizes (noisier,
# more restarts needed), 5 restarts for the four largest.
#   fractions:  0.01  0.025  0.05  0.1  0.25  0.5  0.75  1.0
#   restarts:    10    10     10   10    5     5    5     5
#
# No SBATCH header -- this project moved off Slurm. Wrap this script
# with whatever the new cluster's job submission command is, or run it
# directly (bash gap_curve_fer2013.sh) / in the background (nohup ... &).
#
# Adjust the conda env name / project path below if they differ on the
# new cluster.
set -e
conda activate image_gnn
cd "$(dirname "$0")/../.."

mkdir -p log
exec > >(tee -a "log/gap_fer2013.log") 2>&1

echo "Starting matched-n gap-curve training (fer2013)..."
python src/scripts/train_gap_curve.py \
    -d config/gnn/fer2013.yaml \
    -m config/models/gnn.yaml \
    -t config/gnn_training_config.yaml \
    --size-fractions 0.01,0.025,0.05,0.1,0.25,0.5,0.75,1.0 \
    --num-seeds 10,10,10,10,5,5,5,5 \
    --reference-n 28709 \
    --results-dir results/gap_curve/gap_curve_matched_n
