#!/bin/bash
#SBATCH --job-name=mlp_baseline_fer2013
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1     # 1 task
#SBATCH --cpus-per-task=8       # Request 8 CPU cores for data loading
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --time=08:00:00         # Max wall-time
#SBATCH --mem=32G               # Request 32GB of RAM
#SBATCH --account=yxu70_gpu

# Load modules and activate environment
module load anaconda
conda activate image_gnn_env

# Navigate to working directory
cd /scratch4/scr4_yxu70/rliu107/744_final/image-gnn/

# Non-graph MLP baseline for FER2013 (Gap 2). Trains an MLP directly on
# the raw-embeddings file (features + label, SMSL flag column stripped)
# at data/FER2013Embeddings/fer2013_train_embeddings.npy --
# no KNN graph involved -- and appends one row to
# results/mlp_baseline/fer2013.csv. Only needs config/mlp/fer2013.yaml, so
# this does not depend on the KNN graph step and can run as soon as
# that embeddings file exists.
echo "Starting MLP baseline training (FER2013)..."
python src/scripts/train_mlp_baseline.py \
    -d config/mlp/fer2013.yaml \
    -m config/models/mlp.yaml \
    -t config/gnn_training_config.yaml
