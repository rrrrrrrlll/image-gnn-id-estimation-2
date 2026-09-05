#!/bin/bash
#SBATCH --job-name=mlp_baseline_octmnist
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

# Non-graph MLP baseline for OCTMNIST (Gap 2). Trains an MLP directly on
# the Step 5 VAE embeddings -- no KNN graph involved -- and appends one
# row to results/mlp_baseline/octmnist.csv. Only needs config/mlp/octmnist.yaml,
# which only needs the Step 5 embeddings, so this does not depend on
# the KNN graph step and can run as soon as embeddings are generated.
echo "Starting MLP baseline training (OCTMNIST)..."
python src/scripts/train_mlp_baseline.py \
    -d config/mlp/octmnist.yaml \
    -m config/models/mlp.yaml \
    -t config/gnn_training_config.yaml
