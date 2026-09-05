#!/bin/bash
#SBATCH --job-name=train_gnn_pathmnist
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1     # 1 task
#SBATCH --cpus-per-task=8       # Request 8 CPU cores for data loading
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --time=01:30:00         # Max wall-time
#SBATCH --mem=32G               # Request 32GB of RAM
#SBATCH --account=yxu70_gpu

# Load modules and activate environment
module load anaconda
conda activate image_gnn_env

# Navigate to working directory
cd /home/rliu107/image-gnn-id-est-2/

# Execute the final GNN training loop for PathMNIST
# Requires config/gnn/pathmnist.yaml to point at the PathMNIST KNN graph
# produced by src/scripts/gen_knn_graphs.py (README Step 9).
echo "Starting GNN Training (PathMNIST)..."
python src/scripts/train.py \
    -d config/gnn/pathmnist.yaml \
    -m config/models/gnn.yaml \
    -t config/gnn_training_config.yaml
