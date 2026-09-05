#!/bin/bash
#SBATCH --job-name=train_gnn_fer2013
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
cd /scratch4/scr4_yxu70/rliu107/744_final/image-gnn/

# Execute the final GNN training loop for FER2013
# Requires config/gnn/fer2013.yaml to point at the FER2013 KNN graph
# produced by src/scripts/gen_knn_graphs.py (README Step 9).
echo "Starting GNN Training (FER2013)..."
python src/scripts/train.py \
    -d config/gnn/fer2013.yaml \
    -m config/models/gnn.yaml \
    -t config/gnn_training_config.yaml
