#!/bin/bash
#SBATCH --job-name=train_gnn_celeba_sb
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

# Execute the final GNN training loop for CelebA-Smiling (sb)
# Requires config/gnn/celeba_sb.yaml to point at the CelebA-Smiling KNN graph (built from all 129 latent dims -- no reduced-dim file was supplied for CelebA, unlike the other datasets)
# produced by src/scripts/gen_knn_graphs.py (README Step 9).
echo "Starting GNN Training (CelebA-Smiling (sb))..."
python src/scripts/train.py \
    -d config/gnn/celeba_sb.yaml \
    -m config/models/gnn.yaml \
    -t config/gnn_training_config.yaml
