#!/bin/bash
#SBATCH --job-name=gen_knn_celeba_sb
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1     # 1 task
#SBATCH --cpus-per-task=8       # Annoy's index build parallelizes across cores
#SBATCH --time=02:00:00         # Max wall-time (rough estimate -- see note below)
#SBATCH --mem=48G               # Request RAM
#SBATCH --account=yxu70_gpu

# Load modules and activate environment
module load anaconda
conda activate image_gnn_env

# Navigate to working directory
cd /scratch4/scr4_yxu70/rliu107/744_final/image-gnn/

# Build the KNN graph for CelebA-Smiling (README Step 9).
# Reads config/knn/celeba_sb.yaml (the full-dimension, non-"_pca_" embedding
# files after the dimensionality fix) and writes
# data/CELEBAGraph/celeba_sb_{train,test}_knn_graph-100.pkl,
# which config/gnn/celeba_sb.yaml then points training at.
#
# NOTE ON --time: this is a CPU-only step (Annoy + numpy, no .cuda() calls
# in gen_knn_graphs.py), so --gres=gpu:1 was deliberately left out to avoid
# reserving a GPU for work that never touches one. The wall-time above is a
# rough estimate, not measured -- CelebA-Smiling showed roughly 40+ it/s on comparable low-dimensional embeddings during a
# manual run. Adjust once you see a real completion time.
echo "Starting KNN graph generation (CelebA-Smiling)..."
python src/scripts/gen_knn_graphs.py \
    -d config/knn/celeba_sb.yaml \
    -k 100 \
    --dataset_name celeba_sb
