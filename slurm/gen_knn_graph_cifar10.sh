#!/bin/bash
#SBATCH --job-name=gen_knn_cifar10
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1     # 1 task
#SBATCH --cpus-per-task=16       # Annoy's index build parallelizes across cores
#SBATCH --time=06:00:00         # Max wall-time (rough estimate -- see note below)
#SBATCH --mem=64G               # Request RAM
#SBATCH --account=yxu70_gpu

# Load modules and activate environment
module load anaconda
conda activate image_gnn_env

# Navigate to working directory
cd /home/rliu107/image-gnn-id-est-2/

# Build the KNN graph for CIFAR10 (README Step 9).
# Reads config/knn/cifar10.yaml (the full-dimension, non-"_pca_" embedding
# files after the dimensionality fix) and writes
# data/CIFAR10Graph/cifar10_{train,test}_knn_graph-100.pkl,
# which config/gnn/cifar10.yaml then points training at.
#
# NOTE ON --time: this is a CPU-only step (Annoy + numpy, no .cuda() calls
# in gen_knn_graphs.py), so --gres=gpu:1 was deliberately left out to avoid
# reserving a GPU for work that never touches one. The wall-time above is a
# rough estimate, not measured -- CIFAR10 showed roughly only ~5.9 it/s (4096-dim features, ~7x slower than the other datasets) during a
# manual run. Adjust once you see a real completion time.
echo "Starting KNN graph generation (CIFAR10)..."
python src/scripts/gen_knn_graphs.py \
    -d config/knn/cifar10.yaml \
    -k 100 \
    --dataset_name cifar10
