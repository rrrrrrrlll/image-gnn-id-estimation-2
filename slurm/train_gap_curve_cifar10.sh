#!/bin/bash
#SBATCH --job-name=gap_curve_cifar10
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
cd /home/rliu107/image-gnn-id-est-2/

# Generalization-gap-vs-graph-size curve for CIFAR10 (Gap 1).
# Trains a fresh model from scratch at each of several graph sizes
# (--size-fractions, default 10/25/50/75/100% of eligible training
# nodes), repeated --num-seeds times per size (default 3), and appends
# one row per (size, seed) to results/gap_curve/cifar10.csv.
# Requires config/gnn/cifar10.yaml to point at the CIFAR10 KNN graph
# produced by src/scripts/gen_knn_graphs.py (README Step 9) -- same
# prerequisite as train_gnn_cifar10.sh.
echo "Starting Gap-1 curve training (CIFAR10)..."
python src/scripts/train_gap_curve.py \
    -d config/gnn/cifar10.yaml \
    -m config/models/gnn.yaml \
    -t config/gnn_training_config.yaml
