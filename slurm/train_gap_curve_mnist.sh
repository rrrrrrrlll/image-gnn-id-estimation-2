#!/bin/bash
#SBATCH --job-name=gap_curve_mnist
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

# Generalization-gap-vs-graph-size curve for MNIST (Gap 1).
# Trains a fresh model from scratch at each of several graph sizes
# (--size-fractions, default 10/25/50/75/100% of eligible training
# nodes), repeated --num-seeds times per size (default 3), and appends
# one row per (size, seed) to results/gap_curve/mnist.csv.
# Requires config/gnn/mnist.yaml to point at the MNIST KNN graph
# produced by src/scripts/gen_knn_graphs.py (README Step 9) -- same
# prerequisite as train_gnn_mnist.sh.
echo "Starting Gap-1 curve training (MNIST)..."
python src/scripts/train_gap_curve.py \
    -d config/gnn/mnist.yaml \
    -m config/models/gnn.yaml \
    -t config/gnn_training_config.yaml
