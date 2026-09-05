#!/bin/bash
#SBATCH --job-name=gnn_sweep
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1     # 1 task
#SBATCH --cpus-per-task=8       # Request 8 CPU cores for data loading
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --time=72:00:00         # Max wall-time (24 hours)
#SBATCH --mem=32G               # Request 32GB of RAM
#SBATCH --account=yxu70_gpu

# Load modules and activate environment
module load anaconda
conda activate image_gnn_env

cd /scratch4/scr4_yxu70/rliu107/744_final/image-gnn/

# Start the W&B agent
# Before submitting, create this dataset's sweep once with:
#   wandb sweep --project image-gnn config/wandb_cnnvae_sweep_config.yaml
# then replace REPLACE_WITH_PATHMNIST_SWEEP_ID below with the printed sweep ID.
# (sweep_id is now a CLI flag, not hardcoded in train_model.py, so per-dataset
#  sweeps can be submitted and run concurrently.)
python src/scripts/train_model.py \
    -d config/datasets/pathmnist.yaml \
    -m config/models/cnnvae_rgb.yaml \
    -t config/vae_training_config.yaml \
    --sweep_id ynb50nld
