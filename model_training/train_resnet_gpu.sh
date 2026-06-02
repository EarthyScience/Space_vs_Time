#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=10:00:00

# Output logs
#SBATCH -o logs/%x.out-%j
#SBATCH -e logs/%x.error-%j

# Load required modules
module load cuda

# Activate conda environment
source /path/to/miniconda3/bin/activate my_environment

# Run training script
python train_models_gpu_basic.py