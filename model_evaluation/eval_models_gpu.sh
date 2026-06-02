#!/bin/bash

# -----------------------------------------------------------------------------
# Evaluate trained machine learning models on GPU resources using SLURM.
#
# This script:
#   1. Requests GPU and memory resources.
#   2. Activates the required conda environment.
#   3. Runs the model evaluation workflow.
#
# Adapt the conda installation path and environment name to your system.
# -----------------------------------------------------------------------------

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=10:00:00

# Output logs
#SBATCH --output=logs/%x.out-%j
#SBATCH --error=logs/%x.error-%j

# Load required modules
module load cuda

# Activate conda environment
source /path/to/miniconda3/bin/activate my_environment

# Run evaluation script
python src/model_evaluation/model_anom_space_time.py

exit