#!/bin/bash

# -----------------------------------------------------------------------------
# Evaluate trained machine learning models on a CPU-based SLURM cluster.
#
# This script:
#   1. Requests compute resources from the scheduler.
#   2. Activates the required conda environment.
#   3. Runs the model evaluation workflow.
#
# Adapt the conda installation path, environment name, and script paths
# to your local system.
# -----------------------------------------------------------------------------

#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=10:00:00

# Output logs
#SBATCH --output=logs/%x.out-%j
#SBATCH --error=logs/%x.error-%j

# Activate conda environment
source /path/to/miniconda3/bin/activate my_environment

# Optional evaluation script
# python src/model_evaluation/model_splits_eval.py

# Run evaluation
python src/model_evaluation/eval_model_site_year.py

exit