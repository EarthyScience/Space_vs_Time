#!/bin/bash

#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700GB
#SBATCH --time=20:00:00

# Output logs
#SBATCH -o logs/%x.out-%j
#SBATCH -e logs/%x.error-%j

# Activate conda environment
source /path/to/miniconda3/bin/activate my_environment

# Run dataset creation script
python src/create_data_splits.py

exit