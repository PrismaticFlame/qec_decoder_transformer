#!/bin/bash
#SBATCH --account=quantum_training
#SBATCH --partition=h200_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G