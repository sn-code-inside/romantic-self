#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=CollocationsWithSelf
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgf@kent.ac.uk
#SBATCH --output=/home/mgf/slurm/logs/%j.out
python ~/romantic-self/collocation_analysis.py