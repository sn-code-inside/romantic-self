#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=FastTextLitReview
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgf@kent.ac.uk
#SBATCH --output=/home/mgf/slurm/logs/%j.out
python train_fasttext.py
