#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RomSelfAssoc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgf@kent.ac.uk
#SBATCH --output=/home/mgf/slurm/logs/%j.out
python association_analysis.py -w 15
python association_analysis.py -w 25
python association_analysis.py -w 35
