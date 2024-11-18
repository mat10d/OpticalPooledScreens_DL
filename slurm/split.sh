#!/bin/bash
#SBATCH --job-name=split
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output out/split-%j.out

source /lab/barcheese01/miniconda3/etc/profile.d/conda.sh

conda activate ops_dl

python ../scripts/data/split.py
