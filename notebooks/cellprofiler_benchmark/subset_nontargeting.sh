#!/bin/bash
#SBATCH --job-name=subset
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output subset-%j.out

source /lab/barcheese01/miniconda3/etc/profile.d/conda.sh

conda activate ops_dl

python subset_nontargeting.py