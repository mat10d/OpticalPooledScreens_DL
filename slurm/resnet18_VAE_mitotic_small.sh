#!/bin/bash
#SBATCH --job-name=VAE_resnet18
#SBATCH --partition=nvidia-L40S-20
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=out/VAE_resnet18_%j.out

source /lab/barcheese01/miniconda3/etc/profile.d/conda.sh

conda activate ops_dl

VAE_RESNET_PATH="/lab/barcheese01/mdiberna/OpticalPooledScreens_DL/scripts/train/train_resnet18.py"

python $VAE_RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_mitotic_all.csv" \
    --dataset "dataset_mitotic_all" \
    --z_dim 10 \
    --epochs 100 \
    --loss_type "L1" \