#!/bin/bash
#SBATCH --job-name=VAE_resnet18
#SBATCH --partition=nvidia-t4-20
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=out/VAE_resnet18_%j.out

source /lab/barcheese01/miniconda3/etc/profile.d/conda.sh

conda activate ops_dl

VAE_RESNET_PATH="/lab/barcheese01/mdiberna/OpticalPooledScreens_DL/scripts/train/train_resnet18.py"

# Run the Python script with custom arguments
# python $VAE_RESNET_PATH \
#     --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.csv" \
#     --dataset "CCT2-nontargeting" \
#     --epochs 50

# python $VAE_RESNET_PATH \
#     --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting-diane.csv" \
#     --dataset "CCT2-nontargeting-diane" \
#     --epochs 50

# python $VAE_RESNET_PATH \
#     --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_TRNT1-nontargeting.csv" \
#     --dataset "TRNT1-nontargeting" \
#     --epochs 50

# python $VAE_RESNET_PATH \
#     --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_cluster_109_mitotic.csv" \
#     --dataset "cluster_109_mitotic" \
#     --epochs 50

# python $VAE_RESNET_PATH \
#     --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_LMNA-nontargeting_interphase.csv" \
#     --dataset "LMNA-nontargeting" \
#     --epochs 50

# python $VAE_RESNET_PATH \
#     --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_PPP1CC-nontargeting_interphase.csv" \
#     --dataset "PPP1CC-nontargeting_interphase" \
#     --epochs 50

python $VAE_RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_interphase_100.csv" \
    --dataset "dataset_interphase_100" \
    --epochs 50

python $VAE_RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_mitotic_all.csv" \
    --dataset "dataset_mitotic_all" \
    --epochs 50

python $VAE_RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_interphase_500.csv" \
    --dataset "dataset_interphase_500" \
    --epochs 50