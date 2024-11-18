#!/bin/bash
#SBATCH --job-name=resnet18_classify
#SBATCH --partition=nvidia-A6000-20
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=out/resnet18_classify_%j.out

source /lab/barcheese01/miniconda3/etc/profile.d/conda.sh

conda activate ops_dl

RESNET_PATH="/lab/barcheese01/mdiberna/OpticalPooledScreens_DL/scripts/train/train_resnet18_classifier.py"

Run the Python script with custom arguments
python $RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.csv" \
    --dataset "CCT2-nontargeting" \
    --epochs 50

python $RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting-diane.csv" \
    --dataset "CCT2-nontargeting-diane" \
    --epochs 50

python $RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_TRNT1-nontargeting.csv" \
    --dataset "TRNT1-nontargeting" \
    --epochs 50

python $RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_cluster_109_mitotic.csv" \
    --dataset "cluster_109_mitotic" \
    --epochs 50

python $RESNET_PATH \
    --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_LMNA-nontargeting_interphase.csv" \
    --dataset "LMNA-nontargeting_interphase" \
    --epochs 50

# python $RESNET_PATH \
#     --csv_file "/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_PPP1CC-nontargeting_interphase.csv" \
#     --dataset "PPP1CC-nontargeting_interphase" \
#     --epochs 50