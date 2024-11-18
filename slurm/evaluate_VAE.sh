#!/bin/bash
#SBATCH --job-name=evaluate_vae
#SBATCH --partition=nvidia-t4-20
#SBATCH --time=10:00:00  # Increased time limit for sequential execution
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --output=out/evaluate_vae_%j.out

source /lab/barcheese01/miniconda3/etc/profile.d/conda.sh

conda activate ops_dl

VAES=(
    "20240925-185336_VAE_ResNet18_crop_size_96_nc_4_z_dim_30_lr_0.0001_beta_1_transform_min_loss_MSE_CCT2-nontargeting-diane:epoch_45"
    "20241113-193236_VAE_ResNet18_crop_size_96_nc_4_z_dim_10_lr_0.0001_beta_1_transform_min_loss_L1_dataset_mitotic_all:epoch_20"
    "20241109-141926_VAE_ResNet18_crop_size_96_nc_4_z_dim_30_lr_0.0001_beta_1_transform_min_loss_L1_dataset_interphase_100:epoch_51"
    "20241109-141926_VAE_ResNet18_crop_size_96_nc_4_z_dim_10_lr_0.0001_beta_1_transform_min_loss_L1_dataset_interphase_100:epoch_57"
)

# Set the path to your evaluation script
EVALUATE_VAE_PATH="/lab/barcheese01/mdiberna/OpticalPooledScreens_DL/scripts/evaluate/evaluate_VAE.py"

for VAE in "${VAES[@]}"; do
    IFS=':' read -ra VAE_PARTS <<< "$VAE"
    CHECKPOINT_DIR=${VAE_PARTS[0]}
    EPOCH=${VAE_PARTS[1]}
    echo "Evaluating VAE: $CHECKPOINT_DIR"
    echo "Epoch: $EPOCH"
    python $EVALUATE_VAE_PATH --checkpoint_dir "$CHECKPOINT_DIR" --epoch "$EPOCH"
done