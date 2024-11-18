#!/bin/bash
#SBATCH --job-name=evaluate_classifier
#SBATCH --partition=nvidia-L40S-20
#SBATCH --time=10:00:00  # Increased time limit for sequential execution
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=out/evaluate_classifier_%j.out

source /lab/barcheese01/miniconda3/etc/profile.d/conda.sh

conda activate ops_dl

CLASSIFIERS=(
    "20240924-191644_Classifier_VGG2D_crop_96_lr_0.0001_transform_min_CCT2-nontargeting:epoch_15"
    "20240924-213458_Classifier_VGG2D_crop_96_lr_0.0001_transform_min_CCT2-nontargeting-diane:epoch_15"
    "20240924-232118_Classifier_VGG2D_crop_96_lr_0.0001_transform_min_TRNT1-nontargeting:epoch_15"
    "20240925-124738_Classifier_VGG2D_crop_96_lr_0.0001_transform_min_cluster_109_mitotic:epoch_15"
    "20240925-130315_Classifier_VGG2D_crop_96_lr_0.0001_transform_min_LMNA-nontargeting_interphase:epoch_15"
    
    "20240925-091000_Classifier_ResNet18_crop_96_lr_0.0001_transform_min_CCT2-nontargeting:epoch_15"
    "20240925-103108_Classifier_ResNet18_crop_96_lr_0.0001_transform_min_CCT2-nontargeting-diane:epoch_15"
    "20240925-113233_Classifier_ResNet18_crop_96_lr_0.0001_transform_min_TRNT1-nontargeting:epoch_15"
    "20240925-130631_Classifier_ResNet18_crop_96_lr_0.0001_transform_min_cluster_109_mitotic:epoch_15"
    "20240925-131508_Classifier_ResNet18_crop_96_lr_0.0001_transform_min_LMNA-nontargeting_interphase:epoch_15"
)

# Set the path to your evaluation script
EVALUATE_CLASSIFIER_PATH="/lab/barcheese01/mdiberna/OpticalPooledScreens_DL/scripts/evaluate/evaluate_classifier.py"

for CLASSIFIER in "${CLASSIFIERS[@]}"; do
    IFS=':' read -ra CLASSIFIER_PARTS <<< "$CLASSIFIER"
    CHECKPOINT_DIR=${CLASSIFIER_PARTS[0]}
    EPOCH=${CLASSIFIER_PARTS[1]}

    echo "Evaluating classifier: $CHECKPOINT_DIR"
    echo "Epoch: $EPOCH"

    python $EVALUATE_CLASSIFIER_PATH --checkpoint_dir "$CHECKPOINT_DIR" --epoch "$EPOCH"
done