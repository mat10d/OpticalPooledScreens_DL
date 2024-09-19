import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.splitter import DatasetTable, DatasetSplitter

# Generate the initial dataset table
table = DatasetTable(
    parent_dir="/lab/barcheese01/aconcagua_results/primary_screen_patches",
    output_dir="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits",
    num_workers=-1    
)
table.generate_table()

# Split the dataset for CCT2 and nontargeting genes
splitter = DatasetSplitter(
    parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
    output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.csv",
    gene_list=["nontargeting", "CCT2"],
    balance_classes=True,
    train_ratio=0.7,
    val_ratio=0.15
)
splitter.generate_split()

# Split the dataset for interphase stage with sample size 100
splitter = DatasetSplitter(
    parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
    output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_interphase_100.csv",
    stage_list=["interphase"],
    sample_size=100,
    train_ratio=0.7,
    val_ratio=0.15
)
splitter.generate_split()