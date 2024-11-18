import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.splitter import DatasetTable, DatasetSplitter

# # Generate the initial dataset table
# table = DatasetTable(
#     parent_dir="/lab/barcheese01/aconcagua_results/primary_screen_patches",
#     output_dir="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits",
#     num_workers=-1    
# )
# table.generate_table()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.csv",
#     gene_list=["nontargeting", "CCT2"],
#     balance_classes=False,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     number_nt_guides=10
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting_interphase.csv",
#     gene_list=["nontargeting", "CCT2"],
#     stage_list=["interphase"],
#     balance_classes=False,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     number_nt_guides=10
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting_interphase_balanced.csv",
#     gene_list=["nontargeting", "CCT2"],
#     stage_list=["interphase"],
#     balance_classes=True,
#     train_ratio=0.7,
#     val_ratio=0.15
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_KIF11-nontargeting_mitotic.csv",
#     gene_list=["nontargeting", "KIF11"],
#     stage_list=["mitotic"],
#     balance_classes=False,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     number_nt_guides=10
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_KIF11-nontargeting_mitotic_balanced.csv",
#     gene_list=["nontargeting", "KIF11"],
#     stage_list=["mitotic"],
#     balance_classes=True,
#     train_ratio=0.7,
#     val_ratio=0.15
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_TRNT1-nontargeting.csv",
#     gene_list=["nontargeting", "TRNT1"],
#     balance_classes=False,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     number_nt_guides=10
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_cluster_109_mitotic.csv",
#     gene_list=['nontargeting', 'AURKA', 'CEP85', 'CEP192', 'ERH', 'GAK', 'KIF11', 'MYBL2', 'NEDD1', 'PLK4', 'PSMD1', 'PSMD11', 'PSMD12', 'PSMD3', 'PSMD7', 'PSMD8', 'SASS6', 'STIL', 'TACC3', 'TPX2', 'TUBGCP2', 'TUBGCP3', 'TUBGCP6', 'ZNF335'],
#     stage_list=["mitotic"],
#     balance_classes=False,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     number_nt_guides=22
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_LMNA-nontargeting_interphase.csv",
#     gene_list=['nontargeting', 'LMNA'],
#     stage_list=["interphase"],
#     balance_classes=False,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     number_nt_guides=10
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_PPP1CC-nontargeting_interphase.csv",
#     gene_list=['nontargeting', 'PPP1CC'],
#     stage_list=["interphase"],
#     balance_classes=False,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     number_nt_guides=10
# )
# splitter.generate_split()

# splitter = DatasetSplitter(
#     parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
#     output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_interphase_100.csv",
#     stage_list=["interphase"],
#     sample_size=100,
#     train_ratio=0.7,
#     val_ratio=0.15
# )
# splitter.generate_split()

splitter = DatasetSplitter(
    parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
    output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_mitotic_all.csv",
    stage_list=["mitotic"],
    balance_classes=False,
    train_ratio=0.7,
    val_ratio=0.15
)
splitter.generate_split()

splitter = DatasetSplitter(
    parent_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_5050.csv",
    output_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_interphase_500.csv",
    stage_list=["interphase"],
    sample_size=500,
    balance_classes=False,
    train_ratio=0.7,
    val_ratio=0.15
)
splitter.generate_split()