import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import ZarrCellDataset
from src.data.dataloader import collate_wrapper
from src.data.utils import read_config
from torch.utils.data import DataLoader
import torchvision.transforms as v2

# Create the dataset
dataset = ZarrCellDataset(
    parent_dir="/lab/barcheese01/aconcagua_results/primary_screen_patches",
    csv_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.csv",
    split='train',
    channels=[0, 1, 2, 3], 
    mask='min',
    normalizations=v2.Compose([v2.CenterCrop(96)]),
    interpolations=None,
    mean=None, 
    std=None
)

# Print the dataset length
print("Dataset length:", len(dataset))

# Save dataset info into a yaml file
dataset.save_info("/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.yaml")

# Create a DataLoader for the dataset
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=72,    
    collate_fn=collate_wrapper(
        metadata_keys=['gene', 'barcode', 'stage'], 
        images_keys=['cell_image'])
)

# Print the first batch size
for batch in dataloader:
    print("Batch dimensions", batch['cell_image'].shape)
    break

# Create the dataset
dataset = ZarrCellDataset(
    parent_dir="/lab/barcheese01/aconcagua_results/primary_screen_patches",
    csv_file="/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_interphase_100.csv",
    split='train',
    channels=[0, 1, 2, 3], 
    mask='min',
    normalizations=v2.Compose([v2.CenterCrop(96)]),
    interpolations=None,
    mean=None, 
    std=None
)

# Print the dataset length
print("Dataset length:", len(dataset))

# Save dataset info into a yaml file
dataset.save_info("/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_interphase_100.yaml")

# Create a DataLoader for the dataset
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=72,    
    collate_fn=collate_wrapper(
        metadata_keys=['gene', 'barcode', 'stage'], 
        images_keys=['cell_image'])
)

# Print the first batch size
for batch in dataloader:
    print("Batch dimensions", batch['cell_image'].shape)
    break