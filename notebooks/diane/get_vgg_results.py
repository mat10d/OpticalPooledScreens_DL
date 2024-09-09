# %%  [markdown]
# Loading the results of vgg experiments and showing their losses, accuracies, and confusion matrices.
# 
# %% 
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from models.dataset_static import ZarrCellDataset
from models.dataloader_static import collate_wrapper
from funlib.learn.torch.models import Vgg2D
from models.static_utils import read_config
from torchvision import transforms as v2
import seaborn as sns

# %% Utilities
def plot_metrics(metrics): 
    metrics.plot(subplots=True, figsize=(10, 10))
    plt.show()

def load_best_checkpoint(directory, metrics):
    # get epoch in metric with highest val_accuracy
    best_index = metrics['val_accuracy'].idxmax()
    best_epoch = metrics['epoch'][best_index]
    checkpoint = directory / f"{best_epoch}.pth"
    return checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_confusion_matrix(model, val_dataloader, class_names, label_type, normalize='true'):
    model.eval()
    predictions = []
    labels = []

    for batch in tqdm(val_dataloader, desc="Validation", total=len(val_dataloader)):
        images, batch_labels = batch['cell_image'], batch[label_type]
        batch_labels = torch.tensor(
            [class_names.index(label) for label in batch_labels]
        )
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        output = model(images)
        predictions.append(output.argmax(dim=1).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())

    cm = confusion_matrix(np.concatenate(labels), np.concatenate(predictions), normalize=normalize)
    return cm


def create_dataloader(dataset, label_type, batch_size=16, num_workers=8, balance_dataset=True):
    csv_file = f"/mnt/efs/dlmbl/G-et/csv/dataset_split_{dataset}.csv"
    subdir = Path(f"/mnt/efs/dlmbl/G-et/da_testing/vgg2d_{dataset}/{label_type}_{balance_dataset}")
    df = pd.read_csv(csv_file)
    class_names = df[label_type].sort_values().unique().tolist()
    num_classes = len(class_names)

    metadata_keys = ['gene', 'barcode', 'stage']
    images_keys = ['cell_image']
    crop_size = 96
    normalizations = v2.Compose([v2.CenterCrop(crop_size)])
    yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
    dataset = "benchmark"
    dataset_mean, dataset_std = read_config(yaml_file_path)

    val_dataset = ZarrCellDataset(
        parent_dir = '/mnt/efs/dlmbl/S-md/',
        csv_file = csv_file, 
        split='val',
        channels=[0, 1, 2, 3], 
        mask='min', 
        normalizations=normalizations,
        interpolations=None, 
        mean=dataset_mean, 
        std=dataset_std
    )

    # Create a DataLoader for the validation dataset
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_wrapper(metadata_keys, images_keys),
        drop_last=False
    )
    return subdir, val_dataloader, class_names, num_classes

# %% Setup happens here
dataset = "benchmark_nontargeting_barcode"
label_type = 'barcode'
batch_size = 16
num_workers = 8
balance_dataset = True

subdir, val_dataloader, class_names, num_classes = create_dataloader(dataset, label_type, batch_size, num_workers)

metrics = pd.read_csv(subdir / "metrics.csv")
plot_metrics(metrics)
# %% Get the model to load the best checkpoint, create a confusion matrix
checkpoint = load_best_checkpoint(subdir, metrics)
model = Vgg2D(
    input_size=(96, 96),
    input_fmaps=4,
    output_classes=num_classes,
)
model = model.to(device)
model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
model.eval()

cm = get_confusion_matrix(model, val_dataloader, class_names, label_type)

# %% Validation loop for confusion matrix
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
# Set tick labels
# plt.xticks(np.arange(num_classes) + 0.5, class_names)
# plt.yticks(np.arange(num_classes) + 0.5, class_names)
plt.show()

# %%
len(class_names)
# %%
df = pd.read_csv(f"/mnt/efs/dlmbl/G-et/csv/dataset_split_{dataset}_{balance_dataset}.csv")
df = df[df.split == 'val']
df.barcode.value_counts()
# %%
dataset
# %%
