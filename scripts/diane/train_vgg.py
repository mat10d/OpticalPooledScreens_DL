# %%
from models.dataset_static import ZarrCellDataset
from models.dataloader_static import collate_wrapper
from funlib.learn.torch.models import Vgg2D
from torchvision.transforms import v2
from models.static_utils import read_config
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# %% Load the dataset
# Define the metadata keys
metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['cell_image']
crop_size = 96
normalizations = v2.Compose([v2.CenterCrop(crop_size)])
yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
dataset = "benchmark_nontargeting_barcode"
csv_file = f"/mnt/efs/dlmbl/G-et/csv/dataset_split_{dataset}.csv"
label_type = 'barcode'
balance_classes = True

save_dir = Path(f"/mnt/efs/dlmbl/G-et/da_testing/vgg2d_{dataset}/{label_type}_{balance_classes}")
save_dir.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(csv_file)
class_names = df[label_type].sort_values().unique().tolist()
num_classes = len(class_names)

print(f"Class names: {class_names}")

# Hyperparameters
batch_size = 16
num_workers = 16
epochs = 30

# %% Load the training dataset
# Create the dataset
dataset_mean, dataset_std = read_config(yaml_file_path)
dataset = ZarrCellDataset(
    parent_dir = '/mnt/efs/dlmbl/S-md/',
    csv_file = csv_file, 
    split='train',
    channels=[0, 1, 2, 3], 
    mask='min', 
    normalizations=normalizations,
    interpolations=None, 
    mean=dataset_mean, 
    std=dataset_std
)

if balance_classes:
    df = pd.read_csv(csv_file)
    df = df[df['split'] == 'train']
    all_labels = df[label_type].tolist()
    weights = [1 / all_labels.count(label) for label in all_labels]
    print(f"Weighting classes: {np.unique(weights)}")
    balanced_sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        sampler=balanced_sampler,
        collate_fn=collate_wrapper(metadata_keys, images_keys),
        drop_last=True
    )
else:
    # Create a DataLoader for the dataset
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_wrapper(metadata_keys, images_keys)
    )

# %% Load the validation dataset
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
    shuffle=True, 
    num_workers=num_workers,
    collate_fn=collate_wrapper(metadata_keys, images_keys)
)
# %%
# print the length of both datasets
len(dataset), len(val_dataset)

# %% Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Vgg2D(
    input_size=(96, 96),
    input_fmaps=4,
    output_classes=num_classes,
)
model = model.to(device)

# %% Define the loss function
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# %% Training loop
losses = []
val_losses = []
val_accuracies = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader)):
        images, labels = batch['cell_image'], batch[label_type]
        labels = torch.tensor(
            [class_names.index(label) for label in labels]
        )
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch}, loss: {epoch_loss / len(dataloader)}")
    losses.append(epoch_loss / len(dataloader))

    model.eval()
    epoch_val_loss = 0
    correct = 0
    with torch.inference_mode():
        for batch in tqdm(val_dataloader, desc=f"Validation", total=len(val_dataloader)):
            images, labels = batch['cell_image'], batch[label_type]
            labels = torch.tensor(
                [class_names.index(label) for label in labels]
            )
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_function(output, labels)
            epoch_val_loss += loss.item()

            correct += (output.argmax(dim=1) == labels).sum().item()
    print(f"Validation loss: {epoch_val_loss / len(val_dataloader)}")
    val_losses.append(epoch_val_loss / len(val_dataloader))
    print(f"Validation accuracy: {correct / len(val_dataset)}")
    val_accuracies.append(correct / len(val_dataset))

    # Save the model
    state_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_loss': epoch_loss / len(dataloader),
        'epoch_val_loss': epoch_val_loss / len(val_dataloader),
        'val_accuracy': correct / len(val_dataset)
    }
    torch.save(state_dict, save_dir / f"{epoch}.pth")


# %% Plot the loss
plt.plot(losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.show()
plt.plot(val_accuracies, label="Validation accuracy")
plt.legend()
plt.show()

# %% Save the losses and accuracies
with open(save_dir / "metrics.csv", "w") as f:
    f.write("epoch,loss,val_loss,val_accuracy\n")
    for i in range(epochs):
        f.write(f"{i},{losses[i]},{val_losses[i]},{val_accuracies[i]}\n")
