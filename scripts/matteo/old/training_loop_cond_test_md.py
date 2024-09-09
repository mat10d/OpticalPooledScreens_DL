#%%
# Imports
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import functional as F
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import yaml

# Import your custom modules
from models.splitter_static import DatasetSplitter
from models.dataset_static import ZarrCellDataset
from models.dataloader_static import collate_wrapper
from models.models_contrastive import VAEmodel, Encoder, Decoder

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Yaml file reader
def read_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract 'Dataset mean' and 'Dataset std' from the config
    mean = config['Dataset mean'][0]  # Access the first (and only) element of the list
    std = config['Dataset std'][0]

    # Split the strings and convert to floats
    mean = [float(i) for i in mean.split()]
    std = [float(i) for i in std.split()]
    
    # Convert to ndarrays
    mean = np.array(mean)
    std = np.array(std)

    return mean, std

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# Basic values for logging 
parent_dir = '/mnt/efs/dlmbl/S-md/'
output_path = parent_dir + 'training_logs/'
model_name = "static_vanilla_vae_md_10"
run_name= "initial_params"
find_port = True

# Function to find an available port
def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# # Launch TensorBoard on the browser
# def launch_tensorboard(log_dir):
#     port = find_free_port()
#     tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
#     process = subprocess.Popen(tensorboard_cmd, shell=True)
#     print(
#         f"TensorBoard started at http://localhost:{port}. \n"
#         "If you are using VSCode remote session, forward the port using the PORTS tab next to TERMINAL."
#     )
#     return process

# # Launch tensorboard and click on the link to view the logs.
# if find_port:
#     tensorboard_process = launch_tensorboard("embed_time_static_runs")
# logger = SummaryWriter(f"embed_time_static_runs/{model_name}")
#%%
# Define variables for the dataset read in
csv_file = '/mnt/efs/dlmbl/G-et/csv/split_804.csv'
split = 'train'
channels = [0, 1, 2, 3]
transform = "masks"
crop_size = 100
normalizations = v2.Compose([v2.CenterCrop(crop_size)])
yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
dataset_mean, dataset_std = read_config(yaml_file_path)
#%%

# Create the dataset
dataset = ZarrCellDataset(parent_dir, csv_file, split, channels, transform, normalizations, None, dataset_mean, dataset_std)

# Define the metadata keys
metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['nuclei_image']

# Create a DataLoader for the dataset
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_wrapper(metadata_keys, images_keys)
)

# Hyperparameters
batch_size = 16
learning_rate = 1e-4
num_epochs = 100
latent_dim = 32
base_channel_size = 32
step_size = 1000  # for cyclic KL annealing
#%%

# Model setup
from models.models_contrastive import VAEmodel, Encoder, Decoder
from models.model_VAE_resnet18 import VAEResNet18

model = VAEmodel(
    model_name="static_vanilla_vae_md_10",
    optimizer_param={"optimizer": "Adam", "lr": learning_rate},
    latent_dim=latent_dim,
    base_channel_size=base_channel_size,
    num_input_channels=4,
    image_size=96,
    step_size=step_size,
    encoder_class=Encoder,
    decoder_class=Decoder
)
model = model.to(device)

#%%
dataset[0]['nuclei_image'].unsqueeze(dim=0).shape
#%%
# use torchview to visualize the model and save the image
# import torchview
# torchview.draw_graph(
#     model,
#     dataset[0]['nuclei_image'].unsqueeze(dim=0),
#     roll=True,
#     depth=3,  # adjust depth to zoom in.
#     device="cpu",
#     save_graph=True,
#     filename="graphs/cond_test_md_96"
# )

vae = VAEResNet18(nc = 4, z_dim = 10 ).to(device)

torchview.draw_graph(
    vae,
    dataset[0]['cell_image'].unsqueeze(dim=0),
    roll=True,
    depth=3,  # adjust depth to zoom in.
    device="cpu",
    save_graph=True,
    filename="graphs/vae_100_md"
)
#%%
# Optimizer
optimizer = model.configure_optimizer()

# TensorBoard setup
log_dir = f"embed_time_static_runs/gt_vanilla_vae_md_10/initial_params_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir)

# Define training function
training_log = []
epoch_log = []
loss_per_epoch = 0

def train(
        epoch, 
        model, 
        loader, 
        optimizer, 
        log_interval=100, 
        log_image_interval=20, 
        tb_logger=None,  # Changed from writer to None as default
        device=device,
        early_stop=False,
        training_log=training_log,
        epoch_log=epoch_log,
        loss_per_epoch=loss_per_epoch):
    
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(loader):
        data = batch['nuclei_image'].to(device)
        optimizer.zero_grad()
        
        # Use the model's train_step method
        loss, metrics = model.train_step(data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # Log to console
        if batch_idx % 5 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} "
                  f"({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # Log to training_log
        if batch_idx % log_interval == 0:
            row = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'len_data': len(data),
                'len_dataset': len(loader.dataset),
                'loss': loss.item() / len(data),
                **{k: v / len(data) for k, v in metrics.items()}
            }
            training_log.append(row)
        
        # Log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_idx
            tb_logger.add_scalar("train/loss", loss.item(), step)
            for key, value in metrics.items():
                tb_logger.add_scalar(f"train/{key}", value, step)
            
            # Log images
            if step % log_image_interval == 0:
                with torch.no_grad():
                    x_hat, _, _ = model(data)
                    for i in range(model.num_input_channels):
                        tb_logger.add_images(f"input_{i}", data[:, i:i+1, ...], step)
                        tb_logger.add_images(f"reconstruction_{i}", x_hat[:, i:i+1, ...], step)
                
                # Add embedding (adjust as necessary)
                metadata = list(zip(batch.get('gene', []), batch.get('barcode', []), batch.get('stage', [])))
                embeddings = model.get_image_embedding(data)
                tb_logger.add_embedding(embeddings, metadata=metadata, label_img=data[:, 2:3, ...], global_step=step)
        
        if early_stop and batch_idx > 5:
            print("Stopping training early!")
            break
    
    # Log epoch summary
    avg_loss = train_loss / len(loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    epoch_log.append({'epoch': epoch, 'Average Loss': avg_loss})
    if tb_logger is not None:
        tb_logger.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    return avg_loss  # Return the average loss for the epoch

# Training loop
num_epochs = 2  # Adjust as needed

# You can uncomment and adjust these paths when you're ready to save checkpoints and logs
# output_path = "path/to/your/output/"
# folder_suffix = datetime.now().strftime("%Y%m%d_%H%M_") + run_name
# checkpoint_path = os.path.join(output_path, "checkpoints", "static", folder_suffix)
# log_path = os.path.join(output_path, "logs", "static", folder_suffix)
# os.makedirs(checkpoint_path, exist_ok=True)
# os.makedirs(log_path, exist_ok=True)

for epoch in range(1, num_epochs + 1):
    avg_loss = train(epoch, model, dataloader, optimizer)
    loss_per_epoch = avg_loss
    
    # You can uncomment this section when you're ready to save checkpoints
    # checkpoint = {
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': loss_per_epoch 
    # }
    # torch.save(checkpoint, os.path.join(checkpoint_path, f"epoch_{epoch}_checkpoint.pth"))

print("Training completed!")