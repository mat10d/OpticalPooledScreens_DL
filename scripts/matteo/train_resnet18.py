# Imports
import os
from models.splitter_static import DatasetSplitter
from models.dataset_static import ZarrCellDataset
from models.dataloader_static import collate_wrapper
from models.model_VAE_resnet18 import VAEResNet18
from models.static_utils import read_config
import piq
from ignite.metrics import SSIM
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchview
import yaml
import argparse

loss_ssim = piq.SSIMLoss()

parser = argparse.ArgumentParser(description='VAE Training')
parser.add_argument('--z_dim', type=int, default=30, help='Dimension of latent space')
parser.add_argument('--loss_type', type=str, default='MSE', choices=['L1', 'MSE', 'SSIM'], help='Type of reconstruction loss')
parser.add_argument('--crop_size', type=int, default=64, help='Size of image crop')
parser.add_argument('--beta', type=float, default=1e-5, help='Weight of KL divergence in loss')
parser.add_argument('--transform', type=str, default='min', help='Masking type')
args = parser.parse_args()

# Define metadata keys
metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['cell_image']
crop_size = args.crop_size
channels = [0, 1, 2, 3]
split = 'train'
parent_dir = '/mnt/efs/dlmbl/S-md/'
normalizations = v2.Compose([v2.CenterCrop(crop_size)])
yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
dataset_mean, dataset_std = read_config(yaml_file_path)
dataset = "benchmark"
csv_file = f"/mnt/efs/dlmbl/G-et/csv/dataset_split_{dataset}.csv"
output_dir = "/mnt/efs/dlmbl/G-et/"
find_port = True

# Hyperparameters
batch_size = 16
num_workers = 8
epochs = 20
nc = 4
z_dim = args.z_dim
lr = 1e-4
beta = args.beta
alpha = 0.5
loss_type = args.loss_type
transform = args.transform
model_name = "VAE_ResNet18"

# run name concatenates all hyperparameters
run_name = f"{model_name}_crop_size_{crop_size}_nc_{nc}_z_dim_{z_dim}_lr_{lr}_beta_{beta}_transform_{transform}_loss_{loss_type}_{dataset}"

folder_suffix = datetime.now().strftime("%Y%m%d_%H%M_") + run_name
log_path = output_dir + "logs/static/Matteo/"+ folder_suffix + "/"
checkpoint_path = output_dir + "checkpoints/static/Matteo/" + folder_suffix + "/"

# Check and create necessary directories
for path in [log_path, checkpoint_path]:
    if not os.path.exists(path):
        os.makedirs(path)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# Function to find an available port
def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# Launch TensorBoard on the browser
def launch_tensorboard(log_dir):
    port = find_free_port()
    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(tensorboard_cmd, shell=True)
    print(f"TensorBoard started at http://localhost:{port}.")
    print("If using VSCode remote session, forward the port using the PORTS tab next to TERMINAL.")
    return process

# Launch tensorboard and click on the link to view the logs.
if find_port:
    tensorboard_process = launch_tensorboard("embed_time_static_runs")
logger = SummaryWriter(f"embed_time_static_runs/{run_name}")

# Create the dataset
dataset = ZarrCellDataset(parent_dir, csv_file, split, channels, transform, normalizations, None, dataset_mean, dataset_std)

# Create a DataLoader for the dataset
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,    
    collate_fn=collate_wrapper(metadata_keys, images_keys)
)

# Create the model
vae = VAEResNet18(nc = nc, z_dim = z_dim)

torchview.draw_graph(
    vae,
    dataset[0]['cell_image'].unsqueeze(dim=0),
    roll=True,
    depth=3,  # adjust depth to zoom in.
    device="cpu",
    save_graph=True,
    filename="graphs/" + run_name,
)

vae = vae.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

def loss_function(recon_x, x, mu, logvar, loss_type=loss_type): 
    if loss_type == "MSE":
        RECON = F.mse_loss(recon_x, x, reduction='mean')
    elif loss_type == "L1":
        RECON = F.l1_loss(recon_x, x, reduction='mean')
    elif loss_type == "SSIM":
        # normalize x for ssim (remember shape is BxCxHxW)
        x_norm = (x - x.min()) / (x.max() - x.min())
        recon_x_norm = (recon_x - recon_x.min()) / (recon_x.max() - recon_x.min())
        ssim = loss_ssim(recon_x_norm, x_norm)
        RECON = F.l1_loss(recon_x, x, reduction='mean') + ssim * alpha
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON, KLD   

# Define training function
training_log = []
epoch_log = []
loss_per_epoch = 0
def train(
    epoch,
    model = vae,
    loader = dataloader,
    optimizer = optimizer,
    loss_function = loss_function,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=device,
    early_stop=False,
    training_log = training_log,
    epoch_log = epoch_log,
    loss_per_epoch = loss_per_epoch
    ):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch['cell_image'].to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)
        RECON, KLD  = loss_function(recon_batch, data, mu, logvar)
        loss = RECON + KLD * beta
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        loss_per_epoch = train_loss / len(dataloader.dataset)
        
        # log to console
        if batch_idx % 5 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.item(),
                )
            )
            
        if batch_idx % log_interval == 0:
            row = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'len_data': len(batch['cell_image']),
                'len_dataset': len(loader.dataset),
                'loss': loss.item() / len(batch['cell_image']),
                'RECON': RECON.item() / len(batch['cell_image']),
                'KLD': KLD.item() / len(batch['cell_image'])
            }
            training_log.append(row)

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_idx
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="train_RECON", scalar_value=RECON.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="train_KLD", scalar_value=KLD.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                input_image = data.to("cpu").detach()       
                predicted_image = recon_batch.to("cpu").detach()

                tb_logger.add_images(
                    tag="input_channel_0", img_tensor=input_image[0:3,0:1,...], global_step=step
                )
                tb_logger.add_images(
                    tag= "reconstruction_0", img_tensor=predicted_image[0:3,0:1,...], global_step=step
                )
                
                tb_logger.add_images(
                    tag="input_1", img_tensor=input_image[0:3,1:2,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_1", img_tensor=predicted_image[0:3,1:2,...], global_step=step
                )

                tb_logger.add_images(
                    tag="input_2", img_tensor=input_image[0:3,2:3,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_2", img_tensor=predicted_image[0:3,2:3,...], global_step=step
                )
                tb_logger.add_images(
                    tag="input_3", img_tensor=input_image[0:3,3:4,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_3", img_tensor=predicted_image[0:3,3:4,...], global_step=step
                )
                metadata = [list(item) for item in zip(batch['gene'], batch['barcode'], batch['stage'])]
                tb_logger.add_embedding(
                    torch.flatten(mu, start_dim=1), metadata=metadata, label_img = input_image[:,2:3,...], global_step=step, metadata_header=metadata_keys
                )
                           
        # early stopping
        if early_stop and batch_idx > 5:
            print("Stopping test early!")
            break
    
    # save the DF
    epoch_raw = {
                'epoch': epoch,
                'Average Loss': train_loss / len(dataloader.dataset)}
    epoch_log.append(epoch_raw)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))

for epoch in range(epochs):
    train(epoch, log_interval=100, log_image_interval=20, tb_logger=logger)
    filename_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_") + "epoch_"+str(epoch) + "_"
    training_logDF = pd.DataFrame(training_log)
    training_logDF.to_csv(log_path + filename_suffix+"training_log.csv", index=False)

    epoch_logDF = pd.DataFrame(epoch_log)
    epoch_logDF.to_csv(log_path + filename_suffix+"epoch_log.csv", index=False)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_per_epoch 
    }
    torch.save(checkpoint, checkpoint_path + filename_suffix + str(epoch) + "checkpoint.pth")