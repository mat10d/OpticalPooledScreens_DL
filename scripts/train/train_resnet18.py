import os
import sys
import subprocess
import socket
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as v2
import torchview
import piq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import ZarrCellDataset
from src.data.dataloader import collate_wrapper
from src.data.utils import read_config
from src.models.VAE_resnet18 import VAEResNet18

parser = argparse.ArgumentParser(description='VAE Training')

# Dataset and file paths
parser.add_argument('--parent_dir', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches', help='Parent directory of the dataset')
parser.add_argument('--csv_file', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.csv', help='CSV file with dataset information')
parser.add_argument('--yaml_file', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.yaml', help='YAML file with dataset info')
parser.add_argument('--output_dir', type=str, default='/lab/barcheese01/aconcagua_results', help='Output directory')
parser.add_argument('--dataset', type=str, default='CCT2-nontargeting', help='Dataset name')

# Model parameters
parser.add_argument('--model_name', type=str, default='VAE_ResNet18', help='Model name')
parser.add_argument('--z_dim', type=int, default=30, help='Dimension of latent space')
parser.add_argument('--nc', type=int, default=4, help='Number of channels in the input')

# Training parameters
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loading')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--beta', type=float, default=1, help='Weight of KL divergence in loss')
parser.add_argument('--alpha', type=float, default=1e-3, help='Alpha parameter for SSIM loss')
parser.add_argument('--loss_type', type=str, default='MSE', choices=['L1', 'MSE', 'SSIM'], help='Type of reconstruction loss')

# Data processing parameters
parser.add_argument('--crop_size', type=int, default=96, help='Size of image crop')
parser.add_argument('--transform', type=str, default='min', help='Masking type')
parser.add_argument('--metadata_keys', nargs='+', default=['gene', 'barcode', 'stage'], help='Metadata keys')
parser.add_argument('--images_keys', nargs='+', default=['cell_image'], help='Image keys')
parser.add_argument('--channels', nargs='+', type=int, default=[0, 1, 2, 3], help='Channels to use')

# Misc
parser.add_argument('--find_port', type=bool, default=True, help='Whether to find an available port')

args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to find an available port
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# Launch TensorBoard on the browser
def launch_tensorboard(log_dir):
    port = find_free_port()
    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port} --host=0.0.0.0"
    process = subprocess.Popen(tensorboard_cmd, shell=True)
    print(f"TensorBoard started at http://adrenaline.wi.mit.edu:{port}.")
    print("If using VSCode remote session, forward the port using the PORTS tab next to TERMINAL.")
    return process

# Setup paths and logging
run_name = f"{args.model_name}_crop_size_{args.crop_size}_nc_{args.nc}_z_dim_{args.z_dim}_lr_{args.lr}_beta_{args.beta}_transform_{args.transform}_loss_{args.loss_type}_{args.dataset}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_suffix = timestamp + "_" + run_name
log_path = os.path.join(args.output_dir, "logs", folder_suffix)
checkpoint_path = os.path.join(args.output_dir, "checkpoints", folder_suffix)
tensorboard_path = os.path.join(args.output_dir, "tensorboard")

for path in [log_path, checkpoint_path, tensorboard_path]:
    os.makedirs(path, exist_ok=True)

# Launch tensorboard
if args.find_port:
    tensorboard_process = launch_tensorboard(tensorboard_path)
logger = SummaryWriter(log_dir=tensorboard_path)

# Create datasets
dataset_mean, dataset_std = read_config(args.yaml_file)
normalizations = v2.Compose([v2.CenterCrop(args.crop_size)])

train_dataset = ZarrCellDataset(
    parent_dir=args.parent_dir,
    csv_file=args.csv_file,
    split='train',
    channels=args.channels, 
    mask=args.transform,
    normalizations=normalizations,
    interpolations=None,
    mean=dataset_mean, 
    std=dataset_std
)

val_dataset = ZarrCellDataset(
    parent_dir=args.parent_dir,
    csv_file=args.csv_file,
    split='val',
    channels=args.channels, 
    mask=args.transform,
    normalizations=normalizations,
    interpolations=None,
    mean=dataset_mean, 
    std=dataset_std
)

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.num_workers,    
    collate_fn=collate_wrapper(
        metadata_keys=args.metadata_keys, 
        images_keys=args.images_keys)
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=args.num_workers,    
    collate_fn=collate_wrapper(
        metadata_keys=args.metadata_keys, 
        images_keys=args.images_keys)
)

# Create the model
vae = VAEResNet18(nc=args.nc, z_dim=args.z_dim)

torchview.draw_graph(
    vae,
    train_dataset[0]['cell_image'].unsqueeze(0),
    roll=True,
    depth=3,  # adjust depth to zoom in.
    device="cpu",
    save_graph=True,
    filename="graphs/" + run_name,
)

vae = vae.to(device)

# Define the optimizer
optimizer = optim.Adam(vae.parameters(), lr=args.lr)

# Define the loss function
def loss_function(recon_x, x, mu, logvar): 
    if args.loss_type == "MSE":
        RECON = F.mse_loss(recon_x, x, reduction='none')
    elif args.loss_type == "L1":
        RECON = F.l1_loss(recon_x, x, reduction='none')
    elif args.loss_type == "SSIM":
        loss_ssim = piq.SSIMLoss()
        x_norm = (x - x.min()) / (x.max() - x.min())
        recon_x_norm = (recon_x - recon_x.min()) / (recon_x.max() - recon_x.min())
        ssim = loss_ssim(recon_x_norm, x_norm)
        RECON = F.l1_loss(recon_x, x, reduction='none') + ssim * args.alpha
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = (1, 2, 3))
    return RECON, KLD

# Define training function
training_log = []
epoch_log = []

def train(epoch, print_interval=10, log_interval=100):
    vae.train()
    train_loss = 0
    train_recon = 0
    train_kld = 0
    for batch_idx, batch in enumerate(train_dataloader):
        data = batch['cell_image'].to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)        
        RECON, KLD = loss_function(recon_batch, data, mu, logvar)

        denominator = torch.prod(torch.tensor(data.shape)) if torch.prod(torch.tensor(data.shape)) > torch.prod(torch.tensor(mu.shape)) else torch.prod(torch.tensor(mu.shape))
        RECON = RECON.sum() / denominator
        KLD = KLD.sum() / denominator

        loss = RECON + KLD * args.beta
        
        loss.backward()
        train_loss += loss.item()
        train_recon += RECON.item()
        train_kld += KLD.item()
        optimizer.step()
        
        # Log to console
        if batch_idx % print_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_dataloader.dataset),
                    100.0 * batch_idx / len(train_dataloader),
                    loss.item(),
                )
            )
            
        # Log detailed metrics
        if batch_idx % log_interval == 0:
            step = epoch * len(train_dataloader) + batch_idx
            row = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'len_data': len(batch['cell_image']),
                'len_dataset': len(train_dataloader.dataset),
                'loss': loss.item(),
                'RECON': RECON.item(),
                'KLD': KLD.item()
            }
            training_log.append(row)

            logger.add_scalar("train_loss", loss.item(), step)
            logger.add_scalar("train_RECON", RECON.item(), step)
            logger.add_scalar("train_KLD", KLD.item(), step)
            
 
            input_image = data.to("cpu").detach()       
            predicted_image = recon_batch.to("cpu").detach()

            for i in range(args.nc): 
                logger.add_images(f"input_{i}", input_image[0:3,i:i+1,...], step)
                logger.add_images(f"reconstruction_{i}", predicted_image[0:3,i:i+1,...], step)

            # Log embeddings
            metadata = [list(item) for item in zip(batch['gene'], batch['barcode'], batch['stage'])]
            logger.add_embedding(
                torch.flatten(mu, start_dim=1), 
                metadata=metadata, 
                label_img=input_image[:,2:3,...], 
                global_step=step, 
                metadata_header=args.metadata_keys
            )
                              
    train_loss /= len(train_dataloader)
    train_recon /= len(train_dataloader)
    train_kld /= len(train_dataloader)
    
    # Save epoch summary
    epoch_raw = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_recon': train_recon,
        'train_kld': train_kld
    }
    epoch_log.append(epoch_raw)

    print('====> Epoch: {} Average loss: {:.4f} Train RECON: {:.4f} Train KLD: {:.4f}'.format(epoch, train_loss, train_recon, train_kld))
    
    return train_loss, train_recon, train_kld

def validate(epoch):
    vae.eval()
    val_loss = 0
    val_recon = 0
    val_kld = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            data = batch['cell_image'].to(device)
            recon_batch, mu, logvar = vae(data)
            RECON, KLD = loss_function(recon_batch, data, mu, logvar)
            denominator = torch.prod(torch.tensor(data.shape)) if torch.prod(torch.tensor(data.shape)) > torch.prod(torch.tensor(mu.shape)) else torch.prod(torch.tensor(mu.shape))
            RECON = RECON.sum() / denominator
            KLD = KLD.sum() / denominator
            loss = RECON + KLD * args.beta
            val_loss += loss.item()
            val_recon += RECON.item()
            val_kld += KLD.item()

    val_loss /= len(val_dataloader)
    val_recon /= len(val_dataloader)
    val_kld /= len(val_dataloader)

    print(f"Validation Loss: {val_loss:.4f}, RECON: {val_recon:.4f}, KLD: {val_kld:.4f}")

    # Log to TensorBoard
    logger.add_scalar("val_loss", val_loss, epoch)
    logger.add_scalar("val_RECON", val_recon, epoch)
    logger.add_scalar("val_KLD", val_kld, epoch)

    return val_loss, val_recon, val_kld

# Main training loop
best_val_loss = float('inf')
for epoch in range(args.epochs):
    train_loss, train_recon, train_kld = train(epoch, log_interval=100)
    val_loss, val_recon, val_kld = validate(epoch)

    # Save checkpoint
    is_best = val_loss < best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, os.path.join(checkpoint_path, f"epoch_{epoch}.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_path, "best_model.pt"))

    # Save logs
    pd.DataFrame(training_log).to_csv(os.path.join(log_path, "training_log.csv"), index=False)
    pd.DataFrame(epoch_log).to_csv(os.path.join(log_path, "epoch_log.csv"), index=False)

# Flush the logger and close
logger.flush()
logger.close()

# If tensorboard was launched, terminate the process
if args.find_port:
    tensorboard_process.terminate()