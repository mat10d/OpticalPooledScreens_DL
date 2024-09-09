# Imports
from models.dataset_static import ZarrCellDataset
from models.dataloader_static import collate_wrapper
from models.model_VAE_resnet18 import VAEResNet18
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import v2
import subprocess
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import yaml
from models.static_utils import read_config

# All settings
# Hyperparameters
beta = 1e-5
nc = 4
z_dim = 10
num_workers = 8
lr = 1e-4
batch_size = 16
num_epochs = 30
transform = "min"
crop_size = 96
channels = [0, 1, 2, 3]
# Basic values for logging 
parent_dir = '/mnt/efs/dlmbl/S-md/'
output_dir = '/mnt/efs/dlmbl/G-et/da_testing/'
output_path = output_dir + 'training_logs/'
model_name = f"static_resnet_linear_vae_da_benchmark_nonlinear_{beta}_{z_dim}_{lr}"
run_name= "da_testing"
find_port = True

# Define variables for the dataset read in
csv_file = '/mnt/efs/dlmbl/G-et/csv/dataset_split_benchmark_nontargeting.csv'
split = 'train'
crop_size = 96
normalizations = v2.Compose([v2.CenterCrop(crop_size)])
yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"

# Define the metadata keys
metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['cell_image']



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
    print(
        f"TensorBoard started at http://localhost:{port}. \n"
        "If you are using VSCode remote session, forward the port using the PORTS tab next to TERMINAL."
    )
    return process

# Launch tensorboard and click on the link to view the logs.
if find_port:
    tensorboard_process = launch_tensorboard(output_path)
logger = SummaryWriter(f"{output_path}/{model_name}")

# Create the dataset
dataset_mean, dataset_std = read_config(yaml_file_path)
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
vae = VAEResNet18(nc=nc, z_dim=z_dim) # , input_spatial_dim=[crop_size,crop_size])

vae = vae.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, KLD   

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
    beta=1e-3
    ):
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    model.train()
    log_losses = {
        "train_loss": 0, 
        "train_MSE": 0,
        "train_KLD": 0
    }
    train_loss = 0
    for batch_idx, batch in pbar:
        data = batch['cell_image'].to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)
        MSE, KLD  = loss_function(recon_batch, data, mu, logvar)
        loss = MSE + KLD * beta
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        log_losses["train_loss"] += loss.item()
        log_losses["train_MSE"] += MSE.item()
        log_losses["train_KLD"] += KLD.item()
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': log_losses["train_loss"] / log_interval })
            row = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'len_data': len(batch['cell_image']),
                'len_dataset': len(loader.dataset),
                'loss': log_losses["train_loss"] / log_interval,
                'MSE': log_losses["train_MSE"] / log_interval,
                'KLD': log_losses["train_KLD"] / log_interval 
            }
            training_log.append(row)
            log_losses = {
                "train_loss": 0, 
                "train_MSE": 0,
                "train_KLD": 0
            }

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_idx
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="train_MSE", scalar_value=MSE.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="train_KLD", scalar_value=KLD.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                input_image = data.to("cpu").detach()       
                predicted_image = recon_batch.to("cpu").detach()

                tb_logger.add_images(
                    tag="input_channel_0", img_tensor=input_image[:,0:1,...], global_step=step
                )
                tb_logger.add_images(
                    tag= "reconstruction_0", img_tensor=predicted_image[:,0:1,...], global_step=step
                )
                
                tb_logger.add_images(
                    tag="input_1", img_tensor=input_image[:,1:2,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_1", img_tensor=predicted_image[:,1:2,...], global_step=step
                )

                tb_logger.add_images(
                    tag="input_2", img_tensor=input_image[:,2:3,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_2", img_tensor=predicted_image[:,2:3,...], global_step=step
                )

                tb_logger.add_images(
                    tag="input_3", img_tensor=input_image[:,3:4,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_3", img_tensor=predicted_image[:,3:4,...], global_step=step
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
                'Average Loss': train_loss / len(dataloader)}
    epoch_log.append(epoch_raw)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader)))

# Training loop
folder_suffix = datetime.now().strftime("%Y%m%d_%H%M_") + run_name
checkpoint_path = output_path + "checkpoints/static/" + folder_suffix + "/"
log_path = output_path + "logs/static/"+ folder_suffix + "/"

# Create the directories
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
Path(log_path).mkdir(parents=True, exist_ok=True)

print(
    f"Saving checkpoints to {checkpoint_path} and logs to {log_path}",
    f"Model: {model_name}",
    f"Run: {run_name}",
    sep="\n",
)

for epoch in range(0, num_epochs):
    train(epoch, log_interval=100, log_image_interval=20, tb_logger=logger, beta=beta)
    filename_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_") + "epoch_"+str(epoch) + "_"
    training_logDF = pd.DataFrame(training_log)
    training_logDF.to_csv(log_path + filename_suffix+"training_log.csv", index=False)

    epoch_logDF = pd.DataFrame(epoch_log)
    epoch_logDF.to_csv(log_path + filename_suffix+"epoch_log.csv", index=False)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_per_epoch / len(dataloader)
    }
    torch.save(checkpoint, checkpoint_path + filename_suffix + "checkpoint.pth")