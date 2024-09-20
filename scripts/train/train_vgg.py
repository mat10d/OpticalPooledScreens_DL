import os
import sys
import argparse
import socket
import subprocess
from datetime import datetime

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as v2
import torchview
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import ZarrCellDataset
from src.data.dataloader import collate_wrapper
from src.data.utils import read_config
from src.models.vgg2d import Vgg2D

parser = argparse.ArgumentParser(description='VGG Classifier Training')

# Dataset and file paths
parser.add_argument('--parent_dir', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches', help='Parent directory of the dataset')
parser.add_argument('--csv_file', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.csv', help='CSV file with dataset information')
parser.add_argument('--yaml_file', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.yaml', help='YAML file with dataset info')
parser.add_argument('--output_dir', type=str, default='/lab/barcheese01/aconcagua_results', help='Output directory')
parser.add_argument('--dataset', type=str, default='CCT2-nontargeting', help='Dataset name')

# Model parameters
parser.add_argument('--model_name', type=str, default='VGG2D', help='Model name')

# Training parameters
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loading')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--label_type', type=str, default='gene', help='Type of label to use')

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
run_name = f"{args.model_name}_crop{args.crop_size}_lr{args.lr}_transform{args.transform}_{args.dataset}"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(args.output_dir, "tensorboard", f"{timestamp}_{run_name}")
log_path = os.path.join(args.output_dir, "logs", f"{timestamp}_{run_name}")
checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"{timestamp}_{run_name}")

for path in [log_dir, log_path, checkpoint_path]:
    os.makedirs(path, exist_ok=True)

# Launch tensorboard
if args.find_port:
    tensorboard_process = launch_tensorboard(os.path.dirname(log_dir))
logger = SummaryWriter(log_dir=log_dir)

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

# Get class names and number of classes
df = pd.read_csv(args.csv_file)
class_names = df[args.label_type].sort_values().unique().tolist()
num_classes = len(class_names)
print(f"Class names: {class_names}")

# Create the model
classifier = Vgg2D(
    input_size=(args.crop_size, args.crop_size),
    input_fmaps=len(args.channels),
    output_classes=num_classes,
)

torchview.draw_graph(
    classifier,
    train_dataset[0]['cell_image'].unsqueeze(0),
    roll=True,
    depth=3,  # adjust depth to zoom in.
    device="cpu",
    save_graph=True,
    filename=os.path.join("graphs", run_name)
)

classifier = classifier.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

# Define training function
training_log = []
epoch_log = []

def train(epoch, print_interval=10, log_interval=100):
    classifier.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(train_dataloader):
        data = batch['cell_image'].to(device)
        labels = torch.tensor([class_names.index(label) for label in batch[args.label_type]]).to(device)
        
        optimizer.zero_grad()
        outputs = classifier(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % print_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} "
                  f"({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}")
        
        if batch_idx % log_interval == 0:
            step = epoch * len(train_dataloader) + batch_idx
            
            # Log scalar values
            logger.add_scalar("train/loss", loss.item(), step)
            logger.add_scalar("train/accuracy", 100. * correct / total, step)
            
            # Log detailed metrics
            row = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'loss': loss.item(),
                'accuracy': 100. * correct / total
            }
            training_log.append(row)
            
            # # Log images and predictions
            # logger.add_images("train/input", data[:4].cpu(), step)
            # logger.add_text("train/predictions", str(predicted[:4].tolist()), step)
            # logger.add_text("train/ground_truth", str(labels[:4].tolist()), step)
            
    train_loss /= len(train_dataloader)
    train_accuracy = 100. * correct / total
    
    # Save epoch summary
    epoch_row = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy
    }
    epoch_log.append(epoch_row)

    print(f'====> Epoch: {epoch} Average loss: {train_loss:.4f} Accuracy: {train_accuracy:.4f}')
    
    return train_loss, train_accuracy

def validate(epoch):
    classifier.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            data = batch['cell_image'].to(device)
            labels = torch.tensor([class_names.index(label) for label in batch[args.label_type]]).to(device)
            
            outputs = classifier(data)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_dataloader)
    val_accuracy = 100. * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    
    # Log to TensorBoard
    logger.add_scalar("val/loss", val_loss, epoch)
    logger.add_scalar("val/accuracy", val_accuracy, epoch)
    
    return val_loss, val_accuracy

# Main training loop
best_val_loss = float('inf')
for epoch in range(args.epochs):
    train_loss, train_accuracy = train(epoch, log_interval=100)
    val_loss, val_accuracy = validate(epoch)
    
    # Save checkpoint
    is_best = val_loss < best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
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