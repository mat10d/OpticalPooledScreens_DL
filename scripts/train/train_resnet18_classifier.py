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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import torchview
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import ZarrCellDataset
from src.data.dataloader import collate_wrapper
from src.data.utils import read_config
from src.models.VAE_resnet18 import ResNet18Classifier

parser = argparse.ArgumentParser(description='VGG Classifier Training')

# Dataset and file paths
parser.add_argument('--parent_dir', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches', help='Parent directory of the dataset')
parser.add_argument('--csv_file', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.csv', help='CSV file with dataset information')
parser.add_argument('--yaml_file', type=str, default='/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.yaml', help='YAML file with dataset info')
parser.add_argument('--output_dir', type=str, default='/lab/barcheese01/aconcagua_results', help='Output directory')
parser.add_argument('--dataset', type=str, default='CCT2-nontargeting', help='Dataset name')

# Model parameters
parser.add_argument('--model_name', type=str, default='Classifier_ResNet18', help='Model name')

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
parser.add_argument('--balance_classes', type=bool, default=True, help='Whether to balance classes')

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
    print(f"TensorBoard started at http://thyroxine.wi.mit.edu:{port}.")
    print("If using VSCode remote session, forward the port using the PORTS tab next to TERMINAL.")
    return process

# Setup paths and logging
run_name = f"{args.model_name}_crop_{args.crop_size}_lr_{args.lr}_transform_{args.transform}_{args.dataset}"
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
if args.balance_classes:
    df = pd.read_csv(args.csv_file)
    df = df[df['split'] == 'train']
    all_labels = df[args.label_type].tolist()
    weights = [1 / all_labels.count(label) for label in all_labels]
    print(f"Weighting classes: {np.unique(weights)}")
    balanced_sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        sampler=balanced_sampler,
        collate_fn=collate_wrapper(args.metadata_keys, args.images_keys),
        drop_last=True
    )
else:
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
classifier = ResNet18Classifier(
    num_classes=num_classes,
    nc=len(args.channels)
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

# Define metrics
def calculate_balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def calculate_auc(y_true, y_pred_proba, classes):
    y_pred_proba = np.array(y_pred_proba)  # Convert list to numpy array
    if len(classes) == 2:
        return roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
        return roc_auc_score(y_true_binarized, y_pred_proba, average='macro', multi_class='ovr')

# Define training function
training_log = []
def train(epoch, print_interval=10, log_interval=100):
    classifier.train()
    train_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
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

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
        
        if batch_idx % print_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} "
                  f"({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}")
        
        if batch_idx % log_interval == 0:
            step = epoch * len(train_dataloader) + batch_idx
            
            # Log scalar values
            logger.add_scalar("train_classify/loss", loss.item(), step)
            logger.add_scalar("train_classify/accuracy", correct / total, step)
            
            # Log detailed metrics
            row = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'loss': loss.item(),
                'accuracy': correct / total
            }
            training_log.append(row)
            
            
    train_loss /= len(train_dataloader)
    train_accuracy = correct / total
    train_balanced_accuracy = calculate_balanced_accuracy(all_labels, all_predictions)
    train_auc = calculate_auc(np.array(all_labels), np.array(all_probabilities), class_names)

    logger.add_scalar("train_classify/balanced_accuracy", train_balanced_accuracy, epoch)
    logger.add_scalar("train_classify/auc", train_auc, epoch)

    print(f'====> Epoch: {epoch} Train loss: {train_loss:.4f} Accuracy: {train_accuracy:.2f} Balanced Accuracy: {train_balanced_accuracy:.2f} AUC: {train_auc:.2f}')
    
    return train_loss, train_accuracy, train_balanced_accuracy, train_auc

def validate(epoch):
    classifier.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
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

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(F.softmax(outputs, dim=1).cpu().numpy())

    val_loss /= len(val_dataloader)
    val_accuracy = correct / total
    val_balanced_accuracy = calculate_balanced_accuracy(all_labels, all_predictions)
    val_auc = calculate_auc(np.array(all_labels), np.array(all_probabilities), class_names)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f} Balanced Accuracy: {val_balanced_accuracy:.4f} AUC: {val_auc:.2f}")
    
    # Log to TensorBoard
    logger.add_scalar("val/loss", val_loss, epoch)
    logger.add_scalar("val/accuracy", val_accuracy, epoch)
    logger.add_scalar("val/balanced_accuracy", val_balanced_accuracy, epoch)
    logger.add_scalar("val/auc", val_auc, epoch)

    return val_loss, val_accuracy, val_balanced_accuracy, val_auc

epoch_log = []
# Main training loop
for epoch in range(args.epochs):
    train_loss, train_accuracy, train_balanced_accuracy, train_auc = train(epoch, log_interval=100)
    val_loss, val_accuracy, val_balanced_accuracy, val_auc = validate(epoch)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'train_balanced_accuracy': train_balanced_accuracy,
        'train_auc': train_auc,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_balanced_accuracy': val_balanced_accuracy,
        'val_auc': val_auc
    }
    torch.save(checkpoint, os.path.join(checkpoint_path, f"epoch_{epoch}.pt"))
    
    # Update epoch_log with both training and validation metrics
    epoch_row = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'train_balanced_accuracy': train_balanced_accuracy,
        'train_auc': train_auc,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_balanced_accuracy': val_balanced_accuracy,
        'val_auc': val_auc
    }
    epoch_log.append(epoch_row)

    # Save logs
    pd.DataFrame(training_log).to_csv(os.path.join(log_path, "training_log.csv"), index=False)
    pd.DataFrame(epoch_log).to_csv(os.path.join(log_path, "epoch_log.csv"), index=False)

# Flush the logger and close
logger.flush()
logger.close()

# If tensorboard was launched, terminate the process
if args.find_port:
    tensorboard_process.terminate()