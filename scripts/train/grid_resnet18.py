import itertools
from tqdm import tqdm
import subprocess
import os
from datetime import datetime

# Define the parameter grid
param_grid = {
    'z_dim': [30, 10],
    'loss_type': ['L1', 'MSE', 'SSIM'],
    'crop_size': [64, 96],
    'beta': [1e-5, 1e-6],
    'transform': ['min', 'mask']
}

# Generate all combinations of parameters
param_combinations = list(itertools.product(*param_grid.values()))

# Main loop for grid search
for params in tqdm(param_combinations, desc="Grid Search Progress"):
    z_dim, loss_type, crop_size, beta, transform = params
        
    # Create command to run the main script with current parameters
    command = [
        "python", "training_loop_resnet18_md_grid.py",
        "--z_dim", str(z_dim),
        "--loss_type", loss_type,
        "--crop_size", str(crop_size),
        "--beta", str(beta),
        "--transform", transform,
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred with parameters: {params}")
        print(f"Error details: {e}")

print("Grid search completed!")