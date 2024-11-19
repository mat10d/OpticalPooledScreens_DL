#!/usr/bin/env python
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.eval.vae_eval import LatentSpaceVisualizer
from evaluate_VAE import generate_config

def main():
    # Specific model details
    base_dir = '20240925-185336_VAE_ResNet18_crop_size_96_nc_4_z_dim_30_lr_0.0001_beta_1_transform_min_loss_MSE_CCT2-nontargeting-diane'
    checkpoint_dir = f'/lab/barcheese01/aconcagua_results/checkpoints/{base_dir}'
    log_dir = f'/lab/barcheese01/aconcagua_results/logs/{base_dir}'
    epoch = 'epoch_45'
    
    # Generate config
    config = generate_config(checkpoint_dir, log_dir, epoch)
  
    # Construct the exact path where evaluation results are stored
    eval_output_dir = os.path.join(config['output_dir'], epoch)
    print(f"Using evaluation directory: {eval_output_dir}")
    
    # Initialize and run visualizer with the correct path
    visualizer = LatentSpaceVisualizer(eval_output_dir, config)
    visualizer.run(debug=True)

if __name__ == '__main__':
    main()