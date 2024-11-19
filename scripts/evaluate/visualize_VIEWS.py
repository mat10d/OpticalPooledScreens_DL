#!/usr/bin/env python
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.eval.vae_eval import VIEWSAnalyzer
from evaluate_VAE import generate_config

def main():
    # Setup paths and config
    base_dir = '20240925-185336_VAE_ResNet18_crop_size_96_nc_4_z_dim_30_lr_0.0001_beta_1_transform_min_loss_MSE_CCT2-nontargeting-diane'
    checkpoint_dir = f'/lab/barcheese01/aconcagua_results/checkpoints/{base_dir}'
    log_dir = f'/lab/barcheese01/aconcagua_results/logs/{base_dir}'
    epoch = 'epoch_45'
    
    config = generate_config(checkpoint_dir, log_dir, epoch)
    latent_dir = os.path.join(config['output_dir'], epoch)
    output_dir = os.path.join(config['output_dir'], epoch, 'views')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize VIEWS analyzer for train split
    latent_vectors_path = os.path.join(latent_dir, 'train_latent_vectors.csv')
    views = VIEWSAnalyzer(latent_vectors_path, config, split='train', n_top_dims = 25)
    
    # Analyze dimensions with modified parameters
    dimensions_to_analyze = range(25)
    for dim in dimensions_to_analyze:
        print(f"\nAnalyzing dimension {dim}...")
        
        selected_indices = views.find_constrained_samples(
            target_dim=dim,
            n_samples_per_percentile=3,  # Get 3 samples for each percentile
            percentiles=[1, 15, 50, 85, 99],  # Specific percentiles to sample at
            constraint_percentile=10
        )

        print(f"Selected indices: {selected_indices}")
       
        output_path = os.path.join(output_dir, f'views_dimension_{dim}.png')
        views.visualize_dimension(
            target_dim=dim,
            selected_indices=selected_indices,
            output_path=output_path
        )
        
        stats = views.analyze_dimension_distribution(dim)
        print(f"Statistics for dimension {dim}:")
        
if __name__ == '__main__':
    main()
