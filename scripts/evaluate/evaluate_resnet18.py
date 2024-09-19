import re
import os
import re
from eval.evaluate_static import ModelEvaluator

def get_checkpoint_dirs():
    parent_dir = '/mnt/efs/dlmbl/G-et/checkpoints/static/Matteo/'
    checkpoint_dirs = os.listdir(parent_dir)
    checkpoint_dirs = [os.path.join(parent_dir, d) for d in checkpoint_dirs]
    checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]
    
    def get_timestamp(checkpoint_dir):
        filename = checkpoint_dir.split('/')[-1]
        match = re.search(r'(\d{8}_\d{4})', filename)
        if match:
            return match.group(1)
        return ''
    
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: get_timestamp(x))
    checkpoint_dirs = [d for d in checkpoint_dirs if get_timestamp(d) > '20240903_2100']
    print("number of checkpoints:", len(checkpoint_dirs))

    return checkpoint_dirs

def parse_checkpoint_dir(checkpoint_dir):
    filename = checkpoint_dir.split('/')[-1]
    print(filename)
    params = ['model', 'crop_size', 'nc', 'z_dim', 'lr', 'beta', 'transform', 'loss']
    result = {}
    model_match = re.search(r'_(VAE_ResNet18)_', filename)
    if model_match:
        result['model'] = model_match.group(1)
    
    for param in params:
        if param == 'model':
            continue
        match = re.search(rf'{param}_([^_]+)', filename)
        if match:
            value = match.group(1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            result[param] = value
    
    if 'benchmark' in filename:
        result['csv_file'] = 'dataset_split_benchmark.csv'
    
    return result

def generate_config(checkpoint_dir):
    config = parse_checkpoint_dir(checkpoint_dir)
    
    # Add invariant parameters
    config.update({
        'checkpoint_dir': checkpoint_dir,
        'parent_dir': '/mnt/efs/dlmbl/S-md/',
        'channels': [0, 1, 2, 3],
        'yaml_file_path': '/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml',
        'output_dir': os.path.join('/home/S-md/embed_time/scripts/latent', checkpoint_dir.split('/')[-1]),
        'sampling_number': 3,
        'csv_file': '/mnt/efs/dlmbl/G-et/csv/' + config['csv_file'],
        'batch_size': 16,
        'num_workers': 8,
        'metadata_keys': ['gene', 'barcode', 'stage', 'cell_idx'],
        'images_keys': ['cell_image']
    })
    
    return config

def run_evaluator(checkpoint_dir):
    config = generate_config(checkpoint_dir)
    return ModelEvaluator(config)

# Example usage
if __name__ == "__main__":
    # checkpoint_dir = '/mnt/efs/dlmbl/G-et/checkpoints/static/Matteo/20240903_2130_VAE_ResNet18_crop_size_64_nc_4_z_dim_30_lr_0.0001_beta_1e-05_transform_min_loss_L1_benchmark'
    checkpoint_dirs = get_checkpoint_dirs()
    for checkpoint_dir in checkpoint_dirs:
        run_evaluator(checkpoint_dir)