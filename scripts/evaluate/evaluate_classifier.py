import argparse
import re
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.eval.classifier_eval import Classifier_Evaluator

def get_checkpoint_log_dirs(parent_dir='/lab/barcheese01/aconcagua_results/', checkpoint_dir=None, checkpoint_name=None):
    checkpoint_base = os.path.join(parent_dir, 'checkpoints')
    log_base = os.path.join(parent_dir, 'logs')
    
    checkpoint_dirs = os.listdir(checkpoint_base)
    if checkpoint_dir:
        checkpoint_dirs = [d for d in checkpoint_dirs if checkpoint_dir in d]
    
    valid_checkpoints = []
    valid_logs = []
    
    for d in checkpoint_dirs:
        checkpoint_path = os.path.join(checkpoint_base, d)
        log_path = os.path.join(log_base, d)
        
        if checkpoint_name:
            if os.path.exists(os.path.join(checkpoint_path, checkpoint_name + '.pt')):
                valid_checkpoints.append(checkpoint_path)
                valid_logs.append(log_path)
        else:
            if os.path.exists(os.path.join(checkpoint_path, 'best_model.pt')):
                valid_checkpoints.append(checkpoint_path)
                valid_logs.append(log_path)
    
    print(f"Number of valid checkpoints: {len(valid_checkpoints)}")

    return valid_checkpoints, valid_logs

def parse_checkpoint_dir(checkpoint_dir):
    filename = os.path.basename(checkpoint_dir)
    print(f"Processing: {filename}")
    
    result = {}
    model_match = re.search(r'_(Classifier_(?:ResNet18|VGG2D))_', filename)
    if model_match:
        result['model_name'] = model_match.group(1)
    
    params = ['crop', 'lr', 'transform']
    for param in params:
        match = re.search(rf'{param}_([^_]+)', filename)
        if match:
            value = match.group(1)
            result[param] = int(value) if value.isdigit() else float(value) if value.replace('.', '').isdigit() else value
    
    dataset_match = re.search(r'transform_[^_]+_(.+)$', filename)
    if dataset_match:
        result['dataset'] = dataset_match.group(1)
    else:
        result['dataset'] = 'unknown'

    print(f"Dataset: {result['dataset']}")

    return result

def generate_config(checkpoint_dir, log_dir, checkpoint_name):
    config = parse_checkpoint_dir(checkpoint_dir)
    config['model_name'] = config['model_name'].replace('Classifier_', '')
    
    csv_file = f"/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_{config['dataset']}.csv"
    print(f"CSV file: {csv_file}")

    config.update({
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
        'checkpoint_epoch': checkpoint_name,
        'parent_dir': '/lab/barcheese01/aconcagua_results/primary_screen_patches',
        'csv_file': csv_file,
        'yaml_file': '/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting.yaml',
        'batch_size': 16,
        'num_workers': 12,
        'label_type': 'gene',
        'crop_size': config['crop'],
        'transform': config['transform'],
        'metadata_keys': ['gene', 'barcode', 'stage', 'cell_idx'],
        'images_keys': ['cell_image'],
        'channels': [0, 1, 2, 3],
        'output_dir': os.path.join('/lab/barcheese01/aconcagua_results/classify', os.path.basename(checkpoint_dir)),
    })

    df = pd.read_csv(config['csv_file'])
    config['class_names'] = df[config['label_type']].sort_values().unique().tolist()
    print(f"Number of classes: {len(config['class_names'])}")
    
    return config

def run_evaluator(checkpoint_dir, log_dir, checkpoint_name):
    config = generate_config(checkpoint_dir, log_dir, checkpoint_name)
    evaluator = Classifier_Evaluator(config)
    results = evaluator.evaluate()
    print(f"Evaluation completed for {checkpoint_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a classifier checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='The specific checkpoint directory to evaluate')
    parser.add_argument('--epoch', type=str, required=True, help='The specific epoch to evaluate (e.g., "epoch_12" or "best_model")')
    args = parser.parse_args()

    checkpoint_base = '/lab/barcheese01/aconcagua_results/checkpoints'
    log_base = '/lab/barcheese01/aconcagua_results/logs'

    full_checkpoint_dir = os.path.join(checkpoint_base, args.checkpoint_dir)
    full_log_dir = os.path.join(log_base, args.checkpoint_dir)

    if not os.path.exists(full_checkpoint_dir):
        print(f"Error: Checkpoint directory {full_checkpoint_dir} does not exist")
        sys.exit(1)

    if not os.path.exists(full_log_dir):
        print(f"Error: Log directory {full_log_dir} does not exist")
        sys.exit(1)

    try:
        results = run_evaluator(full_checkpoint_dir, full_log_dir, args.epoch)
        print(f"Successfully evaluated {args.checkpoint_dir}")
    except Exception as e:
        print(f"Error evaluating {args.checkpoint_dir}: {str(e)}")
        sys.exit(1)