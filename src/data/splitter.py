import os
import numpy as np
import zarr
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
from joblib import Parallel, delayed
import argparse

class DatasetTable:
    def __init__(self, parent_dir, output_dir, num_workers=-1):
        self.parent_dir = Path(parent_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers

    def generate_cells_from_gene(self, gene_path):
        gene_path = Path(gene_path)
        gene_name = gene_path.stem
        cell_data = []
        
        def filter_dirs(path):
            return [item for item in path.iterdir() if item.is_dir() and item.name not in ['.zarr', '.DS_Store']]
        
        barcodes = filter_dirs(gene_path)

        if not barcodes:
            print(f"Warning: No barcodes found in {gene_path}. Skipping this gene.")
            return cell_data
        
        for barcode in barcodes:
            barcode_name = barcode.name 
            stages = filter_dirs(barcode)
            
            if not stages:
                print(f"Warning: No stages found in {barcode}. Skipping this barcode.")
                continue
            
            for stage in stages:
                stage_name = stage.name
                try:
                    cells_zarr = zarr.open(stage / "images")
                    num_cells = cells_zarr.shape[0]
                    
                    if num_cells == 0:
                        print(f"Warning: No cells found in {stage}. Skipping this stage.")
                        continue
                    
                    for cell_idx in range(num_cells):
                        cell_data.append([gene_name, barcode_name, stage_name, cell_idx])
                
                except Exception as e:
                    print(f"Error processing {stage}: {str(e)}. Skipping this stage.")
        
        return cell_data

    def generate_table(self):
        self.output_dir.mkdir(exist_ok=True)
        
        genes = list(self.parent_dir.glob("*.zarr"))
        genes = [gene for gene in genes if any(gene.iterdir())]
        
        print(f"Processing {len(genes)} genes...")

        results = Parallel(n_jobs=self.num_workers, verbose=1)(
            delayed(self.generate_cells_from_gene)(gene) for gene in genes
        )
        
        print("Combining results...")
        all_cell_data = [item for sublist in results for item in sublist]

        df = pd.DataFrame(all_cell_data, columns=["gene", "barcode", "stage", "cell_idx"])
        output_file = self.output_dir / f"dataset_{len(genes)}.csv"
        df.to_csv(output_file, index=False)
        print(f"Dataset CSV saved to {output_file}")

class DatasetSplitter:
    def __init__(self, parent_file, output_file, gene_list=None, stage_list=None, sample_size=None, balance_classes=False, train_ratio=0.7, val_ratio=0.15):
        self.parent_file = Path(parent_file)
        self.output_file = Path(output_file)
        self.gene_list = gene_list
        self.stage_list = stage_list
        self.sample_size = sample_size
        self.balance_classes = balance_classes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split_data(self, group):
        num_cells = len(group)
        indices = torch.randperm(num_cells).tolist()  # Convert to list
        
        train_size = int(num_cells * self.train_ratio)
        val_size = int(num_cells * self.val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        group.loc[group.index[train_indices], 'split'] = 'train'
        group.loc[group.index[val_indices], 'split'] = 'val'
        group.loc[group.index[test_indices], 'split'] = 'test'
        
        return group
    def generate_split(self):
        df = pd.read_csv(self.parent_file)

        if self.gene_list:
            df = df[df['gene'].isin(self.gene_list)]

        if self.stage_list:
            df = df[df['stage'].isin(self.stage_list)]

        if self.sample_size:
            df = df.groupby('gene').apply(lambda x: x.sample(n=min(self.sample_size, len(x)), random_state=42)).reset_index(drop=True)

        if self.balance_classes:
            min_class_size = df['gene'].value_counts().min()
            df = df.groupby('gene').apply(lambda x: x.sample(n=min_class_size, random_state=42)).reset_index(drop=True)

        df = df.groupby('gene').apply(self.split_data).reset_index(drop=True)

        df.to_csv(self.output_file, index=False)

        print(f"Dataset split CSV saved to {self.output_file}")
        print(df['gene'].value_counts())
        print(df['split'].value_counts(normalize=True))