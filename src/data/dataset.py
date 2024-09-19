import os
import numpy as np
import zarr
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml

class ZarrCellDataset(Dataset):
    def __init__(self, parent_dir, csv_file, split="train", channels=[0, 1, 2, 3], 
                 mask="masks", normalizations=None, interpolations=None, mean=None, std=None):
        self.parent_dir = Path(parent_dir)
        self.channels = channels
        self.mask = mask
        self.normalizations = normalizations
        self.interpolations = interpolations

        self.data_info = pd.read_csv(csv_file)
        self.data_info = self.data_info[self.data_info['split'] == split]
        self.grouped_data = self.data_info.groupby(['gene', 'barcode', 'stage'])
        self.zarr_data = self._load_all_zarr_data()

        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        gene = row['gene']
        barcode = row['barcode']
        stage = row['stage']
        cell_idx = row['cell_idx']

        # Get the zarr data for this gene, barcode, and stage
        zarr_group = self.zarr_data[(gene, barcode, stage)]

        # Load images and masks
        original_image = zarr_group['images'][cell_idx]
        original_image = original_image[self.channels]  # Select specified channels
        cell_mask = zarr_group['cells'][cell_idx]
        nuclei_mask = zarr_group['nuclei'][cell_idx]

        # Apply mask and normalization
        cell_image, nuclei_image = self._apply_mask_normalization(original_image, cell_mask, nuclei_mask)

        # Apply interpolations
        cell_image, nuclei_image = self._apply_interpolation(cell_image, nuclei_image)

        sample = {
            'gene': gene,
            'barcode': barcode,
            'stage': stage,
            'cell_idx': cell_idx,
            'split': row['split'],
            'original_image': original_image,
            'cell_mask': cell_mask,
            'nuclei_mask': nuclei_mask,
            'cell_image': cell_image,
            'nuclei_image': nuclei_image
        }

        return sample
    
    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._compute_mean()
        return self._mean
    
    @property
    def std(self):
        if self._std is None:
            self._std = self._compute_std()
        return self._std

    def _load_all_zarr_data(self):
        zarr_data = {}
        for (gene, barcode, stage), group in self.grouped_data:
            zarr_file = self.parent_dir / f"{gene}.zarr" / barcode / stage
            if not zarr_file.is_dir():
                raise ValueError(f"Zarr file not found: {zarr_file}")
            zarr_data[(gene, barcode, stage)] = zarr.open(zarr_file, mode='r')
        return zarr_data
    
    def _compute_mean(self):
        total_sum = np.zeros(len(self.channels))
        total_count = 0
        for batch in self:
            image = batch['original_image']
            total_sum += image.sum(axis=(1, 2))
            total_count += image.shape[1] * image.shape[2]
        mean = total_sum / total_count
        return mean
    
    def _compute_std(self):
        sum_squared_diff = np.zeros(len(self.channels))
        total_count = 0
        for batch in self:
            image = batch['original_image']
            sum_squared_diff += ((image - self.mean[:, None, None]) ** 2).sum(
                axis=(1, 2)
            )
            total_count += image.shape[1] * image.shape[2]

        variance = sum_squared_diff / total_count
        std = np.sqrt(variance)
        return std
      
    def _apply_mask_normalization(self, original_image, cell_mask, nuclei_mask):
        if self.mask == "masks":
            fill = self._mean[:, None, None] if self._mean is not None else 0
            cell_image = np.where(cell_mask, original_image, fill)
            nuclei_image = np.where(nuclei_mask, original_image, fill)
        elif self.mask == "min":
            fill = original_image.min(axis=(1, 2))[:, None, None]
            cell_image = np.where(cell_mask, original_image, fill)
            nuclei_image = np.where(nuclei_mask, original_image, fill)
        else:
            cell_image = original_image
            nuclei_image = original_image

        if self._mean is not None and self._std is not None:
            cell_image = (cell_image - self._mean[:, None, None]) / self._std[:, None, None]
            nuclei_image = (nuclei_image - self._mean[:, None, None]) / self._std[:, None, None]
        cell_image = torch.from_numpy(cell_image).float()
        nuclei_image = torch.from_numpy(nuclei_image).float()

        if self.normalizations:
            if isinstance(self.normalizations, list):
                for normalization in self.normalizations:
                    cell_image = normalization(cell_image)
                    nuclei_image = normalization(nuclei_image)
            else:
                cell_image = self.normalizations(cell_image)
                nuclei_image = self.normalizations(nuclei_image)

        return cell_image, nuclei_image

    def _apply_interpolation(self, cell_image, nuclei_image):
        if self.interpolations:
            if isinstance(self.interpolations, list):
                for interpolation in self.interpolations:
                    cell_image, nuclei_image = interpolation(cell_image, nuclei_image)
            else:
                cell_image, nuclei_image = self.interpolations(cell_image, nuclei_image)
        return cell_image, nuclei_image
    
    def save_info(self, path):
        info = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }
        
        # Determine the file extension
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(info, f, indent=2)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(info, f)
        else:
            raise ValueError("Unsupported file format. Use .json, .yaml, or .yml")



class ZarrCellDataset_specific(Dataset):
    def __init__(self, parent_dir, gene_name, barcode_name, channels=[0, 1, 2, 3], cell_cycle_stages="interphase", 
                 mask="masks", normalizations=None, interpolations=None, mean=None, std=None):
        self.parent_dir = parent_dir
        self.gene_name = gene_name
        self.barcode_name = barcode_name
        self.channels = channels
        self.cell_cycle_stages = cell_cycle_stages
        self.mask = mask
        self.normalizations = normalizations
        self.interpolations = interpolations
        self._mean = mean
        self._std = std

        self.zarr_data = self._load_zarr_data()
        self.original_images, self.cell_masks, self.nuclei_masks = self._load_images_and_masks()

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        original_image = self.original_images[idx]
        cell_mask = self.cell_masks[idx]
        nuclei_mask = self.nuclei_masks[idx]
        
        cell_image, nuclei_image = self._apply_mask_normalization(original_image, cell_mask, nuclei_mask)
        cell_image, nuclei_image = self._apply_interpolation(cell_image, nuclei_image)

        sample = {
            'gene': self.gene_name,
            'barcode': self.barcode_name,
            'stage': self.cell_cycle_stages,
            'original_image': original_image,
            'cell_mask': cell_mask,
            'nuclei_mask': nuclei_mask,
            'cell_image': cell_image,
            'nuclei_image': nuclei_image
        }
        return sample
    
    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._compute_mean()
        return self._mean
    
    @property
    def std(self):
        if self._std is None:
            self._std = self._compute_std()
        return self._std

    def _load_zarr_data(self):
        zarr_file_gene = os.path.join(self.parent_dir, f"{self.gene_name}.zarr")
        if not os.path.isdir(zarr_file_gene):
            raise ValueError(f"Gene {zarr_file_gene} does not exist")
        
        zarr_file_barcode = os.path.join(zarr_file_gene, self.barcode_name)
        if not os.path.isdir(zarr_file_barcode):
            raise ValueError(f"Barcode {zarr_file_barcode} does not exist")

        zarr_file_stage = os.path.join(zarr_file_barcode, self.cell_cycle_stages)
        if not os.path.isdir(zarr_file_stage):
            raise ValueError(f"Stage {zarr_file_stage} does not exist")

        self._read_zattrs(zarr_file_stage)  # You might want to do something with zattrs
        
        return zarr.open(zarr_file_gene, mode='r')

    def _load_images_and_masks(self):
        original_images = self.zarr_data[self.barcode_name][self.cell_cycle_stages]['images'][:, self.channels, :, :]
        cell_masks = self.zarr_data[self.barcode_name][self.cell_cycle_stages]['cells']
        nuclei_masks = self.zarr_data[self.barcode_name][self.cell_cycle_stages]['nuclei']

        if len(original_images) != len(cell_masks) or len(original_images) != len(nuclei_masks):
            raise ValueError("Number of images, cells, and nuclei are not the same")

        cell_masks = np.expand_dims(cell_masks, 1)
        nuclei_masks = np.expand_dims(nuclei_masks, 1)

        return original_images, cell_masks, nuclei_masks
    
    def _compute_mean(self):
        total_sum = np.zeros(len(self.channels)) # (1,4,250,250)
        total_count = 0
        for batch in self:
            image = batch['original_image'] # (1,4,250,250)
            total_sum += image.sum(axis=(1, 2))
            total_count += image.shape[1] * image.shape[2]
        mean = total_sum / total_count
        return mean
    
    def _compute_std(self):
        sum_squared_diff = np.zeros(len(self.channels))
        total_count = 0
        for batch in self:
            image = batch['original_image']
            sum_squared_diff += ((image - self.mean[:, None, None]) ** 2).sum(
                axis=(1, 2)
            )
            total_count += image.shape[1] * image.shape[2]

        variance = sum_squared_diff / total_count
        std = np.sqrt(variance)
        return std
    
    def _apply_mask_normalization(self, original_image, cell_mask, nuclei_mask):

        if self.mask == "masks":
            fill = self._mean[:, None, None] if self._mean is not None else 0
            cell_image = np.where(cell_mask, original_image, fill)
            nuclei_image = np.where(nuclei_mask, original_image, fill)
        elif self.mask == "min":
            fill = original_image.min(axis=(1, 2))[:, None, None]
            cell_image = np.where(cell_mask, original_image, fill)
            nuclei_image = np.where(nuclei_mask, original_image, fill)
        else:
            cell_image = original_image
            nuclei_image = original_image


        if self._mean is not None and self._std is not None:
            cell_image = (cell_image - self._mean[:, None, None]) / self._std[:, None, None]
            nuclei_image = (nuclei_image - self._mean[:, None, None]) / self._std[:, None, None]

        cell_image = torch.from_numpy(cell_image).float()
        nuclei_image = torch.from_numpy(nuclei_image).float()

        if self.normalizations:
            if isinstance(self.normalizations, list):
                for normalization in self.normalizations:
                    cell_image = normalization(cell_image)
                    nuclei_image = normalization(nuclei_image)
            else:
                cell_image = self.normalizations(cell_image)
                nuclei_image = self.normalizations(nuclei_image)

        return cell_image, nuclei_image

    def _apply_interpolation(self, cell_image, nuclei_image):
        if self.interpolations:
            if isinstance(self.interpolations, list):
                for interpolation in self.interpolations:
                    cell_image, nuclei_image = interpolation(cell_image, nuclei_image)
            else:
                cell_image, nuclei_image = self.interpolations(cell_image, nuclei_image)
        return cell_image, nuclei_image

    def _read_zattrs(self, path):
        zattrs = {}
        zattrs_path = os.path.join(path, ".zattrs")
        if os.path.exists(zattrs_path):
            with open(zattrs_path, "r") as f:
                zattrs = json.load(f)
        return zattrs