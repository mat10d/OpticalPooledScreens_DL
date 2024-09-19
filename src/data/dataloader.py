import torch
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

class CustomBatch:
    def __init__(self, data, metadata_keys, images_keys):
        self.metadata = defaultdict(list)
        self.images = defaultdict(list)
        
        for item in data:
            for key in images_keys:
                # convert to float and then to tensor
                self.images[key].append(torch.tensor(item[key], dtype=torch.float32))
            for key in metadata_keys:
                self.metadata[key].append(item[key])
        
        # Convert lists to tensors
        for key in self.images:
            self.images[key] = torch.stack(self.images[key], 0)
        
        # Convert metadata to tensors where possible
        for key in self.metadata:
            if all(isinstance(item, (int, float)) for item in self.metadata[key]):
                self.metadata[key] = torch.tensor(self.metadata[key])
            else:
                self.metadata[key] = tuple(self.metadata[key])
    
    def __getitem__(self, key):
        if key in self.images:
            return self.images[key]
        elif key in self.metadata:
            return self.metadata[key]
        else:
            raise KeyError(f"Key '{key}' not found in batch")
        
    def pin_memory(self):
        for key in self.images:
            self.images[key] = self.images[key].pin_memory()
        return self

    def to(self, device):
        for key in self.images:
            self.images[key] = self.images[key].to(device)
        return self

def collate_wrapper(metadata_keys, images_keys):
    def collate_fn(batch):
        return CustomBatch(batch, metadata_keys, images_keys)
    return collate_fn 