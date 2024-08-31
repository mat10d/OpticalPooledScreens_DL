import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Transforms
data_transform_train = transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Bring the dataset
dataset = torchvision.datasets.ImageFolder(root='/nfs/research/uhlmann/afoix/datasets/image_datasets/BBBC010_v1_foreground_eachworm/', transform=data_transform_train)

# Split datatset
train, val, test = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

# Create data datatloader
trainLoader = torch.utils.data.DataLoader(train, batch_size=batch_size, 
                                           num_workers=num_workers, drop_last=True, shuffle=True)
valLoader = torch.utils.data.DataLoader(val, batch_size=batch_size, 
                                          num_workers=num_workers, drop_last=True)
testLoader = torch.utils.data.DataLoader(test, batch_size=batch_size, 
                                          num_workers=num_workers, drop_last=True)

