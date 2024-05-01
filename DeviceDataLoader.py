import os 
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import Dataset


import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from utils import *

class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device """
    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self): # called once for every batch
        """ Yield a batch of data after moving it to device"""
        # b is a dictionary with all infos of 32 images
        for b in self.dl: # a list of 3 dictionary or 1 dictionary with 5 entries
            # print(f'b {b}')
            batch = {k: to_device(v, self.device) for k, v in b.items()} # same as b, but just in devices
            # print(f'b len {len(b)}')
            # print(f'batch {batch}')
            # for data in b:
            #     batch = {k: to_device(v, self.device) for k, v in data.items()}
            yield batch
            
    def __len__(self):
        """ Number of batches """
        return len(self.dl)
