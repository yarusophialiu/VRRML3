import os 
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import Dataset


import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

from VideoSinglePatchDataset import VideoSinglePatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
from ImageClassificationBase import *



class DecRefClassification(ImageClassificationBase):
    def __init__(self, num_framerates, num_resolutions, VELOCITY=False):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(), 
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),  
            nn.ReLU(), 
            nn.MaxPool2d(2,2), 
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),  
            nn.MaxPool2d(2,2), 
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2,2), 
            
            nn.Flatten(),
            nn.Linear(16384,1024), # output vector of size 1024 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32) # embedding of size 32
        )

        self.velocity = VELOCITY

        
        num_extra_features = 3 if self.velocity else 2
        self.fc_network = nn.Sequential(
            nn.Linear(32+num_extra_features, 16),  # fps, bitrate, velocity
            # nn.Linear(32 + 2, 16),  # Adjust input features to match your extended vector size
            nn.ReLU(),
            # nn.Linear(16, 8),
            # nn.ReLU(),
            # nn.Linear(8, 1)  # Adjust the output size based on your specific task
        )


        # Branch for resolution prediction
        self.fc_res = nn.Linear(16, num_resolutions)
        # Branch for framerate prediction
        self.fc_fps = nn.Linear(16, num_framerates)

    
    def forward(self, images, fps, bitrate, resolution, velocity=0):
        # print(f'image {images.size()} ')
        
        features = self.network(images)    
        # print(f'========= forward =========')
        # print(f'features \n {features[0]}')
        # print(f'resolution {resolution.size()} \n {resolution}')
        # print(f'fps {fps}')
        if self.velocity:
            fps_resolution_bitrate = torch.stack([fps, bitrate, velocity], dim=1).float()  # Example way to combine fps and bitrate
        else:
            fps_resolution_bitrate = torch.stack([fps, bitrate], dim=1).float()
        # print(f'fps_resolution_bitrate {fps_resolution_bitrate}')

        combined = torch.cat((features, fps_resolution_bitrate), dim=1)
        # print(f'combined {combined}')
        combined = minMaxNormalizer(combined)
        # print(f'combined {combined.size()}\n')               

        x = self.fc_network(combined)

        res_out = F.softmax(self.fc_res(x), dim=1)  # Softmax for categorical output
        fps_out = F.softmax(self.fc_fps(x), dim=1)  # Softmax for categorical output
        # print(f'res_out {res_out.squeeze(1)} \n\n\n')

        return res_out, fps_out

    