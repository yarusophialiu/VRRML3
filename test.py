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



# print(f'velocity {velocity}')
# print(f'pixel {pixel}')
# print(f'fps {fps}')
# print(f'bitrate {bitrate}')


# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(f'input {input}')
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(f'target {target.size} {target}')

# output = loss(input, target)
# print(f'output {output}')


fps_out = torch.tensor([[2.0797e-25, 8.9059e-25, 5.2397e-25, 9.2256e-25, 9.9996e-01, 3.7245e-05],
                        [2.5361e-25, 9.5832e-25, 6.7537e-25, 1.0615e-24, 9.9996e-01, 3.8012e-05],
                        [2.2934e-25, 9.2507e-25, 5.9047e-25, 9.9108e-25, 9.9996e-01, 3.7417e-05]])

fps_targets = [i for i in range(30, 81, 10)]
fps_softmax = torch.argmax(fps_out, dim=1)


fps_values = [fps_targets[idx] for idx in fps_softmax]
fps_values = torch.tensor(fps_values)

print(torch.unique(fps_values))

# print(f'{fps_targets}')
# print(f'{fps_softmax}')
# print(f'{fps_targets[fps_softmax]}')
