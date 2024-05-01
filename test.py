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


img_path = '000d77d4_43_50_720_4000_825943.png'
filename = os.path.basename(img_path)
        
parts = filename.split('_')     
fps = float(parts[2])
pixel = int(parts[3])  
bitrate = int(parts[4])  
velocity = int(parts[-1].split('.')[0]) / 10000  # Remove .png and convert to integer


# print(f'velocity {velocity}')
# print(f'pixel {pixel}')
# print(f'fps {fps}')
# print(f'bitrate {bitrate}')

res_out = torch.tensor([[0.2377, 0.1925, 0.2035, 0.1998, 0.1665],
        [0.2321, 0.2041, 0.1992, 0.2030, 0.1617],
        [0.2333, 0.2142, 0.1986, 0.1791, 0.1748],
        [0.2478, 0.2029, 0.2074, 0.1771, 0.1647],
        [0.2579, 0.1771, 0.2016, 0.2117, 0.1518],
        [0.2551, 0.1840, 0.2099, 0.1964, 0.1546],
        [0.2359, 0.2137, 0.1977, 0.1871, 0.1656],
        [0.2325, 0.2060, 0.2008, 0.1860, 0.1748],
        [0.2352, 0.2045, 0.1950, 0.2034, 0.1620],
        [0.2541, 0.1942, 0.2172, 0.1760, 0.1584],
        [0.2330, 0.2070, 0.1942, 0.2006, 0.1652],
        [0.2499, 0.1892, 0.2116, 0.1918, 0.1575],
        [0.2332, 0.2079, 0.1989, 0.1901, 0.1700],
        [0.2464, 0.1895, 0.2139, 0.1919, 0.1584],
        [0.2464, 0.1895, 0.2141, 0.1890, 0.1610],
        [0.2373, 0.2133, 0.1955, 0.1817, 0.1721],
        [0.2422, 0.2105, 0.2021, 0.1768, 0.1683],
        [0.2350, 0.2134, 0.2017, 0.1773, 0.1726],
        [0.2345, 0.2133, 0.2032, 0.1778, 0.1712],
        [0.2374, 0.2071, 0.1968, 0.1900, 0.1687],
        [0.2460, 0.1948, 0.2012, 0.2028, 0.1551],
        [0.2387, 0.2122, 0.2025, 0.1726, 0.1741],
        [0.2401, 0.1988, 0.1916, 0.2130, 0.1566],
        [0.2480, 0.1897, 0.2136, 0.1929, 0.1558],
        [0.2343, 0.2068, 0.1975, 0.1846, 0.1768],
        [0.2649, 0.2087, 0.2039, 0.1662, 0.1562],
        [0.2440, 0.1925, 0.2027, 0.1921, 0.1687],
        [0.2212, 0.1840, 0.2108, 0.2139, 0.1701],
        [0.2345, 0.2104, 0.1937, 0.1893, 0.1721],
        [0.2344, 0.2067, 0.1973, 0.1941, 0.1675],
        [0.2206, 0.1903, 0.2208, 0.1823, 0.1860],
        [0.2337, 0.2087, 0.2005, 0.1858, 0.1713]])
res_targets = torch.tensor([120, 110, 120, 110, 120, 120, 110, 120, 110, 110, 120, 110, 110, 110,
        110, 110, 110, 110, 120, 120, 110, 110, 110, 120, 110, 120, 110, 120,
        110, 120, 110, 120])


# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(f'input {input}')
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(f'target {target.size} {target}')

# output = loss(input, target)
# print(f'output {output}')


# print([i for i in range(30, 121, 10)])

x = [0.21875, 0.28515625, 0.2734375, 0.22265625, 0.234375, 0.31640625, 0.34375, 0.20703125, 0.28515625, 0.2890625, 0.25390625, 0.265625, 0.18359375, 0.28125, 0.21875, 0.2265625, 0.296875, 0.28125, 0.26171875, 0.265625, 0.2265625, 0.19140625, 0.32421875]