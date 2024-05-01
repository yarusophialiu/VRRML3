
import os 
import torch
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image



fps_map = {30: 0, 40: 1, 50: 2, 60: 3, 70: 4, 80: 5, 90: 6, 100: 7, 110: 8, 120: 9}
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}

# dataset: handling batching, shuffling, and iterations over the dataset during training or inference
class VideoPatchDataset(Dataset):
    def __init__(self, directory, patch_size=((64, 64)), num_patches=2, transform=None, STACK=False):
        self.root_directory = directory
        self.patch_size = patch_size
        self.samples = []  # To store tuples of (image path, label)
        labels = os.listdir(directory)
        # self.labels = [float(label) if float(label) % 1 != 0 else int(label) for label in self.labels]
        self.fps_targets = [int(label.split('x')[0]) for label in labels]
        self.res_targets = [int(label.split('x')[1]) for label in labels]
        print(f'self.fps_targets {self.fps_targets}')
        # print(f'self.res_targets {self.res_targets}')
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor(),])

        for label in labels: # label must be floats not integers
                label_dir = os.path.join(directory, str(label))
                # print(f'label_dir {label_dir}')
                for root, _, filenames in os.walk(label_dir):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        # np.concate()
                        # torch.cat((file_path, int(label.split('x')[0])))   
                        self.samples.append((file_path, label))   
        # print(f'self.samples {self.samples}\n')

        
    def __len__(self):
        return len(self.samples)
    
    # load individual data sample, apply transformations, 
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        fps_targets = int(label.split('x')[1])
        res_targets = int(label.split('x')[0])
        # print(f'img_path {img_path}')

        filename = os.path.basename(img_path)
        
        parts = filename.split('_')     
        fps = float(parts[2])
        pixel = int(parts[3])  
        bitrate = int(parts[4])  
        velocity = int(parts[-1].split('.')[0]) / 10000  # Remove .png and convert to integer

        # Load the image
        image = Image.open(img_path)
        # image.show()
        # print(f'image.size {image.size}')
        
        if self.transform:
            image = self.transform(image)
            # print(f'image.size {image.size()}')
            # print(f'image \n {image}')

            # image.show()

        sample = {"image": image, "fps": fps, "bitrate": bitrate, "resolution": pixel, 'velocity': velocity, \
                  "fps_targets": fps_map[fps_targets], "res_targets": res_map[res_targets]}
        return sample
