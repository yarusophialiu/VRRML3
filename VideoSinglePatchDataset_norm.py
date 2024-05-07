
import os 
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image



fps_map = {30: 0, 40: 1, 50: 2, 60: 3, 70: 4, 80: 5, 90: 6, 100: 7, 110: 8, 120: 9}
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}


class CustomTransform:
    def __init__(self, output_size, TYPE):
        # print(f'num_patches {num_patches}')
        self.output_size = output_size
        self.num_patches = 2
        self.TYPE = TYPE
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        w, h = image.size
        left_half = image.crop((0, 0, h, h))
        right_half = image.crop((64, 0, w, h))

        # left_half.show()
        # right_half.show()
        # print(f'self.TYPE {self.TYPE}')
        # print(f'patches {patches.size()} \n {patches}')
        if self.TYPE == 'decoded': 
            return left_half 
        if self.TYPE == 'reference':
            return right_half
    

# dataset: handling batching, shuffling, and iterations over the dataset during training or inference
class VideoSinglePatchDataset(Dataset):
    def __init__(self, directory, TYPE, min_bitrate, max_bitrate, patch_size=((64, 64)), VELOCITY=False):
        self.root_directory = directory
        self.patch_size = patch_size
        self.velocity = VELOCITY
        self.samples = []  # To store tuples of (image path, label)
        labels = os.listdir(directory)

        self.fps_targets = [int(label.split('x')[0]) for label in labels]
        self.res_targets = [int(label.split('x')[1]) for label in labels]

        self.min_fps = 30
        self.max_fps = 120
        self.min_res = 360
        self.max_res = 1080

        # self.min_bitrate = min_bitrate
        # self.max_bitrate = max_bitrate

        # print(f'TYPE {TYPE}')
        # print(f'self.min_bitrate, self.max_bitrate {self.min_bitrate, self.max_bitrate}')

        self.transform = transforms.Compose([
                    CustomTransform(patch_size, TYPE) ,
                    transforms.Resize((64, 64)),  # Resize images to 64x64
                    transforms.ToTensor(),  # Convert images to PyTorch tensors
                ])
                       

        for label in labels: 
                label_dir = os.path.join(directory, str(label))
                # print(f'label_dir {label_dir}')
                for root, _, filenames in os.walk(label_dir):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        self.samples.append((file_path, label))   
        # print(f'self.samples {self.samples}\n')

        
    def __len__(self):
        return len(self.samples)
    
    # def normalize_fps(self, sample, min_vals, max_vals):
    #     # print(f'val, min_vals, max_vals {sample, min_vals, max_vals}')
    #     fps = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    #     fps_mean = np.mean(fps)
    #     fps_std_dev = np.std(fps)
    #     sample = (sample - self.min_fps) / (max_vals - self.min_fps)
    #     return round(sample, 3)

    def normalize(self, sample, data):
        # print(f'val, min_vals, max_vals {sample, min_vals, max_vals}')
        # fps = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        # print(f'sample {sample}')
        mean = np.mean(data)
        std_dev = np.std(data)
        sample = (sample - mean) / std_dev
        # print(f'mean std_dev {mean, std_dev}')
        return round(sample, 3)
    
    # load individual data sample, apply transformations, 
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # print(f'img_path {img_path}')

        fps_targets = int(label.split('x')[1])
        res_targets = int(label.split('x')[0])

        filename = os.path.basename(img_path) # 00a0f6e8_50_864_2000.png or 00cb4b2e_40_864_2000_776102.png
        parts = filename.split('_')     
        fps = float(parts[1])
        pixel = int(parts[2])  
        velocity = 0
        if not self.velocity:
            bitrate = int(parts[-1].split('.')[0])  # Remove .png and convert to integer
        else:
            bitrate = int(parts[3])  
            velocity = int(parts[-1].split('.')[0]) / 10000  # Remove .png and convert to integer

        fps_data = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        res_data = np.array([360, 480, 720, 864, 1080])
        bitrate_data = np.array([2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500])
        fps = self.normalize(fps, fps_data)
        # print(f'fps {fps}\n')
        pixel = self.normalize(pixel, res_data)
        # print(f'pixel {pixel}\n')
        bitrate = self.normalize(bitrate, bitrate_data)
        # print(f'bitrate {bitrate}\n')



        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
            # print(f'image.size {image.size()}')
            # image.show()
        sample = {"image": image, "fps": fps, "bitrate": bitrate, "resolution": pixel, \
                  "fps_targets": fps_map[fps_targets], "res_targets": res_map[res_targets]}
        
        # print(f'self.velocity {self.velocity}')
        if self.velocity:
            sample['velocity'] = velocity
            # print(f'velocity {velocity}')
            return sample
        else:
            return sample
