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

from ConsecutivePatchDataset import ConsecutivePatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *



class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch, VELOCITY):
        # print(f'batch \n {batch.shape}')
        images = batch["image"]
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]
        velocity = batch["velocity"] if VELOCITY else 0

        res_targets = batch["res_targets"]
        fps_targets = batch["fps_targets"]
        # print(f'\n\n\n training step')
        
        res_out, fps_out = self(images, fps, bitrate, resolution, velocity)  # NaturalSceneClassification.forward

        # print(f'res_out {res_out.size()} \n {res_out}')
        # print(f'res_targets {res_targets.size()} \n {res_targets}')
        # print(f'fps_out {fps_out.size()} \n {fps_out}')
        total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
        # loss = F.mse_loss(out.squeeze(), labels.float()) # Calculate loss
        # print(f'loss_res {loss_res}')
        # print(f'loss_fps {loss_fps}')

        return total_loss
    
    def validation_step(self, batch):
        # print(f'\n\n\n validation step')
        images = batch["image"]
        # labels = batch["label"]
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]
        # velocity = batch["velocity"]

        # TODO: convert labels into res_targets, fps_targets 
        res_targets = batch["res_targets"]
        fps_targets = batch["fps_targets"]
        res_out, fps_out = self(images, fps, bitrate, resolution)  # NaturalSceneClassification.forward
        # print(f'training_step out {out.size()} \n {out.squeeze()}')
        # print(f'res_targets {res_targets}')
        # loss_fn_res = nn.CrossEntropyLoss()
        # loss_fn_fps = nn.CrossEntropyLoss()
    
        total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
        framerate_accuracy, resolution_accuracy, both_correct_accuracy = compute_accuracy(fps_out, res_out, fps_targets, res_targets)

        # Calculate accuracy, i.e.  proportion of the variance in the dependent variable that is predictable 
     
        # print(f'val_r2_score {val_r2_score}\n\n\n')
        return {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, 'both_acc': both_correct_accuracy} 
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # print(f'batch_losses \n {batch_losses}')
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_res_accs = [x['res_acc'] for x in outputs]
        batch_fps_accs = [x['fps_acc'] for x in outputs]
        batch_both_accs = [x['both_acc'] for x in outputs]
        # print(f'batch_accs \n {batch_accs}')

        epoch_res_acc = torch.stack(batch_res_accs).mean() # Combine accuracies
        epoch_fps_acc = torch.stack(batch_fps_accs).mean()
        epoch_both_acc = torch.stack(batch_both_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_res_acc': epoch_res_acc.item(), \
                'val_fps_acc': epoch_fps_acc.item(), 'val_both_acc': epoch_both_acc.item()}
    
    def epoch_end(self, epoch, result):
        # val_acc_R2
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_res_acc: {:.4f}, val_fps_acc: {:.4f}, val_both_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_res_acc'], result['val_fps_acc'], result['val_both_acc']))
        
