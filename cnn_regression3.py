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

from VideoPatchDataset import VideoPatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
from test import *

# regressin, learn the curves
# https://docs.google.com/presentation/d/16yqaaq5zDZ5-S4394VLBUfxpNjM7nlpssqcShFkklec/edit#slide=id.g2c751bc0d9c_0_18

BASE = 'C:/Users/15142/Desktop/data/VRR-classification'
data_dir = f"{BASE}/seg_train/seg_train/"
test_data_dir = f"{BASE}/seg_test/seg_test"
MLDIR = f'C:/Users/15142/Desktop/VRR/VRRML'


def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()

def show_batch(dl):
    """Plot images grid of single batch"""
    for batch in dl: # dl calls __getitem__
        images = batch["image"]
        print(f'images {images.dtype}')
        labels = batch["label"]
        print(f'Labels: {labels}')
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        # print(f'batch \n {batch.shape}')
        images = batch["image"]
        labels = batch["label"] # int64
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]
        out = self(images, fps, bitrate, resolution)  # NaturalSceneClassification.forward
        # loss = nn.MSELoss(out, labels) # Calculate loss
        # print(f'training_step out {out.size()} \n {out.squeeze()}')
        # print(f'labels out {labels}')
        loss = F.mse_loss(out.squeeze(), labels.float()) # Calculate loss
        # print(f'loss \n {loss}')

        return loss
    
    def validation_step(self, batch):
        images = batch["image"]
        labels = batch["label"]
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]

        # print(f'fps \n {fps}')
        # print(f'bitrate \n {bitrate}')
        out = self(images, fps, bitrate, resolution)      # Generate predictions
        out = out.squeeze()
        # print(f'out {out.size()} \n {out}') # num of valildation data is 100
        # print(f'labels {labels}')
        loss = F.mse_loss(out, labels)   # Calculate loss
        # print(f'loss {loss}')
        
        # Calculate accuracy, i.e.  proportion of the variance in the dependent variable that is predictable 
        val_r2_score = r2_score(labels, out)
        # print(f'val_r2_score {val_r2_score}\n\n\n')
        return {'val_loss_MSE': loss.detach(), 'val_acc_R2': val_r2_score} # val_loss is mse, val_acc is r2 score
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss_MSE'] for x in outputs]
        # print(f'batch_losses \n {batch_losses}')
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc_R2'] for x in outputs]
        # print(f'batch_accs \n {batch_accs}')

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss_MSE': epoch_loss.item(), 'val_acc_R2': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss_MSE: {:.4f}, val_acc_R2: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss_MSE'], result['val_acc_R2']))

class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1), # output (64, 64, 32)
            nn.ReLU(), # output (64, 64, 32)
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),  # output (64, 64, 64)
            nn.ReLU(), # output (64, 64, 64)
            nn.MaxPool2d(2,2), # (32, 32, 64)
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1), # (32, 32, 128)
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1), # (32, 32, 128)
            nn.ReLU(),  # (32, 32, 128)
            nn.MaxPool2d(2,2), # (16, 16, 128)
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), # (16, 16, 256)
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1), # (16, 16, 256)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # (8, 8, 256)
            
            nn.Flatten(),
            nn.Linear(16384,1024), # output vector of size 1024 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32) # embedding of size 32
        )

        self.fc_network = nn.Sequential(
            nn.Linear(32+3, 16),  # Adjust input features to match your extended vector size
            # nn.Linear(32 + 2, 16),  # Adjust input features to match your extended vector size
            nn.ReLU(),
            nn.Linear(16, 1),
            # nn.ReLU(),
            # nn.Linear(8, 1)  # Adjust the output size based on your specific task
        )
    
    def forward(self, images, fps, bitrate, resolution):
        # print(f'image {images.size()} ')
        
        features = self.network(images)    
        # print(f'========= forward =========')
        # print(f'features \n {features[0]}')
        # print(f'resolution {resolution.size()} \n {resolution}')
        # print(f'fps {fps}')
        fps_resolution_bitrate = torch.stack([fps, resolution, bitrate], dim=1).float()  # Example way to combine fps and bitrate
        # resolution = resolution.view(-1, 1)
        # print(f'fps_resolution_bitrate {fps_resolution_bitrate}')
        # Concatenate the 1D tensor to the 2D tensor as the last column
        combined = torch.cat((features, fps_resolution_bitrate), dim=1)
        # print(f'combined {combined.size()}')


        # print(f'fps_bitrate {fps_bitrate.size()}')
        # combined = torch.cat((features, fps_bitrate), dim=1)
        # # print(f'combined {combined.size()}')
        # # print(f'combined {combined}')

        combined = minMaxNormalizer(combined)
        # print(f'combined {combined.size()}') 
        # print(f'combined {combined.size()} \n {combined} \n')               

        output = self.fc_network(combined)

        # print(f'output {output.squeeze()} \n\n\n')

        return output
    

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    # print(f'eval outputs \n {outputs}')
    return model.validation_epoch_end(outputs)


def evaluate_test_data(model, test_loader):
    model.eval()
    with torch.no_grad():  # Ensure gradients are not computed
        for batch in test_loader:
            images = batch["image"]
            labels = batch["label"]
            fps = batch["fps"]
            bitrate = batch["bitrate"]
            resolution = batch["resolution"]

            # print(f'fps \n {fps}')
            # print(f'bitrate \n {bitrate}')
            # print(f'resolution \n {resolution}')
            out = model(images, fps, bitrate, resolution)      # Generate predictions
            out = out.squeeze()
            loss = F.mse_loss(out, labels)   # Calculate loss

            val_r2_score = r2_score(labels, out)
            return {'val_loss_MSE': loss.detach(), 'val_acc_R2': val_r2_score}, out, labels # val_loss is mse, val_acc is r2 score



def fit(epochs, lr, model, train_loader, val_loader, save_path, opt_func = torch.optim.SGD, SAVE=False):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)

    # an epoch is one pass through the entire dataset
    for epoch in range(epochs):
        print(f'================================ epoch {epoch} ================================')
        model.train()
        train_losses = []
        count = 0
        # for each batch, compute gradients for every data
        # after the batch finishes, evaluate
        # requests an iterator from DeviceDataLoader, i.e. __iter__ function
        for batch in train_loader: # batch is a dictionary with 32 images information, e.g. 'fps': [70, 80, ..., 150]
            # print(f'batch {batch[fps]}')
            print(f'=============== batch {count} ===============') # train_size / batch_size
            count += 1
            # fps= batch['fps']
            # print(f"Input batch shape: {images.size()}")
            # print(f"fps batch shape: {fps.size()}")
            loss = model.training_step(batch) # model
            train_losses.append(loss)
            # computes the gradient of the loss with respect to the model parameters
            # part of the backpropagation algorithm, which is how the neural network learns
            loss.backward()  
            # update the model parameters based on the gradients calculated
            optimizer.step()
            # clears the old gradients, so they don't accumulate
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        # print(f'result \n {result}')
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

        if SAVE and epoch % 10 == 0:
            print(f"Epoch {epoch} is a multiple of 20.")
            save_checkpoint(model, optimizer,  f'{save_path}/checkpoint{epoch}.pth', epoch)
    
    return history

# input image 64x64
# learn JOD valules for different fps and bitrate
if __name__ == "__main__":
    data_directory = 'C:/Users/15142/Desktop/data/VRR-cnn-regression/'
    # data_directory = 'C:/Users/15142/Desktop/data/VRR-classification/'
    data_train_directory = f'{data_directory}/train/train' 
    data_test_directory = f'{data_directory}/test_room/test'
    SAVE_MODEL = False
    SAVE_MODEL_HALF_WAY = True
    START_TRAINIGN = True
    TEST_EVAL = True
    PLOT_TEST_RESULT = False
    SAVE_PLOT = True
    TEST_SINGLE_IMG = False

    num_epochs = 1
    lr = 0.001
    opt_func = torch.optim.SGD
    batch_size = 32
    val_size = 1000

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])
    # step1 load data
    dataset = VideoPatchDataset(directory=data_train_directory, transform=transform)
    test_dataset = VideoPatchDataset(directory=data_test_directory, transform=transform)
    train_size = len(dataset) - val_size 
    print(f'total data {len(dataset)}, batch_size {batch_size}')
    print(f'train_size {train_size}, val_size {val_size}, test_size {len(test_dataset)}')
    print(f"Train dataset labels are: \n{dataset.labels}")   # list
    print(f"Test dataset labels are: \n{test_dataset.labels}")   # list

    sample = dataset[0]
    print('sample image has', sample['fps'], 'fps,', sample['bitrate'], 'bps')

    device = get_default_device()
    cuda  = device.type == 'cuda'

    if START_TRAINIGN:
        # step2 split data and prepare batches
        train_data, val_data = random_split(dataset,[train_size,val_size])
        print(f"Length of Train Data : {len(train_data)}")
        print(f"Length of Validation Data : {len(val_data)} \n")

        model = NaturalSceneClassification()
        # print(model)

        train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)

        # show_batch(train_dl) # 128 images
        
        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            train_dl = DeviceDataLoader(train_dl, device)
            val_dl = DeviceDataLoader(val_dl, device)
            to_device(model, device)

            
        # fitting the model on training data and record the result after each epoch
        history = fit(num_epochs, lr, model, train_dl, val_dl, MLDIR, opt_func, SAVE=SAVE_MODEL_HALF_WAY)
        
        now = datetime.now()
        dir_pth = now.strftime("%Y-%m-%d")
        os.makedirs(dir_pth, exist_ok=True)
        hrmin = now.strftime("%H_%M")

        # torch.save(model, f'{save_path}/model.pth')
        # torch.save(model.state_dict(), f'{MLDIR}/classification.pth')

        if SAVE_MODEL:
            torch.save(model.state_dict(), f'{dir_pth}/{hrmin}_cnn_regression.pth')

    if TEST_EVAL:
        model = NaturalSceneClassification()
        model_path = f'checkpoint/2024-04-04/bitrate_fps/19_06_cnn_regression.pth'
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f'test_dataset {len(test_dataset)}')
        test_dl = DataLoader(test_dataset, len(test_dataset))
        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            test_dl = DeviceDataLoader(test_dl, device)
            to_device(model, device)

        result, predictions, true_values = evaluate_test_data(model, test_dl)
        print(f'test result \n {result}')
        # print(f'predictions \n {predictions}')
        # print(f'true_values \n {true_values}')
        # predictions, true_values, _, _, _ = get_values()
        # to_device(predictions, device)
        # to_device(true_values, device)

        if PLOT_TEST_RESULT:
            # print(f"Labels are: {test_dataset.labels}")    

            for batch in test_dl:
                resolution = batch["resolution"]
                test_fps = batch["fps"]
                bitrate = batch["bitrate"]

            # print(f'labels {true_values}')
            # print(f'fps {test_fps}')
            # print(f'resolution {resolution}')
            test_fps_np = test_fps.cpu().numpy()
            bitrate_np = bitrate.cpu().numpy()
            resolution_np = resolution.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            
            x_axis = [360, 480, 720, 864, 1080]
            bitrates_to_plot = [8000]

            # fps_categories = [70, 90, 150]

            fps_categories = {4000: [70, 80, 90], 8000: [70, 90, 150]}
            colors = ['green', 'orange', 'red']
            # colors = ['blue', 'greenyellow', 'red', 'green', 'orange',]
            true_labels_by_fps_8000 = {
                70: [4.799, 5.030, 5.137, 5.198, 5.253],
                90: [5.265, 5.529, 5.669, 5.731, 5.793],
                150: [5.800, 6.108, 6.237, 6.333, 6.439]
            }

            true_labels_by_fps_4000 = {
                70: [4.647, 4.867, 4.954, 5.001, 5.056],
                80: [4.748, 4.956, 5.026, 5.089, 5.158],
                90: [5.043, 5.299, 5.394, 5.451, 5.503]
            }

            true_labels_by_fps = {4000: true_labels_by_fps_4000, 8000: true_labels_by_fps_8000}


            # print(f'bitrate_np \n {bitrate_np}')
            # print(f'indices_for_bitrate \n {indices_for_bitrate}')
            # # print(f'predictions_np \n {predictions_np}\n\n')
            # print(f'test_fps_np \n {test_fps_np}\n\n')
            # print(f'fps_for_bitrate \n {fps_for_bitrate}\n\n')

            for bitrate_to_plot in bitrates_to_plot:
                print(f'\n================= bitrate_to_plot {bitrate_to_plot} ================= ')

                indices_for_bitrate = (bitrate_np == bitrate_to_plot)
                predictions_for_bitrate = predictions_np[indices_for_bitrate]
                resolution_for_bitrate = resolution_np[indices_for_bitrate]
                fps_for_bitrate = test_fps_np[indices_for_bitrate]

                true_labels = true_labels_by_fps[bitrate_to_plot]

                plt.figure(figsize=(10, 6))
                for fps_val, color in zip(fps_categories[bitrate_to_plot], colors):
                    print(f'fps_val {fps_val, color}')
                    indices_for_fps = (fps_for_bitrate == fps_val)
                    resolution_for_fps = resolution_for_bitrate[indices_for_fps]
                    labels_for_fps = predictions_for_bitrate[indices_for_fps]

                    # print(f'indices_for_fps \n {indices_for_fps}')
                    # print(f'resolution_for_fps \n {resolution_for_fps}')
                    # print(f'labels_for_fps \n {labels_for_fps}')

                    plt.plot(x_axis, true_labels[fps_val], color=color, linestyle='--', marker='^', label=f'True FPS {fps_val}')
                    plt.scatter(resolution_for_fps, labels_for_fps, color=color, label=f'Predicted FPS {fps_val}')
        
                    # fps_indices = (fps_for_bitrate == np.full_like(resolution_for_bitrate, fps_val))  # fix fps  
                    # print(f'fps_indices \n {fps_indices}')
                    
                    # for res in x_axis:
                    #     indices = fps_indices & (resolution_for_bitrate == res) # fix fps and resolution
                    #     plt.scatter([res] * indices.sum(), predictions_for_bitrate[indices], color=color, alpha=0.7)

                plt.xticks(x_axis)
                plt.xlabel('Resolution')
                plt.ylabel('JOD')
                plt.title(f'True vs Predicted JOD Bitrate {bitrate_to_plot/1000} Mbps')
                plt.grid(True)
                plt.legend()

                if SAVE_PLOT:
                    current_time = datetime.now().strftime("%m%d_%H%M_%S")
                    plt.savefig(f"plots/{current_time}.png")
                plt.show()