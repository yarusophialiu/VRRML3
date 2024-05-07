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

# regressin, learn the curves
# https://docs.google.com/presentation/d/16yqaaq5zDZ5-S4394VLBUfxpNjM7nlpssqcShFkklec/edit#slide=id.g2c751bc0d9c_0_18



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
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]
        # velocity = batch["velocity"]

        # TODO: convert labels into res_targets, fps_targets 
        res_targets = batch["res_targets"]
        fps_targets = batch["fps_targets"]
        # print(f'\n\n\n training step')
        res_out, fps_out = self(images, fps, bitrate, resolution)  # NaturalSceneClassification.forward
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
        


class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self, num_framerates, num_resolutions):
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

        self.fc_network = nn.Sequential(
            nn.Linear(32+2, 16),  # Adjust input features to match your extended vector size
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

    
    def forward(self, images, fps, bitrate, resolution):
        # print(f'image {images.size()} ')
        
        features = self.network(images)    
        # print(f'========= forward =========')
        # print(f'features \n {features[0]}')
        # print(f'resolution {resolution.size()} \n {resolution}')
        # print(f'fps {fps}')
        fps_resolution_bitrate = torch.stack([fps, bitrate], dim=1).float()  # Example way to combine fps and bitrate
        # print(f'fps_resolution_bitrate {fps_resolution_bitrate}')
        combined = torch.cat((features, fps_resolution_bitrate), dim=1)
        if NORMALIZATION:
            combined = minMaxNormalizer(combined)
        # print(f'\n\n\nnormalization {NORMALIZATION}')
        
        # print(f'combined {combined.size()}\n')               

        x = self.fc_network(combined)

        res_out = F.softmax(self.fc_res(x), dim=1)  # Softmax for categorical output
        fps_out = F.softmax(self.fc_fps(x), dim=1)  # Softmax for categorical output
        # print(f'res_out {res_out.squeeze(1)} \n\n\n')

        return res_out, fps_out

    

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
            # labels = batch["label"]
            fps = batch["fps"]
            bitrate = batch["bitrate"]
            resolution = batch["resolution"]
            # velocity = batch["velocity"]
        
            print(f'bitrate \n {bitrate}')

            # arr = [30, 30, 30, 40, 40, 50, 50, 50]
            # tensor = torch.tensor(arr)
            unique_indices = {}

            # Iterate over the tensor and populate the dictionary
            for index, value in enumerate(bitrate.tolist()):
                if value not in unique_indices:
                    unique_indices[value] = index

            # TODO: convert labels into res_targets, fps_targets 
            res_targets = batch["res_targets"]
            fps_targets = batch["fps_targets"]
            res_out, fps_out = model(images, fps, bitrate, resolution)  # NaturalSceneClassification.forward
            # print(f'training_step out {out.size()} \n {out.squeeze()}')
            # print(f'labels out {labels}')
            total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
            framerate_accuracy, resolution_accuracy, both_correct_accuracy = compute_accuracy(fps_out, res_out, fps_targets, res_targets)

            fps = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
            resolution = [360, 480, 720, 864, 1080]

            fps_idx = torch.argmax(fps_out, dim=1)
            res_idx = torch.argmax(res_out, dim=1)

            fps_values = [fps[idx] for idx in fps_idx]
            res_values = [resolution[idx] for idx in res_idx]


            return {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
                    'both_acc': both_correct_accuracy}, res_out, fps_out, res_targets, fps_targets, \
                        res_values, fps_values, unique_indices


def fit(epochs, lr, model, train_loader, val_loader, save_path, opt_func, SAVE=False):
    
    history = []
    train_accuracies = []
    val_accuracies = []
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
            # print(f'=============== batch {count} ===============') # train_size / batch_size
            count += 1
            # fps= batch['fps']
            # print(f"Input batch shape: {images.size()}")
            # print(f"fps batch shape: {fps.size()}")
            # get accuracy
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
            # save_checkpoint(model, optimizer,  f'{save_path}/checkpoint{epoch}.pth', epoch)
    
    return history

# input image 64x64, enlarged decocded OR reference patch
# no resolution
if __name__ == "__main__":
    data_directory = 'C:/Users/15142/Desktop/data/VRR-video-classification/'
    # data_directory = 'C:/Users/15142/Desktop/data/VRR-classification/'
    data_train_directory = f'{data_directory}/bistro-fast/train_dec_ref_bistro/train' 
    # data_test_directory = f'{data_directory}/test_dec_ref_suntemple/test'
    data_test_directory = f'{data_directory}/bistro-normal/test_consecutive_dec/test'

    SAVE_MODEL = False
    SAVE_MODEL_HALF_WAY = False
    START_TRAINIGN = False
    TEST_EVAL = False
    PLOT_TEST_RESULT = False
    SAVE_PLOT = True
    TEST_SINGLE_IMG = False

    num_epochs = 40
    lr = 0.001
    # opt_func = torch.optim.SGD
    opt_func = torch.optim.Adam
    batch_size = 128
    val_size = 4096

    num_framerates, num_resolutions = 10, 5
    TYPE = "decoded" # decoded, reference
    NORMALIZATION = True
    min_bitrate = 2000
    max_bitrate = 6000

    # step1 load data 
    dataset = VideoSinglePatchDataset(data_train_directory, TYPE, min_bitrate, max_bitrate) # len 27592
    test_dataset = VideoSinglePatchDataset(data_test_directory, TYPE, min_bitrate, max_bitrate) 
    train_size = len(dataset) - val_size 
    print(f'total data {len(dataset)}, batch_size {batch_size}')
    print(f'train_size {train_size}, val_size {val_size}, test_size {len(test_dataset)}\n')
    # print(f"Train dataset  labels are: \n{dataset.labels}")
    print(f"Train dataset fps labels are: \n{dataset.fps_targets}")
    print(f"Train dataset res labels are: \n{dataset.res_targets}\n")
    print(f"Test dataset fps labels are: \n{test_dataset.fps_targets}")
    print(f"Test dataset res labels are: \n{test_dataset.res_targets}\n")
    sample = dataset[0]
    print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')

    device = get_default_device()
    cuda  = device.type == 'cuda'

    if START_TRAINIGN:
        # step2 split data and prepare batches
        train_data, val_data = random_split(dataset,[train_size,val_size])
        print(f"Length of Train Data : {len(train_data)}")
        print(f"Length of Validation Data : {len(val_data)} \n")

        model = NaturalSceneClassification(num_framerates, num_resolutions)
        # print(model)

        train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        val_dl = DataLoader(val_data, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)
        
        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            train_dl = DeviceDataLoader(train_dl, device)
            val_dl = DeviceDataLoader(val_dl, device)
            to_device(model, device)

        model_path = ""
        if SAVE_MODEL:
            now = datetime.now()
            dir_pth = now.strftime("%Y-%m-%d")
            os.makedirs(dir_pth, exist_ok=True)
            hrmin = now.strftime("%H_%M")
            model_path = os.path.join(dir_pth, hrmin)
            os.makedirs(model_path, exist_ok=True)
        # fitting the model on training data and record the result after each epoch
        history = fit(num_epochs, lr, model, train_dl, val_dl, model_path, opt_func, SAVE=SAVE_MODEL_HALF_WAY)
        
        if SAVE_MODEL:
            torch.save(model.state_dict(), f'{model_path}/classification_{TYPE}.pth')

    if TEST_EVAL:
        model = NaturalSceneClassification(num_framerates, num_resolutions)
        # model_path = f'2024-05-05/20_14/classification_reference.pth'
        model_path = f'2024-05-05/14_26/classification_ref_adam.pth'
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f'test_dataset {len(test_dataset)}')
        test_dl = DataLoader(test_dataset, len(test_dataset))
        if device.type == 'cuda':
            print(f'\nLoading data to cuda...')
            test_dl = DeviceDataLoader(test_dl, device)
            to_device(model, device)


        # {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
                    # 'both_acc': both_correct_accuracy} 

        result, res_out, fps_out, res_targets, fps_targets, \
            res_values, fps_values, unique_indices = evaluate_test_data(model, test_dl)
        
        # unique_indices_arr = [val for k, val in unique_indices.items()]
        bitrate_predictions = {}
        print(f"First indices of unique values: {unique_indices}")

        res_values = torch.tensor(res_values)
        fps_values = torch.tensor(fps_values)

        for k, val in unique_indices.items():
            bitrate_predictions[k] = []
            bitrate_predictions[k].append(res_values[val].item())
            bitrate_predictions[k].append(fps_values[val].item())

        print(f'test result \n {result}\n')
        # print(f'res_out \n {res_out}')
        # print(f'fps_out \n {fps_out}')
        # print(f'res_targets \n {res_targets}')
        # print(f'fps_targets \n {fps_targets}')
        # print(f'res_values \n {(res_values)}')
        # print(f'res_values \n {torch.unique(res_values)}\n')
        # print(f'fps_values \n {(fps_values)}')
        # print(f'fps_values \n {torch.unique(fps_values)}\n')
        print(f'bitrate_predictions \n {bitrate_predictions}')


        if PLOT_TEST_RESULT:
            # print(f"Labels are: {test_dataset.labels}")    

            for batch in test_dl:
                resolution = batch["resolution"]
                test_fps = batch["fps"]
                bitrate = batch["bitrate"]
                # velocity = batch["velocity"]

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