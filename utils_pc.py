import os 
import torch
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import Dataset

from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image



def show_patch(patch):
    plt.imshow(patch)
    plt.axis('off')  # Hide axis
    plt.show()


def minMaxNormalizer(feature_tensor):
    """Takes the Torch.tensor object containing the features and performs min-max normalization on the Torch.tensor.
    The function iterates through each column and performs scaling on them individually.
    
    Args-
        feature_tensor- Tensor containing training features
    """
    # print(f'feature_tensor \n {feature_tensor.size()}')
    total_cols = feature_tensor.size()[1] # total unmber of columns 
    # print(f'total_cols {total_cols}')

    normalized_feature_tensor = torch.zeros_like(feature_tensor)

    # total_cols-2 to skip fps and bps, when fps and bps are the same, result is nan
    for i in range(total_cols): # iterating through each column
        feature_col = feature_tensor[:, i]
        # print(f'feature_col {feature_col}')

        maximum = torch.max(feature_col) # maximum stores max value of the column
        minimum = torch.min(feature_col) # minimum stores min value of the column
        scaled_feature_col = (feature_col - minimum) / (maximum - minimum)
        # scaled_feature_col = feature_col

        normalized_feature_tensor[:, i] = scaled_feature_col # min-max scalinng of each element of the column
        # print(f'feature_tensor[:, i] {feature_tensor[:, i]}')

        # if i == 33:
        #     print(f'max min {maximum, minimum}')
        #     print(f'feature_col {feature_col}')
        #     print(f'scaled_feature_col {scaled_feature_col}')
    return normalized_feature_tensor


def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    "Move data to the device"
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)


def r2_score(target, prediction):
    """Calculates the r2 score of the model"""
    r2 = 1- torch.sum((target-prediction)**2) / torch.sum((target-target.float().mean())**2)
    return r2


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    print("Keys in the checkpoint:", checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No optimizer state found in checkpoint!")

    # Load the epoch number
    epoch = checkpoint.get('epoch', None)
    
    return model, optimizer, epoch



def plot_accuracy_curve(accuracies):

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(accuracies)), y=accuracies, marker='o', label='Accuracy')

    plt.title("Accuracy Curve Over Time")
    plt.xlabel("Epochs / Iterations")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()



def compute_weighted_loss(res_out, fps_out, res_targets, fps_targets):
    loss_fn_res = nn.CrossEntropyLoss()
    loss_fn_fps = nn.CrossEntropyLoss()
    loss_res = loss_fn_res(res_out, res_targets)
    loss_fps = loss_fn_fps(fps_out, fps_targets)

    total_loss = loss_res + loss_fps
    return total_loss

    
def compute_accuracy(fps_out, res_out, fps_targets, res_targets):
        # print(f'fps_out \n {}')
        _, fps_preds = torch.max(fps_out, dim=1)
        # print(f'fps_preds \n {fps_preds}')
        _, res_preds = torch.max(res_out, dim=1)
        # print(f'res_out \n {res_out}')
        # print(f'res_preds \n {res_preds}')

        framerate_accuracy = torch.tensor(torch.sum(fps_preds == fps_targets).item() / len(fps_targets))
        resolution_accuracy = torch.tensor(torch.sum(res_preds == res_targets).item() / len(res_targets))

        both_correct_accuracy = torch.tensor(torch.sum((res_preds == res_targets) & (fps_preds == fps_targets)).item() / len(res_targets))

  
        # res_pred = torch.argmax(res_out, dim=1)
        # fps_pred = torch.argmax(fps_out, dim=1)

        # res_correct = (res_pred == res_targets).sum().item()
        # fps_correct = (fps_pred == fps_targets).sum().item()
        # both_correct = ((res_pred == res_targets) & (fps_pred == fps_targets)).sum().item()

        # total_samples = len(res_targets)

        # resolution_accuracy = res_correct / total_samples
        # framerate_accuracy = fps_correct / total_samples
        # both_correct_accuracy = both_correct / total_samples
        return framerate_accuracy, resolution_accuracy, both_correct_accuracy

def plot_test_result(test_dl, predictions, epoch="", SAVE_PLOT=False):
    for batch in test_dl:
        resolution = batch["resolution"]
        test_fps = batch["fps"]
        bitrate = batch["bitrate"]

        test_fps_np = test_fps.cpu().numpy()
        bitrate_np = bitrate.cpu().numpy()
        resolution_np = resolution.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        
        x_axis = [360, 480, 720, 864, 1080]
        bitrates_to_plot = [4000, 8000]

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

        for bitrate_to_plot in bitrates_to_plot:
            print(f'================= bitrate_to_plot {bitrate_to_plot} ================= ')

            indices_for_bitrate = (bitrate_np == bitrate_to_plot)
            predictions_for_bitrate = predictions_np[indices_for_bitrate]
            resolution_for_bitrate = resolution_np[indices_for_bitrate]
            fps_for_bitrate = test_fps_np[indices_for_bitrate]

            true_labels = true_labels_by_fps[bitrate_to_plot]

            plt.figure(figsize=(10, 6))
            for fps_val, color in zip(fps_categories[bitrate_to_plot], colors):
                # print(f'fps_val {fps_val, color}')
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
                plt.savefig(f"plots/{current_time}_{bitrate_to_plot}_{epoch}.png")
            plt.show()
            # plt.show(block = False)
            # plt.pause(1)
            plt.close('all')



def predict_img_class(img, fps, bitrate, model):
    """ Predict the class of image and Return Predicted Class"""
    img = to_device(img.unsqueeze(0), device)
    prediction = model(img, fps, bitrate)
    print(f'prediction {prediction}')
    # _, preds = torch.max(prediction, dim = 1)
    # return dataset.classes[preds[0].item()]
    return prediction


def show_patch(patch):
    plt.imshow(patch)
    plt.axis('off')  # Hide axis
    plt.show()



def display_img(img,label, dataset):
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

