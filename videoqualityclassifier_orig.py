import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoQualityClassifier(nn.Module):
    def __init__(self, num_resolutions=3, num_framerates=10):
        """
        Initializes the model architecture.

        Args:
            num_resolutions (int): The number of resolution categories (e.g., 360p, 720p, 1080p).
            num_framerates (int): The number of framerate categories (e.g., 30, 40, ... 120 fps).
        """
        super(VideoQualityClassifier, self).__init__()

        # Shared feature extractor: simple CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Flatten layer for transition to fully connected layers
        self.flatten = nn.Flatten()

        # Fully connected layers for shared processing
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Assumes input images of size 64x64

        # Branch for resolution prediction
        self.fc_res = nn.Linear(128, num_resolutions)

        # Branch for framerate prediction
        self.fc_fps = nn.Linear(128, num_framerates)

    def forward(self, x):
        # Shared convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for fully connected layers
        x = self.flatten(x)

        # Shared fully connected layer
        x = F.relu(self.fc1(x))

        # Separate branches
        res_out = F.softmax(self.fc_res(x), dim=1)  # Softmax for categorical output
        fps_out = F.softmax(self.fc_fps(x), dim=1)  # Softmax for categorical output

        return res_out, fps_out
