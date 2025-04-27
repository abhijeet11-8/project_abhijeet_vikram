import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn

class ImprovedFER2013Model(nn.Module):
    def __init__(self, num_classes=7):
        super(ImprovedFER2013Model, self).__init__()

        self.features = nn.Sequential(
            # Block 1 (Increased depth with residual connections)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24
            nn.Dropout(0.3),
            
            # Block 2 (Residual Block 1)
            self._residual_block(64, 128),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12
            
            # Block 3 (Residual Block 2)
            self._residual_block(128, 256),
            nn.MaxPool2d(2, 2),  # 12x12 -> 6x6
            
            # Block 4 (Residual Block 3)
            self._residual_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _residual_block(self, in_channels, out_channels):
        """Helper function to create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class TheModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=7):
        super(TheModel, self).__init__()
        # Change the input channels to 1 for grayscale images
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Calculate the output size of the final feature map dynamically
        self._dummy_input = torch.zeros(1, input_channels, 48, 48)  # Assuming 48x48 input images
        self._final_flattened_size = self._get_flattened_size(self._dummy_input)

        # Define fully connected layers after determining the flattened size
        self.fc1 = nn.Linear(self._final_flattened_size, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_flattened_size(self, x):
        # Forward pass once to calculate the output size
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x.view(1, -1).size(1)  # Flatten the tensor and return the number of features

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor before passing to fc layers

        # Pass through fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)  # Apply Dropout
        x = self.fc2(x)
        return x