import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Using ResNet18 as mentioned in the UI dashboard
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.resnet.fc.in_features
        # Single output for binary classification (Real vs Fake)
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)
