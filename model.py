"""
CNN Model Architecture for CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class CIFAR10CNN(nn.Module):
    """
    CNN Architecture for CIFAR-10:
    Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool ->
    Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear -> Dropout -> Linear
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(CIFAR10CNN, self).__init__()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # MaxPool
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third Conv Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # MaxPool
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        # After 2 max pools: 32 -> 16 -> 8, so 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # First block: Conv -> BatchNorm -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second block: Conv -> BatchNorm -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # First MaxPool
        x = self.maxpool1(x)
        
        # Third block: Conv -> BatchNorm -> ReLU
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Second MaxPool
        x = self.maxpool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_weights(self):
        """Return model weights as a dictionary"""
        return {name: param.clone().detach() for name, param in self.named_parameters()}
    
    def set_weights(self, weights):
        """Set model weights from a dictionary"""
        for name, param in self.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])
    
    def get_flat_weights(self):
        """Return all weights as a single flattened tensor"""
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)
    
    def set_flat_weights(self, flat_weights):
        """Set weights from a flattened tensor"""
        offset = 0
        for param in self.parameters():
            numel = param.data.numel()
            param.data = flat_weights[offset:offset + numel].view(param.data.shape)
            offset += numel
    
    def get_weight_update(self, new_model):
        """Calculate weight delta: new_model - current_model"""
        current_weights = self.get_flat_weights()
        new_weights = new_model.get_flat_weights()
        return new_weights - current_weights


def create_model(device=DEVICE):
    """Create a new model instance"""
    model = CIFAR10CNN(num_classes=NUM_CLASSES)
    return model.to(device)
