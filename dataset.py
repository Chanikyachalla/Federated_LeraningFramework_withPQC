"""
Dataset Utilities for CIFAR-10 with Non-IID Dirichlet Distribution
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from config import *


class CIFAR10Dataset:
    """
    CIFAR-10 dataset with non-IID Dirichlet distribution for federated learning
    """
    
    def __init__(self, data_dir='./data', alpha=DIRICHLET_ALPHA, num_clients=NUM_CLIENTS):
        self.data_dir = data_dir
        self.alpha = alpha
        self.num_clients = num_clients
        self.random_seed = RANDOM_SEED
        
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # Download and prepare data
        self._download_data()
        self._distribute_data()
    
    def _download_data(self):
        """Download CIFAR-10 dataset"""
        # Transform for CIFAR-10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Download datasets
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        
        self.num_classes = NUM_CLASSES
    
    def _distribute_data(self):
        """
        Distribute data to clients using Dirichlet distribution
        Creates non-IID data distribution
        """
        # Get all labels
        train_labels = np.array(self.train_dataset.targets)
        num_samples = len(train_labels)
        
        # Create label-wise indices
        label_indices = {i: np.where(train_labels == i)[0] for i in range(self.num_classes)}
        
        # Allocate data to clients using Dirichlet distribution
        self.client_indices = {i: [] for i in range(self.num_clients)}
        
        for class_id in range(self.num_classes):
            indices = label_indices[class_id]
            np.random.shuffle(indices)
            
            # Sample from Dirichlet distribution
            distribution = np.random.dirichlet(np.ones(self.num_clients) * self.alpha)
            
            # Allocate samples to clients based on distribution
            splits = (distribution * len(indices)).astype(int)
            
            # Handle rounding issues
            if splits.sum() != len(indices):
                splits[-1] = len(indices) - splits[:-1].sum()
            
            start_idx = 0
            for client_id, split_size in enumerate(splits):
                self.client_indices[client_id].extend(
                    indices[start_idx:start_idx + split_size]
                )
                start_idx += split_size
    
    def get_client_dataset(self, client_id, train=True):
        """
        Get dataset for a specific client
        
        Args:
            client_id: Client ID
            train: If True, return training data, else test data
        
        Returns:
            Subset dataset
        """
        if train:
            dataset = self.train_dataset
            indices = self.client_indices[client_id]
        else:
            dataset = self.test_dataset
            indices = np.arange(len(dataset))
        
        return Subset(dataset, indices)
    
    def get_client_dataloader(self, client_id, batch_size=BATCH_SIZE, train=True, shuffle=None):
        """
        Get DataLoader for a specific client
        
        Args:
            client_id: Client ID
            batch_size: Batch size
            train: If True, return training data, else test data
            shuffle: If None, shuffle only for training data
        
        Returns:
            DataLoader
        """
        if shuffle is None:
            shuffle = train
        
        dataset = self.get_client_dataset(client_id, train=train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_global_testloader(self, batch_size=BATCH_SIZE):
        """Get DataLoader for global test set"""
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    
    def get_data_info(self):
        """Get information about data distribution"""
        info = {}
        for client_id in range(self.num_clients):
            train_dataset = self.get_client_dataset(client_id, train=True)
            labels = [self.train_dataset.targets[idx] for idx in train_dataset.indices]
            info[f'client_{client_id}'] = {
                'num_samples': len(labels),
                'class_distribution': {
                    class_id: sum(1 for l in labels if l == class_id)
                    for class_id in range(self.num_classes)
                }
            }
        return info


def get_cifar10_dataset(data_dir='./data'):
    """
    Convenience function to get CIFAR-10 dataset
    
    Returns:
        CIFAR10Dataset instance
    """
    return CIFAR10Dataset(data_dir=data_dir)
