"""
Attack Implementations: Byzantine Model Poisoning and Data Poisoning
"""

import torch
import numpy as np
from typing import Tuple, List
from config import *


class ModelPoisoningAttack:
    """Byzantine Model Poisoning Attack"""
    
    def __init__(self, scale=BYZANTINE_SCALE):
        """
        Initialize Byzantine attack
        
        Args:
            scale: Poisoning scale factor
        """
        self.scale = scale
    
    def poison_update(self, global_weights: torch.Tensor, 
                     local_update: torch.Tensor) -> torch.Tensor:
        """
        Apply Byzantine model poisoning to an update
        
        Poison formula: poisoned_delta = -scale * delta
        where delta = local_model - global_model
        
        Args:
            global_weights: Global model weights
            local_update: Local model update (delta)
        
        Returns:
            Poisoned update
        """
        # poisoned_delta = -scale * delta
        poisoned_update = -self.scale * local_update
        return poisoned_update
    
    def poison_update_alternative(self, local_update: torch.Tensor) -> torch.Tensor:
        """
        Alternative poisoning: Add random noise scaled by factor
        
        Args:
            local_update: Local model update
        
        Returns:
            Poisoned update
        """
        noise = torch.randn_like(local_update) * self.scale
        return local_update + noise


class DataPoisoningAttack:
    """Data Poisoning Attack through label flipping"""
    
    def __init__(self, poison_ratio=POISON_RATIO, label_flip_type='random'):
        """
        Initialize data poisoning attack
        
        Args:
            poison_ratio: Fraction of samples to poison (0-1)
            label_flip_type: 'random' or 'specific' (Cat<->Dog)
        """
        self.poison_ratio = poison_ratio
        self.label_flip_type = label_flip_type
        self.poisoned_indices = set()
    
    def poison_dataset(self, data_indices: np.ndarray, 
                      labels: np.ndarray, 
                      num_classes: int = NUM_CLASSES) -> np.ndarray:
        """
        Poison dataset labels
        
        Args:
            data_indices: Indices of samples to poison
            labels: Original labels
            num_classes: Number of classes
        
        Returns:
            Poisoned labels
        """
        poisoned_labels = labels.copy()
        
        # Calculate number of samples to poison
        num_samples = len(data_indices)
        num_poison = max(1, int(num_samples * self.poison_ratio))
        
        # Randomly select samples to poison
        poison_indices = np.random.choice(data_indices, size=num_poison, replace=False)
        self.poisoned_indices = set(poison_indices)
        
        if self.label_flip_type == 'random':
            # Random label flipping
            for idx in poison_indices:
                current_label = poisoned_labels[idx]
                new_label = np.random.randint(0, num_classes)
                # Ensure label is different
                while new_label == current_label:
                    new_label = np.random.randint(0, num_classes)
                poisoned_labels[idx] = new_label
        
        elif self.label_flip_type == 'specific':
            # Specific label flipping (e.g., Cat <-> Dog)
            # For CIFAR-10: Cat=3, Dog=5
            label_map = {3: 5, 5: 3}  # Cat <-> Dog
            
            for idx in poison_indices:
                current_label = poisoned_labels[idx]
                if current_label in label_map:
                    poisoned_labels[idx] = label_map[current_label]
        
        return poisoned_labels
    
    def poison_batch(self, images: torch.Tensor, labels: torch.Tensor, 
                    num_classes: int = NUM_CLASSES) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Poison a batch of data
        
        Args:
            images: Batch of images
            labels: Batch of labels
            num_classes: Number of classes
        
        Returns:
            (poisoned_images, poisoned_labels)
        """
        poisoned_labels = labels.clone()
        batch_size = len(labels)
        
        # Calculate number of samples to poison in batch
        num_poison = max(1, int(batch_size * self.poison_ratio))
        
        # Randomly select samples to poison
        poison_indices = np.random.choice(batch_size, size=num_poison, replace=False)
        
        if self.label_flip_type == 'random':
            for idx in poison_indices:
                current_label = poisoned_labels[idx].item()
                new_label = np.random.randint(0, num_classes)
                while new_label == current_label:
                    new_label = np.random.randint(0, num_classes)
                poisoned_labels[idx] = new_label
        
        elif self.label_flip_type == 'specific':
            label_map = {3: 5, 5: 3}  # Cat <-> Dog
            for idx in poison_indices:
                current_label = poisoned_labels[idx].item()
                if current_label in label_map:
                    poisoned_labels[idx] = label_map[current_label]
        
        return images, poisoned_labels
    
    def get_poisoned_indices(self) -> set:
        """Get indices of poisoned samples"""
        return self.poisoned_indices


class AttackFactory:
    """Factory for creating attacks"""
    
    @staticmethod
    def create_model_poisoning(scale=BYZANTINE_SCALE) -> ModelPoisoningAttack:
        """Create model poisoning attack"""
        return ModelPoisoningAttack(scale=scale)
    
    @staticmethod
    def create_data_poisoning(poison_ratio=POISON_RATIO, 
                             label_flip_type='random') -> DataPoisoningAttack:
        """Create data poisoning attack"""
        return DataPoisoningAttack(poison_ratio=poison_ratio, 
                                  label_flip_type=label_flip_type)


class AttackManager:
    """
    Manages attacks for multiple clients
    """
    
    def __init__(self, num_clients: int, num_byzantine: int = BYZANTINE_CLIENTS,
                 byzantine_scale: float = BYZANTINE_SCALE,
                 poison_ratio: float = POISON_RATIO):
        """
        Initialize attack manager
        
        Args:
            num_clients: Total number of clients
            num_byzantine: Number of Byzantine clients
            byzantine_scale: Byzantine attack scale
            poison_ratio: Data poisoning ratio
        """
        self.num_clients = num_clients
        self.num_byzantine = num_byzantine
        self.byzantine_scale = byzantine_scale
        self.poison_ratio = poison_ratio
        
        # Randomly select Byzantine clients
        self.byzantine_clients = set(
            np.random.choice(num_clients, size=num_byzantine, replace=False)
        )
        
        self.model_poisoning = ModelPoisoningAttack(scale=byzantine_scale)
        self.data_poisoning = DataPoisoningAttack(poison_ratio=poison_ratio)
    
    def is_byzantine_client(self, client_id: int) -> bool:
        """Check if client is Byzantine"""
        return client_id in self.byzantine_clients
    
    def is_data_poisoning_client(self, client_id: int) -> bool:
        """Check if client is data poisoning (for now, same as Byzantine)"""
        return client_id in self.byzantine_clients
    
    def apply_model_poisoning(self, client_id: int, 
                             local_update: torch.Tensor) -> torch.Tensor:
        """Apply model poisoning if client is Byzantine"""
        if self.is_byzantine_client(client_id):
            return self.model_poisoning.poison_update(None, local_update)
        return local_update
    
    def apply_data_poisoning(self, images: torch.Tensor, 
                            labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data poisoning"""
        return self.data_poisoning.poison_batch(images, labels)
    
    def get_byzantine_clients(self) -> List[int]:
        """Get list of Byzantine client IDs"""
        return list(self.byzantine_clients)
