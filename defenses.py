"""
Defense Mechanisms: FedAvg and FoolsGold
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
from config import *


class FedAvgDefense:
    """
    Standard FedAvg aggregation (baseline defense)
    Simply averages all updates
    """
    
    def __init__(self):
        self.name = 'FedAvg'
    
    def aggregate(self, updates: List[torch.Tensor], 
                 weights: List[float] = None) -> torch.Tensor:
        """
        Aggregate updates using FedAvg
        
        Args:
            updates: List of model updates
            weights: Optional weights for each update
        
        Returns:
            Aggregated update
        """
        if not updates:
            return None
        
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        
        # Weighted average
        aggregated = None
        for update, weight in zip(updates, weights):
            if aggregated is None:
                aggregated = weight * update
            else:
                aggregated += weight * update
        
        return aggregated


class FoolsGoldDefense:
    """
    FoolsGold Defense: Detects colluding malicious clients
    
    Algorithm:
    1. Store historical updates (cosine similarity matrix)
    2. Compute update similarities
    3. Identify potentially malicious clients
    4. Reduce their contribution in aggregation
    """
    
    def __init__(self, history_size: int = FOOLSGOLD_HISTORY_SIZE,
                 similarity_threshold: float = FOOLSGOLD_SIMILARITY_THRESHOLD):
        """
        Initialize FoolsGold defense
        
        Args:
            history_size: Number of rounds to keep history
            similarity_threshold: Threshold for similarity-based detection
        """
        self.name = 'FoolsGold'
        self.history_size = history_size
        self.similarity_threshold = similarity_threshold
        self.history = deque(maxlen=history_size)
        self.client_scores = {}
    
    def _compute_cosine_similarity(self, v1: torch.Tensor, 
                                   v2: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors"""
        # Flatten vectors
        v1_flat = v1.view(-1)
        v2_flat = v2.view(-1)
        
        # Compute cosine similarity
        dot_product = torch.sum(v1_flat * v2_flat)
        norm_v1 = torch.sqrt(torch.sum(v1_flat ** 2))
        norm_v2 = torch.sqrt(torch.sum(v2_flat ** 2))
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm_v1 * norm_v2)
        return float(cosine_sim.cpu().detach().numpy())
    
    def compute_similarity_matrix(self, updates: List[torch.Tensor]) -> np.ndarray:
        """
        Compute similarity matrix between all updates
        
        Args:
            updates: List of model updates
        
        Returns:
            Similarity matrix (num_clients x num_clients)
        """
        num_updates = len(updates)
        similarity_matrix = np.zeros((num_updates, num_updates))
        
        for i in range(num_updates):
            for j in range(num_updates):
                sim = self._compute_cosine_similarity(updates[i], updates[j])
                similarity_matrix[i, j] = sim
        
        return similarity_matrix
    
    def detect_adversaries(self, updates: List[torch.Tensor],
                          similarity_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect potentially malicious clients
        
        Args:
            updates: List of model updates
            similarity_matrix: Similarity matrix
        
        Returns:
            (scores, suspicious_indices)
        """
        num_updates = len(updates)
        
        # Compute score for each update based on similarity to others
        # High similarity to many others might indicate collusion
        scores = np.zeros(num_updates)
        
        for i in range(num_updates):
            # Count how many updates are similar to this one
            similar_count = np.sum(
                similarity_matrix[i] >= self.similarity_threshold
            )
            scores[i] = similar_count / num_updates
        
        # Detect suspicious clients (high similarity but potentially attacking)
        suspicious_indices = np.where(scores > 0.5)[0]
        
        return scores, suspicious_indices
    
    def aggregate(self, updates: List[torch.Tensor],
                 client_ids: List[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Aggregate updates with FoolsGold defense
        
        Args:
            updates: List of model updates
            client_ids: Client IDs (optional)
        
        Returns:
            (aggregated_update, defense_info)
        """
        if not updates:
            return None, {}
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(updates)
        
        # Detect adversaries
        scores, suspicious_indices = self.detect_adversaries(updates, similarity_matrix)
        
        # Store history
        self.history.append(similarity_matrix)
        
        # Compute weights for aggregation
        weights = np.ones(len(updates))
        
        # Reduce weights for suspicious clients
        for idx in suspicious_indices:
            weights[idx] *= 0.5  # Reduce contribution
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Aggregate with weighted average
        aggregated = None
        for i, (update, weight) in enumerate(zip(updates, weights)):
            if aggregated is None:
                aggregated = weight * update
            else:
                aggregated += weight * update
        
        # Store stats
        defense_info = {
            'similarity_matrix': similarity_matrix,
            'scores': scores,
            'suspicious_indices': suspicious_indices,
            'weights': weights,
            'num_suspicious': len(suspicious_indices)
        }
        
        return aggregated, defense_info
    
    def update_scores(self, scores: np.ndarray):
        """Update client scores over time"""
        self.client_scores = scores


class ManhattanDistanceDefense:
    """
    Manhattan Distance Defense: Detects anomalous updates based on L1 distance
    
    Algorithm:
    1. Compute Manhattan distance from each update to the median
    2. Identify outliers (updates far from median)
    3. Reduce weight of outlier updates
    4. Aggregate weighted updates
    """
    
    def __init__(self, threshold: float = MANHATTAN_DISTANCE_THRESHOLD,
                 deviation_factor: float = MANHATTAN_DISTANCE_DEVIATION_FACTOR):
        """
        Initialize Manhattan Distance defense
        
        Args:
            threshold: Distance threshold multiplier for deviation
            deviation_factor: Multiplier for standard deviation in outlier detection
        """
        self.name = 'ManhattanDistance'
        self.threshold = threshold
        self.deviation_factor = deviation_factor
    
    def _compute_manhattan_distance(self, v1: torch.Tensor, 
                                   v2: torch.Tensor) -> float:
        """Compute Manhattan distance (L1 norm) between two vectors"""
        # Flatten vectors
        v1_flat = v1.view(-1)
        v2_flat = v2.view(-1)
        
        # Compute L1 distance
        distance = torch.sum(torch.abs(v1_flat - v2_flat))
        return float(distance.cpu().detach().numpy())
    
    def compute_median_update(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute element-wise median of all updates
        
        Args:
            updates: List of model updates
        
        Returns:
            Median update tensor
        """
        # Stack all updates
        stacked = torch.stack([u.view(-1) for u in updates], dim=1)
        
        # Compute median along dimension 1 (across clients)
        median = torch.median(stacked, dim=1)[0]
        
        # Return in original shape (use first update's shape)
        return median.view(updates[0].shape)
    
    def detect_outliers(self, updates: List[torch.Tensor],
                       median_update: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outlier updates based on Manhattan distance
        
        Args:
            updates: List of model updates
            median_update: Median update
        
        Returns:
            (distances, outlier_indices)
        """
        num_updates = len(updates)
        distances = np.zeros(num_updates)
        
        # Compute distance from each update to median
        for i, update in enumerate(updates):
            distances[i] = self._compute_manhattan_distance(update, median_update)
        
        # Compute mean and standard deviation of distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Identify outliers: distance > mean + deviation_factor * std
        threshold_distance = mean_distance + self.deviation_factor * std_distance
        outlier_indices = np.where(distances > threshold_distance)[0]
        
        return distances, outlier_indices
    
    def aggregate(self, updates: List[torch.Tensor],
                 client_ids: List[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Aggregate updates with Manhattan Distance defense
        
        Args:
            updates: List of model updates
            client_ids: Client IDs (optional)
        
        Returns:
            (aggregated_update, defense_info)
        """
        if not updates:
            return None, {}
        
        # Compute median update
        median_update = self.compute_median_update(updates)
        
        # Detect outliers
        distances, outlier_indices = self.detect_outliers(updates, median_update)
        
        # Compute weights for aggregation
        weights = np.ones(len(updates))
        
        # Reduce weights for outlier updates
        for idx in outlier_indices:
            weights[idx] *= 0.5  # Reduce contribution
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Aggregate with weighted average
        aggregated = None
        for i, (update, weight) in enumerate(zip(updates, weights)):
            if aggregated is None:
                aggregated = weight * update
            else:
                aggregated += weight * update
        
        # Store stats
        defense_info = {
            'distances': distances,
            'median_update': median_update,
            'outlier_indices': outlier_indices,
            'weights': weights,
            'num_outliers': len(outlier_indices),
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances))
        }
        
        return aggregated, defense_info


class AdaptiveDefense:
    """Base class for adaptive defenses that can switch strategies"""
    
    def __init__(self):
        self.defense_methods = {
            'fedavg': FedAvgDefense(),
            'foolsgold': FoolsGoldDefense(),
            'manhattan': ManhattanDistanceDefense()
        }
        self.current_defense = 'fedavg'
    
    def set_defense(self, defense_name: str):
        """Set current defense method"""
        if defense_name in self.defense_methods:
            self.current_defense = defense_name
        else:
            raise ValueError(f"Unknown defense: {defense_name}")
    
    def aggregate(self, updates: List[torch.Tensor], 
                 client_ids: List[int] = None) -> Tuple[torch.Tensor, Dict]:
        """Aggregate using current defense method"""
        defense = self.defense_methods[self.current_defense]
        
        if self.current_defense == 'fedavg':
            return defense.aggregate(updates), {'defense': 'FedAvg'}
        elif self.current_defense == 'foolsgold':
            return defense.aggregate(updates, client_ids)
        elif self.current_defense == 'manhattan':
            return defense.aggregate(updates, client_ids)


def create_defense(defense_name: str = DEFENSE_METHOD) -> torch.nn.Module:
    """
    Factory function to create defense mechanism
    
    Args:
        defense_name: 'fedavg', 'foolsgold', or 'manhattan'
    
    Returns:
        Defense instance
    """
    if defense_name == 'fedavg':
        return FedAvgDefense()
    elif defense_name == 'foolsgold':
        return FoolsGoldDefense()
    elif defense_name == 'manhattan':
        return ManhattanDistanceDefense()
    else:
        raise ValueError(f"Unknown defense: {defense_name}")
