"""
Evaluation Metrics and Statistics Tracking
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path
from config import *


class MetricsTracker:
    """
    Tracks all evaluation metrics for federated learning
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize metrics tracker
        
        Args:
            experiment_name: Name of experiment
        """
        self.experiment_name = experiment_name
        self.metrics = {
            'training_loss': [],
            'test_accuracy': [],
            'test_loss': [],
            'attack_success_rate': [],
            'aggregation_time': [],
            'encryption_time': [],
            'decryption_time': [],
            'signature_verification_time': [],
            'communication_overhead': [],
            'num_valid_updates': [],
            'num_invalid_updates': [],
            'foolsgold_suspicious_clients': [],
            'client_losses': []
        }
        self.round_data = []
    
    def add_round(self, round_num: int, round_stats: Dict):
        """
        Add metrics from a round
        
        Args:
            round_num: Round number
            round_stats: Statistics from that round
        """
        # Training loss
        if 'client_losses' in round_stats:
            avg_loss = np.mean(round_stats['client_losses'])
            self.metrics['training_loss'].append(avg_loss)
            self.metrics['client_losses'].append(round_stats['client_losses'])
        
        # Test metrics
        if 'test_accuracy' in round_stats:
            self.metrics['test_accuracy'].append(round_stats['test_accuracy'])
        if 'test_loss' in round_stats:
            self.metrics['test_loss'].append(round_stats['test_loss'])
        
        # PQC metrics
        pqc_stats = round_stats.get('pqc_stats', {})
        if 'time' in pqc_stats:
            self.metrics['aggregation_time'].append(pqc_stats['time'])
        if 'num_valid' in pqc_stats:
            self.metrics['num_valid_updates'].append(pqc_stats['num_valid'])
        if 'num_invalid' in pqc_stats:
            self.metrics['num_invalid_updates'].append(pqc_stats['num_invalid'])
        
        # Aggregation/Defense metrics
        agg_stats = round_stats.get('aggregation_stats', {})
        if 'num_suspicious' in agg_stats:
            self.metrics['foolsgold_suspicious_clients'].append(agg_stats['num_suspicious'])
        
        # Store round data
        self.round_data.append({
            'round': round_num,
            'stats': round_stats
        })
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {}
        
        # Accuracy metrics
        if self.metrics['test_accuracy']:
            summary['final_accuracy'] = self.metrics['test_accuracy'][-1]
            summary['max_accuracy'] = max(self.metrics['test_accuracy'])
            summary['avg_accuracy'] = np.mean(self.metrics['test_accuracy'])
            summary['std_accuracy'] = np.std(self.metrics['test_accuracy'])
        
        # Loss metrics
        if self.metrics['training_loss']:
            summary['final_loss'] = self.metrics['training_loss'][-1]
            summary['min_loss'] = min(self.metrics['training_loss'])
            summary['avg_loss'] = np.mean(self.metrics['training_loss'])
        
        # Timing metrics
        if self.metrics['aggregation_time']:
            summary['avg_aggregation_time'] = np.mean(self.metrics['aggregation_time'])
            summary['total_aggregation_time'] = sum(self.metrics['aggregation_time'])
        
        # PQC metrics
        if self.metrics['encryption_time']:
            summary['avg_encryption_time'] = np.mean(self.metrics['encryption_time'])
        
        if self.metrics['decryption_time']:
            summary['avg_decryption_time'] = np.mean(self.metrics['decryption_time'])
        
        if self.metrics['signature_verification_time']:
            summary['avg_signature_time'] = np.mean(self.metrics['signature_verification_time'])
        
        # Update metrics
        if self.metrics['num_valid_updates']:
            summary['avg_valid_updates'] = np.mean(self.metrics['num_valid_updates'])
        
        if self.metrics['num_invalid_updates']:
            summary['total_invalid_updates'] = sum(self.metrics['num_invalid_updates'])
        
        # Defense metrics
        if self.metrics['foolsgold_suspicious_clients']:
            summary['avg_suspicious_clients'] = np.mean(
                self.metrics['foolsgold_suspicious_clients']
            )
        
        return summary
    
    def save_metrics(self, output_dir: str = RESULTS_DIR):
        """
        Save metrics to file
        
        Args:
            output_dir: Output directory
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (for easy reading)
        summary = self.get_summary()
        summary_file = Path(output_dir) / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed metrics as pickle
        metrics_file = Path(output_dir) / f"{self.experiment_name}_metrics.pkl"
        with open(metrics_file, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        print(f"Metrics saved to {output_dir}")
    
    def plot_results(self, output_dir: str = RESULTS_DIR):
        """
        Plot and save figures
        
        Args:
            output_dir: Output directory
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Accuracy plot
        if self.metrics['test_accuracy']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['test_accuracy'], marker='o')
            plt.xlabel('Round')
            plt.ylabel('Test Accuracy (%)')
            plt.title(f'{self.experiment_name} - Test Accuracy')
            plt.grid(True)
            plt.savefig(Path(output_dir) / f"{self.experiment_name}_accuracy.png")
            plt.close()
        
        # Loss plot
        if self.metrics['training_loss']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['training_loss'], marker='o')
            plt.xlabel('Round')
            plt.ylabel('Training Loss')
            plt.title(f'{self.experiment_name} - Training Loss')
            plt.grid(True)
            plt.savefig(Path(output_dir) / f"{self.experiment_name}_loss.png")
            plt.close()
        
        # Aggregation time plot
        if self.metrics['aggregation_time']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['aggregation_time'], marker='s')
            plt.xlabel('Round')
            plt.ylabel('Time (seconds)')
            plt.title(f'{self.experiment_name} - Aggregation Time')
            plt.grid(True)
            plt.savefig(Path(output_dir) / f"{self.experiment_name}_agg_time.png")
            plt.close()
        
        # Valid updates plot
        if self.metrics['num_valid_updates']:
            plt.figure(figsize=(10, 6))
            valid = self.metrics['num_valid_updates']
            invalid = self.metrics['num_invalid_updates']
            
            x = np.arange(len(valid))
            width = 0.35
            
            plt.bar(x - width/2, valid, width, label='Valid')
            plt.bar(x + width/2, invalid, width, label='Invalid')
            plt.xlabel('Round')
            plt.ylabel('Number of Updates')
            plt.title(f'{self.experiment_name} - Update Verification')
            plt.legend()
            plt.grid(True)
            plt.savefig(Path(output_dir) / f"{self.experiment_name}_updates.png")
            plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*60}")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:40s}: {value:.4f}")
            else:
                print(f"{key:40s}: {value}")
        print(f"{'='*60}\n")


class ComparisonAnalyzer:
    """Analyze and compare multiple experiments"""
    
    def __init__(self):
        self.experiments = {}
    
    def add_experiment(self, name: str, metrics: Dict):
        """Add experiment metrics"""
        self.experiments[name] = metrics
    
    def compare_accuracy(self):
        """Compare final accuracy across experiments"""
        results = {}
        for name, metrics in self.experiments.items():
            if 'test_accuracy' in metrics and metrics['test_accuracy']:
                results[name] = {
                    'final': metrics['test_accuracy'][-1],
                    'best': max(metrics['test_accuracy']),
                    'mean': np.mean(metrics['test_accuracy'])
                }
        return results
    
    def compare_time(self):
        """Compare aggregation time across experiments"""
        results = {}
        for name, metrics in self.experiments.items():
            if 'aggregation_time' in metrics and metrics['aggregation_time']:
                results[name] = {
                    'mean': np.mean(metrics['aggregation_time']),
                    'total': sum(metrics['aggregation_time']),
                    'std': np.std(metrics['aggregation_time'])
                }
        return results
    
    def plot_comparison(self, output_dir: str = RESULTS_DIR):
        """Plot comparison of experiments"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Accuracy comparison
        plt.figure(figsize=(12, 6))
        for name, metrics in self.experiments.items():
            if 'test_accuracy' in metrics and metrics['test_accuracy']:
                plt.plot(metrics['test_accuracy'], marker='o', label=name)
        
        plt.xlabel('Round')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Accuracy Comparison Across Experiments')
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(output_dir) / "comparison_accuracy.png")
        plt.close()
        
        # Loss comparison
        plt.figure(figsize=(12, 6))
        for name, metrics in self.experiments.items():
            if 'training_loss' in metrics and metrics['training_loss']:
                plt.plot(metrics['training_loss'], marker='s', label=name)
        
        plt.xlabel('Round')
        plt.ylabel('Training Loss')
        plt.title('Loss Comparison Across Experiments')
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(output_dir) / "comparison_loss.png")
        plt.close()
