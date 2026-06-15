"""
Federated Learning Framework: Server, Client, and FedAvg Algorithm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from model import create_model, CIFAR10CNN
from defenses import create_defense
from pqc import PostQuantumCrypto, serialize_update, deserialize_update, EncryptedUpdate
from attacks import AttackManager
from config import *


class FLClient:
    """
    Federated Learning Client
    Responsible for local training
    """
    
    def __init__(self, client_id: int, model: CIFAR10CNN, 
                 dataset, attack_manager: Optional[AttackManager] = None,
                 device: str = DEVICE):
        """
        Initialize FL client
        
        Args:
            client_id: Client ID
            model: Model instance
            dataset: Client's dataset
            attack_manager: Attack manager instance
            device: Device to use ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.attack_manager = attack_manager
        self.device = device
        
        # PQC keys
        self.pqc = PostQuantumCrypto() if PQC_ENABLED else None
        if self.pqc and ENCRYPT_MESSAGE:
            self.sig_pub_key, self.sig_sec_key = self.pqc.generate_sig_keypair()
        else:
            self.sig_pub_key, self.sig_sec_key = None, None
        
        self.metrics = {
            'local_loss': [],
            'training_time': [],
            'encryption_time': [],
            'signature_time': []
        }
    
    def local_train(self, num_epochs: int = LOCAL_EPOCHS, 
                   learning_rate: float = LEARNING_RATE,
                   apply_data_poisoning: bool = False) -> Tuple[CIFAR10CNN, float]:
        """
        Perform local training
        
        Args:
            num_epochs: Number of local epochs
            learning_rate: Learning rate
            apply_data_poisoning: Whether to apply data poisoning
        
        Returns:
            (trained_model, average_loss)
        """
        start_time = time.time()
        
        # Create a copy of the model for local training
        local_model = create_model(device=self.device)
        local_model.load_state_dict(self.model.state_dict())
        
        # Define optimizer and loss
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            local_model.train()
            epoch_loss = 0.0
            
            for images, labels in self.dataset:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Apply data poisoning if enabled
                if apply_data_poisoning and self.attack_manager and \
                   self.attack_manager.is_data_poisoning_client(self.client_id):
                    images, labels = self.attack_manager.apply_data_poisoning(images, labels)
                
                # Forward pass
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        training_time = time.time() - start_time
        average_loss = total_loss / (num_epochs * max(num_batches, 1))
        
        self.metrics['local_loss'].append(average_loss)
        self.metrics['training_time'].append(training_time)
        
        return local_model, average_loss
    
    def compute_update(self, local_model: CIFAR10CNN) -> torch.Tensor:
        """
        Compute model update (delta = local - global)
        
        Args:
            local_model: Trained local model
        
        Returns:
            Model update tensor
        """
        global_weights = self.model.get_flat_weights()
        local_weights = local_model.get_flat_weights()
        update = local_weights - global_weights
        return update
    
    def apply_attack(self, update: torch.Tensor) -> torch.Tensor:
        """
        Apply Byzantine attack if this is a Byzantine client
        
        Args:
            update: Model update
        
        Returns:
            Possibly poisoned update
        """
        if self.attack_manager and self.attack_manager.is_byzantine_client(self.client_id):
            return self.attack_manager.apply_model_poisoning(self.client_id, update)
        return update
    
    def encrypt_and_sign_update(self, update: torch.Tensor, 
                               server_kem_pub_key: bytes = None) -> EncryptedUpdate:
        """
        Encrypt and sign model update
        
        Args:
            update: Model update
            server_kem_pub_key: Server's ML-KEM public key
        
        Returns:
            EncryptedUpdate object
        """
        start_time = time.time()
        
        # Serialize update
        update_bytes = serialize_update(update)
        
        signature = None
        kem_ciphertext = None
        ciphertext = update_bytes
        
        if self.pqc and SIGN_MESSAGE:
            # Sign the update
            sig_start = time.time()
            signature = self.pqc.sign(update_bytes, self.sig_sec_key)
            sig_time = time.time() - sig_start
            self.metrics['signature_time'].append(sig_time)
        
        if self.pqc and ENCRYPT_MESSAGE and server_kem_pub_key:
            # Encrypt using ML-KEM
            enc_start = time.time()
            kem_ciphertext, shared_secret = self.pqc.encapsulate(server_kem_pub_key)
            ciphertext = self._xor_encrypt(update_bytes, shared_secret)
            enc_time = time.time() - enc_start
            self.metrics['encryption_time'].append(enc_time)
        
        total_time = time.time() - start_time
        
        return EncryptedUpdate(
            client_id=self.client_id,
            ciphertext=ciphertext,
            signature=signature,
            kem_ciphertext=kem_ciphertext
        )
    
    @staticmethod
    def _xor_encrypt(data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption using shared secret"""
        key_repeated = (key * ((len(data) // len(key)) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_repeated))
    
    def get_public_key(self) -> bytes:
        """Get client's signature public key"""
        return self.sig_pub_key


class FLServer:
    """
    Federated Learning Server
    Responsible for model aggregation and distribution
    """
    
    def __init__(self, model: CIFAR10CNN, 
                 defense_name: str = DEFENSE_METHOD,
                 device: str = DEVICE):
        """
        Initialize FL server
        
        Args:
            model: Global model
            defense_name: Defense mechanism ('fedavg' or 'foolsgold')
            device: Device to use
        """
        self.model = model
        self.device = device
        self.defense = create_defense(defense_name)
        self.defense_name = defense_name
        
        # PQC for server
        self.pqc = PostQuantumCrypto() if PQC_ENABLED else None
        if self.pqc and ENCRYPT_MESSAGE:
            self.kem_pub_key, self.kem_sec_key = self.pqc.generate_kem_keypair()
        else:
            self.kem_pub_key, self.kem_sec_key = None, None
        
        self.client_public_keys = {}
        self.metrics = {
            'aggregation_time': [],
            'decryption_time': [],
            'verification_time': [],
            'num_valid_updates': [],
            'num_invalid_signatures': []
        }
    
    def register_client(self, client_id: int, public_key: bytes):
        """Register client's public key"""
        self.client_public_keys[client_id] = public_key
    
    def get_kem_public_key(self) -> bytes:
        """Get server's ML-KEM public key for clients"""
        return self.kem_pub_key
    
    def verify_and_decrypt_updates(self, encrypted_updates: List[EncryptedUpdate]) \
            -> Tuple[List[torch.Tensor], List[int], Dict]:
        """
        Verify signatures and decrypt updates
        
        Args:
            encrypted_updates: List of EncryptedUpdate objects
        
        Returns:
            (decrypted_updates, valid_client_ids, stats)
        """
        start_time = time.time()
        
        decrypted_updates = []
        valid_client_ids = []
        num_valid = 0
        num_invalid = 0
        
        for enc_update in encrypted_updates:
            client_id = enc_update.client_id
            ciphertext = enc_update.ciphertext
            signature = enc_update.signature
            kem_ciphertext = enc_update.kem_ciphertext
            
            # Decrypt
            decrypted_data = ciphertext
            if self.pqc and ENCRYPT_MESSAGE and kem_ciphertext:
                dec_start = time.time()
                shared_secret = self.pqc.decapsulate(kem_ciphertext, self.kem_sec_key)
                decrypted_data = self._xor_decrypt(ciphertext, shared_secret)
                dec_time = time.time() - dec_start
                self.metrics['decryption_time'].append(dec_time)
            
            # Verify signature
            is_valid = True
            if self.pqc and SIGN_MESSAGE and signature:
                ver_start = time.time()
                client_pub_key = self.client_public_keys.get(client_id)
                if client_pub_key:
                    is_valid = self.pqc.verify(decrypted_data, signature, client_pub_key)
                ver_time = time.time() - ver_start
                self.metrics['verification_time'].append(ver_time)
            
            if is_valid:
                # Deserialize update
                try:
                    update = deserialize_update(decrypted_data)
                    decrypted_updates.append(update)
                    valid_client_ids.append(client_id)
                    num_valid += 1
                except Exception as e:
                    num_invalid += 1
            else:
                num_invalid += 1
        
        total_time = time.time() - start_time
        self.metrics['aggregation_time'].append(total_time)
        self.metrics['num_valid_updates'].append(num_valid)
        self.metrics['num_invalid_signatures'].append(num_invalid)
        
        stats = {
            'time': total_time,
            'num_valid': num_valid,
            'num_invalid': num_invalid,
            'valid_client_ids': valid_client_ids
        }
        
        return decrypted_updates, valid_client_ids, stats
    
    def aggregate_updates(self, updates: List[torch.Tensor],
                         valid_client_ids: List[int]) -> Tuple[torch.Tensor, Dict]:
        """
        Aggregate valid updates
        
        Args:
            updates: List of model updates
            valid_client_ids: IDs of clients with valid updates
        
        Returns:
            (aggregated_update, aggregation_stats)
        """
        if not updates:
            return None, {}
        
        if self.defense_name == 'foolsgold':
            aggregated, stats = self.defense.aggregate(updates, valid_client_ids)
        else:
            aggregated = self.defense.aggregate(updates)
            stats = {}
        
        return aggregated, stats
    
    def update_global_model(self, aggregated_update: torch.Tensor):
        """Update global model with aggregated update"""
        if aggregated_update is None:
            return
        
        current_weights = self.model.get_flat_weights()
        new_weights = current_weights + aggregated_update
        self.model.set_flat_weights(new_weights)
    
    @staticmethod
    def _xor_decrypt(data: bytes, key: bytes) -> bytes:
        """Simple XOR decryption using shared secret"""
        key_repeated = (key * ((len(data) // len(key)) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_repeated))


class FederatedLearner:
    """
    High-level Federated Learning orchestrator
    Manages the entire FL process
    """
    
    def __init__(self, clients: List[FLClient], server: FLServer):
        """
        Initialize federated learner
        
        Args:
            clients: List of FL clients
            server: FL server
        """
        self.clients = clients
        self.server = server
        self.num_clients = len(clients)
        self.history = {
            'loss': [],
            'accuracy': [],
            'attack_success_rate': []
        }
    
    def perform_round(self, round_num: int, test_loader = None,
                     apply_data_poisoning: bool = False,
                     apply_model_poisoning: bool = False) -> Dict:
        """
        Perform one round of federated learning
        
        Args:
            round_num: Round number
            test_loader: Test dataloader for evaluation
            apply_data_poisoning: Whether to apply data poisoning
            apply_model_poisoning: Whether to apply model poisoning
        
        Returns:
            Round statistics
        """
        round_stats = {
            'round': round_num,
            'client_losses': [],
            'test_loss': 0.0,
            'test_accuracy': 0.0,
            'aggregation_stats': {},
            'pqc_stats': {}
        }
        
        # Step 1: Distribute global model to clients
        for client in self.clients:
            client.model.load_state_dict(self.server.model.state_dict())
        
        # Step 2: Local training
        encrypted_updates = []
        for client in self.clients:
            # Local training
            local_model, loss = client.local_train(
                apply_data_poisoning=apply_data_poisoning
            )
            round_stats['client_losses'].append(loss)
            
            # Compute update
            update = client.compute_update(local_model)
            
            # Apply Byzantine attack
            if apply_model_poisoning:
                update = client.apply_attack(update)
            
            # Encrypt and sign
            enc_update = client.encrypt_and_sign_update(
                update, 
                self.server.get_kem_public_key()
            )
            encrypted_updates.append(enc_update)
        
        # Step 3: Verify and decrypt at server
        updates, valid_client_ids, pqc_stats = self.server.verify_and_decrypt_updates(
            encrypted_updates
        )
        round_stats['pqc_stats'] = pqc_stats
        
        # Step 4: Aggregate
        aggregated_update, agg_stats = self.server.aggregate_updates(updates, valid_client_ids)
        round_stats['aggregation_stats'] = agg_stats
        
        # Step 5: Update global model
        self.server.update_global_model(aggregated_update)
        
        # Step 6: Evaluate on test set
        if test_loader:
            test_loss, test_accuracy = self.evaluate(test_loader)
            round_stats['test_loss'] = test_loss
            round_stats['test_accuracy'] = test_accuracy
            self.history['accuracy'].append(test_accuracy)
        
        return round_stats
    
    def evaluate(self, test_loader) -> Tuple[float, float]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test dataloader
        
        Returns:
            (test_loss, accuracy)
        """
        self.server.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = self.server.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        return avg_loss, accuracy
