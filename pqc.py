"""
Post-Quantum Cryptography Layer using ML-KEM (Kyber) and ML-DSA (Dilithium)
Requires: pip install liboqs-python
"""

import pickle
import time
from typing import Tuple, Dict, Any
import torch
import numpy as np
from config import *

try:
    import oqs
    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False
    if PQC_ENABLED:
        print("WARNING: liboqs-python not available. Install with: pip install liboqs-python")


class PostQuantumCrypto:
    """
    Post-Quantum Cryptography Handler using liboqs
    Supports ML-KEM-768 (Kyber) and ML-DSA-65 (Dilithium)
    """
    
    def __init__(self, ml_kem_variant=ML_KEM_VARIANT, ml_dsa_variant=ML_DSA_VARIANT):
        """
        Initialize PQC handler
        
        Args:
            ml_kem_variant: ML-KEM variant (e.g., 'ML-KEM-768')
            ml_dsa_variant: ML-DSA variant (e.g., 'ML-DSA-65')
        """
        self.ml_kem_variant = ml_kem_variant
        self.ml_dsa_variant = ml_dsa_variant
        self.kem = None
        self.sig = None
        
        if PQC_AVAILABLE:
            self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize KEM and Signature algorithms"""
        try:
            self.kem = oqs.KeyEncapsulation(self.ml_kem_variant)
            self.sig = oqs.Signature(self.ml_dsa_variant)
        except Exception as e:
            print(f"Error initializing PQC algorithms: {e}")
            self.kem = None
            self.sig = None
    
    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate ML-KEM (Kyber) keypair
        
        Returns:
            (public_key, secret_key)
        """
        if not PQC_AVAILABLE or self.kem is None:
            return self._mock_generate_keypair()
        
        try:
            public_key = self.kem.generate_keys()
            secret_key = self.kem.export_secret_key()
            return public_key, secret_key
        except Exception as e:
            print(f"Error generating KEM keypair: {e}")
            return self._mock_generate_keypair()
    
    def generate_sig_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate ML-DSA (Dilithium) keypair
        
        Returns:
            (public_key, secret_key)
        """
        if not PQC_AVAILABLE or self.sig is None:
            return self._mock_generate_keypair()
        
        try:
            public_key = self.sig.generate_keys()
            secret_key = self.sig.export_secret_key()
            return public_key, secret_key
        except Exception as e:
            print(f"Error generating signature keypair: {e}")
            return self._mock_generate_keypair()
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate shared secret (KEM)
        
        Args:
            public_key: Recipient's ML-KEM public key
        
        Returns:
            (ciphertext, shared_secret)
        """
        if not PQC_AVAILABLE or self.kem is None:
            return self._mock_encapsulate()
        
        try:
            kem_instance = oqs.KeyEncapsulation(self.ml_kem_variant)
            ciphertext, shared_secret = kem_instance.encap_secret(public_key)
            return ciphertext, shared_secret
        except Exception as e:
            print(f"Error in encapsulation: {e}")
            return self._mock_encapsulate()
    
    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decapsulate shared secret (KEM)
        
        Args:
            ciphertext: Ciphertext from sender
            secret_key: Recipient's ML-KEM secret key
        
        Returns:
            shared_secret
        """
        if not PQC_AVAILABLE or self.kem is None:
            return self._mock_shared_secret()
        
        try:
            kem_instance = oqs.KeyEncapsulation(self.ml_kem_variant)
            kem_instance.import_secret_key(secret_key)
            shared_secret = kem_instance.decap_secret(ciphertext)
            return shared_secret
        except Exception as e:
            print(f"Error in decapsulation: {e}")
            return self._mock_shared_secret()
    
    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        """
        Sign a message (ML-DSA)
        
        Args:
            message: Message to sign
            secret_key: ML-DSA secret key
        
        Returns:
            Signature
        """
        if not PQC_AVAILABLE or self.sig is None:
            return self._mock_signature()
        
        try:
            sig_instance = oqs.Signature(self.ml_dsa_variant)
            sig_instance.import_secret_key(secret_key)
            signature = sig_instance.sign(message)
            return signature
        except Exception as e:
            print(f"Error in signing: {e}")
            return self._mock_signature()
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify a signature (ML-DSA)
        
        Args:
            message: Original message
            signature: Signature to verify
            public_key: ML-DSA public key
        
        Returns:
            True if valid, False otherwise
        """
        if not PQC_AVAILABLE or self.sig is None:
            return True  # Mock verification always passes
        
        try:
            sig_instance = oqs.Signature(self.ml_dsa_variant)
            sig_instance.import_public_key(public_key)
            sig_instance.verify(message, signature)
            return True
        except Exception:
            return False
    
    # Mock functions for when liboqs is not available
    @staticmethod
    def _mock_generate_keypair() -> Tuple[bytes, bytes]:
        """Generate mock keypair"""
        pub = np.random.bytes(1024)
        sec = np.random.bytes(2048)
        return pub, sec
    
    @staticmethod
    def _mock_encapsulate() -> Tuple[bytes, bytes]:
        """Generate mock ciphertext and shared secret"""
        ct = np.random.bytes(1088)
        ss = np.random.bytes(32)
        return ct, ss
    
    @staticmethod
    def _mock_shared_secret() -> bytes:
        """Generate mock shared secret"""
        return np.random.bytes(32)
    
    @staticmethod
    def _mock_signature() -> bytes:
        """Generate mock signature"""
        return np.random.bytes(2420)


class EncryptedUpdate:
    """Container for encrypted model updates with signatures"""
    
    def __init__(self, client_id: int, ciphertext: bytes, signature: bytes, 
                 kem_ciphertext: bytes = None):
        """
        Args:
            client_id: Client ID
            ciphertext: Encrypted update data
            signature: ML-DSA signature
            kem_ciphertext: ML-KEM ciphertext for key encapsulation
        """
        self.client_id = client_id
        self.ciphertext = ciphertext
        self.signature = signature
        self.kem_ciphertext = kem_ciphertext
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'client_id': self.client_id,
            'ciphertext': self.ciphertext,
            'signature': self.signature,
            'kem_ciphertext': self.kem_ciphertext,
            'timestamp': self.timestamp
        }


def serialize_update(update: torch.Tensor) -> bytes:
    """
    Serialize model update to bytes
    
    Args:
        update: Model update tensor
    
    Returns:
        Serialized bytes
    """
    return pickle.dumps(update.cpu().numpy())


def deserialize_update(data: bytes) -> torch.Tensor:
    """
    Deserialize model update from bytes
    
    Args:
        data: Serialized bytes
    
    Returns:
        Model update tensor
    """
    return torch.from_numpy(pickle.loads(data)).float()


def create_pqc_handler():
    """Create PQC handler instance"""
    return PostQuantumCrypto(
        ml_kem_variant=ML_KEM_VARIANT,
        ml_dsa_variant=ML_DSA_VARIANT
    )
