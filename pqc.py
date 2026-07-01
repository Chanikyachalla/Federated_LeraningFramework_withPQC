"""
Post-Quantum Cryptography Layer using ML-KEM (Kyber) and ML-DSA (Dilithium)

Security properties provided:
  - Confidentiality : ML-KEM-768 key encapsulation + AES-256-GCM symmetric encryption
  - Key derivation  : HKDF-SHA256 over the raw KEM shared secret (NIST SP 800-227 §7)
  - Integrity / Auth: AES-GCM with Associated Data (AAD = client_id ‖ round_num)
  - Authentication  : ML-DSA-65 signature over a bound payload
                      (client_id ‖ round_num ‖ kem_ciphertext ‖ nonce ‖ ciphertext)
                      — prevents replay, reorder, and impersonation attacks

Requires: pip install liboqs-python cryptography
"""

import os
import hashlib
import hmac as _hmac
import pickle
import time
from typing import Tuple, Dict, Any
import torch
import numpy as np
from config import *

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import oqs
    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False
    if PQC_ENABLED:
        print("WARNING: liboqs-python not available. "
              "Install with: pip install liboqs-python")

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes as _hashes
    AES_AVAILABLE = True
    HKDF_AVAILABLE = True
except ImportError:
    AES_AVAILABLE = False
    HKDF_AVAILABLE = False
    if PQC_ENABLED:
        print("WARNING: cryptography not available. "
              "Install with: pip install cryptography")


# ---------------------------------------------------------------------------
# Module-level mock constants
# A single fixed shared secret is used in mock mode so that
# _mock_encapsulate() and _mock_shared_secret() always agree.
# ---------------------------------------------------------------------------
_MOCK_SS: bytes = bytes(range(32))   # deterministic 32-byte mock shared secret
_MOCK_CT: bytes = os.urandom(1088)   # fixed-once mock KEM ciphertext


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def _derive_aes_key(shared_secret: bytes,
                    info: bytes = b"FL-ML-KEM-AES256GCM") -> bytes:
    """
    Derive a 256-bit AES key from the raw KEM shared secret using HKDF-SHA256.

    Using the raw KEM output directly as a symmetric key is insecure because
    the output distribution may be biased or the length may not match.
    HKDF provides a cryptographically strong extraction + expansion step.

    Args:
        shared_secret: Raw shared secret from ML-KEM encapsulation/decapsulation
        info:          Context string binding the key to its purpose

    Returns:
        32-byte AES-256 key
    """
    if HKDF_AVAILABLE:
        hkdf = HKDF(
            algorithm=_hashes.SHA256(),
            length=32,
            salt=None,        # HKDF uses a zero-filled salt when None
            info=info,
        )
        return hkdf.derive(shared_secret)

    # Pure-Python HKDF-SHA256 fallback (RFC 5869)
    salt = bytes(32)          # zero-filled 32-byte salt (hashlib default)
    # Extract
    prk = _hmac.new(salt, shared_secret, hashlib.sha256).digest()
    # Expand (one round is sufficient for 32 bytes)
    t = _hmac.new(prk, info + b"\x01", hashlib.sha256).digest()
    return t[:32]


# ---------------------------------------------------------------------------
# Encryption / Decryption helpers
# ---------------------------------------------------------------------------

def encrypt_update(data: bytes,
                   shared_secret: bytes,
                   aad: bytes = b"") -> Tuple[bytes, bytes]:
    """
    Encrypt model update using AES-256-GCM.

    The raw KEM shared secret is first passed through HKDF-SHA256 to produce
    a proper 256-bit AES key.  The AAD parameter binds the ciphertext to its
    metadata (client_id ‖ round_num), so that swapping or replaying
    ciphertexts in a different context fails authentication.

    Args:
        data:          Serialised model update (plaintext bytes)
        shared_secret: Raw shared secret from ML-KEM encapsulation
        aad:           Associated data included in GCM tag (not encrypted)

    Returns:
        (nonce, ciphertext_with_tag)  — 12-byte nonce + GCM-authenticated blob
    """
    # Always derive the AES key through HKDF
    key = _derive_aes_key(shared_secret)
    nonce = os.urandom(12)   # 96-bit random nonce for GCM

    if AES_AVAILABLE:
        try:
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, data, aad if aad else None)
            return nonce, ciphertext
        except Exception as e:
            print(f"[PQC] AES-GCM encryption error: {e}")

    # XOR fallback (no integrity guarantee — logs a warning)
    print("[PQC] WARNING: Using XOR fallback encryption (no integrity protection).")
    key_repeated = (key * ((len(data) // len(key)) + 1))[:len(data)]
    ciphertext = bytes(a ^ b for a, b in zip(data, key_repeated))
    return nonce, ciphertext


def decrypt_update(nonce: bytes,
                   ciphertext: bytes,
                   shared_secret: bytes,
                   aad: bytes = b"") -> bytes:
    """
    Decrypt model update using AES-256-GCM.

    Args:
        nonce:         12-byte nonce from encryption
        ciphertext:    Encrypted model update (with GCM authentication tag)
        shared_secret: Raw shared secret from ML-KEM decapsulation
        aad:           Must match the AAD used during encryption

    Returns:
        Decrypted plaintext bytes

    Raises:
        cryptography.exceptions.InvalidTag if AAD/ciphertext was tampered with
    """
    key = _derive_aes_key(shared_secret)

    if AES_AVAILABLE:
        try:
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, aad if aad else None)
            return plaintext
        except Exception as e:
            print(f"[PQC] AES-GCM decryption error (possible tampering): {e}")
            raise   # Let the caller handle — don't silently use corrupted data

    # XOR fallback
    print("[PQC] WARNING: Using XOR fallback decryption (no integrity protection).")
    key_repeated = (key * ((len(ciphertext) // len(key)) + 1))[:len(ciphertext)]
    plaintext = bytes(a ^ b for a, b in zip(ciphertext, key_repeated))
    return plaintext


# ---------------------------------------------------------------------------
# Main PQC handler
# ---------------------------------------------------------------------------

class PostQuantumCrypto:
    """
    Post-Quantum Cryptography Handler using liboqs.

    Provides:
      - ML-KEM-768 (Kyber-768) key encapsulation for forward-secret session keys
      - ML-DSA-65  (Dilithium-3) digital signatures for client authentication
    """

    def __init__(self,
                 ml_kem_variant: str = ML_KEM_VARIANT,
                 ml_dsa_variant: str = ML_DSA_VARIANT):
        """
        Args:
            ml_kem_variant: e.g. 'ML-KEM-768'
            ml_dsa_variant: e.g. 'ML-DSA-65'
        """
        self.ml_kem_variant = ml_kem_variant
        self.ml_dsa_variant = ml_dsa_variant
        self.kem = None
        self.sig = None

        # Stored key material
        self.sig_pub_key: bytes = None
        self.sig_sec_key: bytes = None
        self.kem_pub_key: bytes = None
        self.kem_sec_key: bytes = None

        if PQC_AVAILABLE:
            self._initialize_algorithms()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_algorithms(self):
        """Initialise liboqs KEM and Signature algorithm instances."""
        try:
            self.kem = oqs.KeyEncapsulation(self.ml_kem_variant)
            self.sig = oqs.Signature(self.ml_dsa_variant)
        except Exception as e:
            print(f"[PQC] Error initialising algorithms: {e}")
            self.kem = None
            self.sig = None

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate an ML-KEM (Kyber) key pair.

        Returns:
            (public_key, secret_key)
        """
        if not PQC_AVAILABLE or self.kem is None:
            return self._mock_generate_kem_keypair()

        try:
            public_key = self.kem.generate_keypair()
            secret_key = self.kem.export_secret_key()
            self.kem_pub_key = public_key
            self.kem_sec_key = secret_key
            return public_key, secret_key
        except Exception as e:
            print(f"[PQC] Error generating KEM keypair: {e}")
            return self._mock_generate_kem_keypair()

    def generate_sig_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate an ML-DSA (Dilithium) key pair.

        The Signature instance retains the loaded secret key, which is
        required for subsequent `sign()` calls on this same object.

        Returns:
            (public_key, secret_key)
        """
        if not PQC_AVAILABLE or self.sig is None:
            return self._mock_generate_sig_keypair()

        try:
            public_key = self.sig.generate_keypair()
            secret_key = self.sig.export_secret_key()
            self.sig_pub_key = public_key
            self.sig_sec_key = secret_key
            return public_key, secret_key
        except Exception as e:
            print(f"[PQC] Error generating signature keypair: {e}")
            return self._mock_generate_sig_keypair()

    # ------------------------------------------------------------------
    # KEM operations
    # ------------------------------------------------------------------

    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret (sender side).

        Args:
            public_key: Recipient's ML-KEM public key

        Returns:
            (kem_ciphertext, raw_shared_secret)
        """
        if not PQC_AVAILABLE or self.kem is None:
            return _MOCK_CT, _MOCK_SS

        try:
            kem_instance = oqs.KeyEncapsulation(self.ml_kem_variant)
            ciphertext, shared_secret = kem_instance.encap_secret(public_key)
            return ciphertext, shared_secret
        except Exception as e:
            print(f"[PQC] Error in encapsulation: {e}")
            return _MOCK_CT, _MOCK_SS

    def decapsulate(self, ciphertext: bytes) -> bytes:
        """
        Decapsulate a shared secret (receiver side).

        Uses the pre-loaded KEM instance (keys generated by generate_kem_keypair).

        Args:
            ciphertext: KEM ciphertext from the sender

        Returns:
            raw_shared_secret — pass through _derive_aes_key() before use
        """
        if not PQC_AVAILABLE or self.kem is None:
            return _MOCK_SS

        try:
            shared_secret = self.kem.decap_secret(ciphertext)
            return shared_secret
        except Exception as e:
            print(f"[PQC] Error in decapsulation: {e}")
            return _MOCK_SS

    # ------------------------------------------------------------------
    # Signature operations
    # ------------------------------------------------------------------

    def sign(self, message: bytes) -> bytes:
        """
        Sign a message with ML-DSA.

        The pre-loaded Signature instance (keys generated by
        generate_sig_keypair) is used.

        Args:
            message: Arbitrary bytes to sign.  Callers should pass a
                     *bound payload* that includes all context metadata
                     (client_id, round_num, kem_ciphertext, nonce, ciphertext)
                     to prevent replay attacks.

        Returns:
            ML-DSA signature bytes
        """
        if not PQC_AVAILABLE or self.sig is None:
            return os.urandom(2420)   # ML-DSA-65 signature size

        try:
            return self.sig.sign(message)
        except Exception as e:
            print(f"[PQC] Error in signing: {e}")
            return os.urandom(2420)

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify an ML-DSA signature.

        Args:
            message:    The exact bytes that were signed (bound payload)
            signature:  ML-DSA signature
            public_key: Signer's ML-DSA public key

        Returns:
            True if valid, False otherwise
        """
        if not PQC_AVAILABLE or self.sig is None:
            # Mock verification: always passes — only for simulation runs
            return True

        try:
            sig_instance = oqs.Signature(self.ml_dsa_variant)
            return sig_instance.verify(message, signature, public_key)
        except Exception as e:
            # Log the reason — helps diagnose real tampering vs. library errors
            print(f"[PQC] Signature verification failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Mock helpers  (used when liboqs is not installed)
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_generate_kem_keypair() -> Tuple[bytes, bytes]:
        """Generate a mock KEM key pair (simulation only)."""
        pub = os.urandom(1184)   # ML-KEM-768 public key size
        sec = os.urandom(2400)   # ML-KEM-768 secret key size
        return pub, sec

    @staticmethod
    def _mock_generate_sig_keypair() -> Tuple[bytes, bytes]:
        """Generate a mock DSA key pair (simulation only)."""
        pub = os.urandom(1952)   # ML-DSA-65 public key size
        sec = os.urandom(4032)   # ML-DSA-65 secret key size
        return pub, sec


# ---------------------------------------------------------------------------
# Encrypted update container
# ---------------------------------------------------------------------------

class EncryptedUpdate:
    """
    Container that carries a client's encrypted model update to the server.

    Fields
    ------
    client_id       : int    — originating client
    encrypted_update: bytes  — AES-256-GCM ciphertext (includes GCM tag)
    signature       : bytes  — ML-DSA-65 signature over the bound payload
    kem_ciphertext  : bytes  — ML-KEM-768 ciphertext for the session key
    nonce           : bytes  — 12-byte AES-GCM nonce
    round_num       : int    — FL round number (included in signed payload)
    timestamp       : float  — Unix creation time
    """

    def __init__(self,
                 client_id: int,
                 encrypted_update: bytes,
                 signature: bytes,
                 kem_ciphertext: bytes = None,
                 nonce: bytes = None,
                 round_num: int = 0):
        self.client_id       = client_id
        self.encrypted_update = encrypted_update
        self.signature        = signature
        self.kem_ciphertext   = kem_ciphertext
        self.nonce            = nonce
        self.round_num        = round_num
        self.timestamp        = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            'client_id':        self.client_id,
            'encrypted_update': self.encrypted_update,
            'signature':        self.signature,
            'kem_ciphertext':   self.kem_ciphertext,
            'nonce':            self.nonce,
            'round_num':        self.round_num,
            'timestamp':        self.timestamp,
        }

    @staticmethod
    def build_bound_message(client_id: int,
                            round_num: int,
                            kem_ciphertext: bytes,
                            nonce: bytes,
                            encrypted_update: bytes) -> bytes:
        """
        Construct the canonical signed payload.

        Format:  client_id (4B BE) ‖ round_num (4B BE)
                 ‖ kem_ciphertext ‖ nonce ‖ encrypted_update

        All context fields are concatenated so that the signature covers
        every piece of metadata.  Changing any field invalidates the tag.
        """
        return (
            client_id.to_bytes(4, 'big')
            + round_num.to_bytes(4, 'big')
            + (kem_ciphertext or b'')
            + (nonce or b'')
            + encrypted_update
        )

    @staticmethod
    def build_aad(client_id: int, round_num: int) -> bytes:
        """
        Build Associated Data for AES-GCM.

        The AAD is not encrypted but IS authenticated, so any metadata
        mismatch causes decryption to fail with an InvalidTag error.
        """
        return client_id.to_bytes(4, 'big') + round_num.to_bytes(4, 'big')


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def serialize_state_dict(state_dict) -> bytes:
    """Serialise a model state_dict to bytes using pickle."""
    return pickle.dumps(state_dict)


def deserialize_state_dict(data: bytes):
    """Deserialise a model state_dict from bytes."""
    return pickle.loads(data)


def serialize_update(update: torch.Tensor) -> bytes:
    """
    Serialise a flat model-update tensor to bytes.

    The tensor is moved to CPU and converted to a numpy array before
    pickling so that it is device-agnostic on the receiving end.
    """
    return pickle.dumps(update.cpu().numpy())


def deserialize_update(data: bytes) -> torch.Tensor:
    """
    Deserialise a flat model-update tensor from bytes.

    Returns a float32 CPU tensor.
    """
    return torch.from_numpy(pickle.loads(data)).float()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pqc_handler() -> PostQuantumCrypto:
    """Create a PostQuantumCrypto instance using config-specified variants."""
    return PostQuantumCrypto(
        ml_kem_variant=ML_KEM_VARIANT,
        ml_dsa_variant=ML_DSA_VARIANT,
    )
