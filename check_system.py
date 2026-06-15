"""
System Check and Validation Script
Verifies all dependencies and components before running experiments
"""

import sys
import platform
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("✓ Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ✗ Python 3.8+ required")
        return False
    print("  ✓ Python version OK")
    return True


def check_torch():
    """Check PyTorch installation"""
    print("\n✓ Checking PyTorch...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("  ✓ PyTorch OK")
        return True
    except ImportError:
        print("  ✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision")
        return False


def check_torchvision():
    """Check torchvision installation"""
    print("\n✓ Checking torchvision...")
    try:
        import torchvision
        print(f"  torchvision version: {torchvision.__version__}")
        print("  ✓ torchvision OK")
        return True
    except ImportError:
        print("  ✗ torchvision not installed")
        print("  Install with: pip install torchvision")
        return False


def check_numpy():
    """Check NumPy installation"""
    print("\n✓ Checking NumPy...")
    try:
        import numpy as np
        print(f"  NumPy version: {np.__version__}")
        print("  ✓ NumPy OK")
        return True
    except ImportError:
        print("  ✗ NumPy not installed")
        print("  Install with: pip install numpy")
        return False


def check_matplotlib():
    """Check Matplotlib installation"""
    print("\n✓ Checking Matplotlib...")
    try:
        import matplotlib
        print(f"  Matplotlib version: {matplotlib.__version__}")
        print("  ✓ Matplotlib OK")
        return True
    except ImportError:
        print("  ✗ Matplotlib not installed")
        print("  Install with: pip install matplotlib")
        return False


def check_scipy():
    """Check SciPy installation"""
    print("\n✓ Checking SciPy...")
    try:
        import scipy
        print(f"  SciPy version: {scipy.__version__}")
        print("  ✓ SciPy OK")
        return True
    except ImportError:
        print("  ✗ SciPy not installed")
        print("  Install with: pip install scipy")
        return False


def check_liboqs():
    """Check liboqs installation (optional)"""
    print("\n✓ Checking liboqs-python (optional)...")
    try:
        import oqs
        print("  ✓ liboqs-python installed")
        print("  Post-quantum cryptography ENABLED")
        return True
    except ImportError:
        print("  ! liboqs-python not installed (optional)")
        print("  For PQC support, install with: pip install liboqs-python")
        print("  Framework will use mock cryptography for testing")
        return False


def check_framework_files():
    """Check if all framework files exist"""
    print("\n✓ Checking framework files...")
    required_files = [
        'basecode.py',
        'config.py',
        'model.py',
        'dataset.py',
        'pqc.py',
        'attacks.py',
        'defenses.py',
        'federated_learning.py',
        'metrics.py',
        'experiments.py'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} NOT FOUND")
            all_exist = False
    
    return all_exist


def check_directories():
    """Check and create necessary directories"""
    print("\n✓ Checking directories...")
    dirs = ['results', 'logs', 'data']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✓ {dir_name}/")
    return True


def test_imports():
    """Test importing framework modules"""
    print("\n✓ Testing framework imports...")
    try:
        print("  Importing config...", end=" ")
        import config
        print("✓")
        
        print("  Importing model...", end=" ")
        from model import create_model
        print("✓")
        
        print("  Importing dataset...", end=" ")
        from dataset import get_cifar10_dataset
        print("✓")
        
        print("  Importing PQC...", end=" ")
        from pqc import PostQuantumCrypto
        print("✓")
        
        print("  Importing attacks...", end=" ")
        from attacks import AttackManager
        print("✓")
        
        print("  Importing defenses...", end=" ")
        from defenses import create_defense
        print("✓")
        
        print("  Importing federated learning...", end=" ")
        from federated_learning import FLClient, FLServer
        print("✓")
        
        print("  Importing experiments...", end=" ")
        from experiments import ExperimentRunner
        print("✓")
        
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"\n  ✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test creating a model"""
    print("\n✓ Testing model creation...")
    try:
        import torch
        from model import create_model
        from config import DEVICE
        
        print("  Creating CNN model...", end=" ")
        model = create_model(device=DEVICE)
        print("✓")
        
        print("  Testing forward pass...", end=" ")
        dummy_input = torch.randn(2, 3, 32, 32).to(DEVICE)
        output = model(dummy_input)
        print("✓")
        
        print(f"  Output shape: {output.shape}")
        print("  ✓ Model creation OK")
        return True
    except Exception as e:
        print(f"\n  ✗ Model creation failed: {e}")
        return False


def test_pqc():
    """Test PQC functionality"""
    print("\n✓ Testing PQC...")
    try:
        from pqc import PostQuantumCrypto, serialize_update, deserialize_update
        import torch
        
        print("  Creating PQC handler...", end=" ")
        pqc = PostQuantumCrypto()
        print("✓")
        
        print("  Generating KEM keypair...", end=" ")
        kem_pub, kem_sec = pqc.generate_kem_keypair()
        print("✓")
        
        print("  Generating signature keypair...", end=" ")
        sig_pub, sig_sec = pqc.generate_sig_keypair()
        print("✓")
        
        print("  Testing encapsulation...", end=" ")
        ct, ss = pqc.encapsulate(kem_pub)
        print("✓")
        
        print("  Testing decapsulation...", end=" ")
        ss2 = pqc.decapsulate(ct, kem_sec)
        print("✓")
        
        print("  Testing serialization...", end=" ")
        update = torch.randn(1000)
        serialized = serialize_update(update)
        deserialized = deserialize_update(serialized)
        print("✓")
        
        print("  ✓ PQC tests passed")
        return True
    except Exception as e:
        print(f"\n  ✗ PQC test failed: {e}")
        return False


def run_mini_experiment():
    """Run a mini experiment with 1 round and 2 clients"""
    print("\n✓ Running mini experiment (1 round, 2 clients)...")
    try:
        print("  Setting up mini configuration...")
        import config
        from model import create_model
        from dataset import get_cifar10_dataset
        from federated_learning import FLClient, FLServer, FederatedLearner
        from attacks import AttackManager
        
        # Minimal setup
        NUM_CLIENTS_MINI = 2
        NUM_ROUNDS_MINI = 1
        
        print("  Creating model...", end=" ")
        model = create_model(device=config.DEVICE)
        print("✓")
        
        print("  Loading dataset...", end=" ")
        dataset = get_cifar10_dataset()
        print("✓")
        
        print("  Creating clients...", end=" ")
        clients = []
        for cid in range(NUM_CLIENTS_MINI):
            client_model = create_model(device=config.DEVICE)
            client_model.load_state_dict(model.state_dict())
            client_dataset = dataset.get_client_dataloader(cid, batch_size=64, train=True)
            client = FLClient(cid, client_model, client_dataset, device=config.DEVICE)
            clients.append(client)
        print("✓")
        
        print("  Creating server...", end=" ")
        server = FLServer(model, defense_name='fedavg', device=config.DEVICE)
        print("✓")
        
        print("  Creating learner...", end=" ")
        learner = FederatedLearner(clients, server)
        print("✓")
        
        print("  Running 1 training round...", end=" ")
        test_loader = dataset.get_global_testloader(batch_size=64)
        stats = learner.perform_round(0, test_loader, apply_data_poisoning=False)
        print("✓")
        
        print(f"  Test Accuracy: {stats['test_accuracy']:.2f}%")
        print("  ✓ Mini experiment completed")
        return True
    except Exception as e:
        print(f"\n  ✗ Mini experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print summary of all checks"""
    print("\n" + "="*60)
    print("SYSTEM CHECK SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check:30s} : {status}")
    
    print("="*60)
    print(f"Total: {passed}/{total} checks passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ All checks passed! System is ready.")
        print("\nNext step: python basecode.py --experiment 1")
        return True
    else:
        print("\n✗ Some checks failed. Please install missing dependencies.")
        return False


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("Post-Quantum Secure Federated Learning Framework")
    print("System Check and Validation")
    print("="*60)
    
    results = {}
    
    # Basic checks
    results['Python Version'] = check_python_version()
    results['PyTorch'] = check_torch()
    results['torchvision'] = check_torchvision()
    results['NumPy'] = check_numpy()
    results['Matplotlib'] = check_matplotlib()
    results['SciPy'] = check_scipy()
    results['liboqs-python'] = check_liboqs()
    results['Framework Files'] = check_framework_files()
    results['Directories'] = check_directories()
    results['Module Imports'] = test_imports()
    results['Model Creation'] = test_model_creation()
    results['PQC Functions'] = test_pqc()
    
    # Optional: run mini experiment (takes ~1-2 minutes)
    print("\n" + "-"*60)
    print("Running mini experiment (this may take 1-2 minutes)...")
    print("-"*60)
    results['Mini Experiment'] = run_mini_experiment()
    
    # Summary
    all_pass = print_summary(results)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
