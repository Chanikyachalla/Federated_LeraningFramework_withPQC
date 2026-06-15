# Quick Start Guide

## 1. Installation (5 minutes)

### Step 1: Install Python Dependencies
```bash
pip install torch torchvision numpy matplotlib scipy
```

### Step 2: Install Post-Quantum Crypto (Optional)
```bash
pip install liboqs-python
```

If installation fails, the framework will still work with mock cryptography for testing.

### Step 3: Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

---

## 2. Run Your First Experiment (10 minutes)

### Start with the Baseline Experiment
```bash
python basecode.py --experiment 1
```

This runs **Clean FL + FedAvg** - a baseline federated learning setup with no attacks.

**Expected Output:**
```
==========================================================
Initializing Experiment: Exp1: Clean FL + FedAvg
==========================================================
Loading CIFAR-10 dataset...
Creating 10 clients...

Training Configuration:
  Rounds: 50
  Local Epochs: 4
  Batch Size: 128
  Byzantine Attack: False
  Data Poisoning: False
  PQC Enabled: False
  Defense: fedavg

Starting training...
Round 1/50 | Loss: 2.3021 | Train Loss: 2.3015 | Accuracy: 10.23%
Round 2/50 | Loss: 2.2876 | Train Loss: 2.2814 | Accuracy: 11.45%
...
```

---

## 3. Understand the Experiments

### The 7 Experiments

| # | Name | What It Tests | Use Case |
|---|------|---------------|----------|
| 1 | Clean FL + FedAvg | Baseline performance | Sanity check |
| 2 | Byzantine Attack + FedAvg | Vulnerability to attacks | Understand impact |
| 3 | Byzantine Attack + FoolsGold | Defense effectiveness | Compare defenses |
| 4 | Data Poisoning + FedAvg | Label flipping attacks | Data integrity |
| 5 | Data Poisoning + FoolsGold | Defense against poisoning | Robustness |
| 6 | Byzantine + PQC + FedAvg | Secure communication | Post-quantum security |
| 7 | Byzantine + PQC + FoolsGold | Full security | Complete solution |

### Run All Experiments
```bash
python basecode.py --all
```

This will take ~30-60 minutes depending on your hardware.

---

## 4. Interpret Results

### Output Files
After each run, check the `results/` directory:

```
results/
├── Exp1_Clean_FL_summary.json      # Summary statistics
├── Exp1_Clean_FL_metrics.pkl       # Detailed metrics
├── Exp1_Clean_FL_accuracy.png      # Accuracy plot
├── Exp1_Clean_FL_loss.png          # Loss plot
└── ... (more files for each experiment)
```

### Example Summary (JSON)
```json
{
  "final_accuracy": 75.43,
  "max_accuracy": 76.21,
  "avg_accuracy": 71.23,
  "final_loss": 0.8934,
  "min_loss": 0.7234,
  "avg_loss": 0.9123,
  "avg_aggregation_time": 0.0234,
  "avg_valid_updates": 9.8,
  "total_invalid_updates": 2
}
```

### Key Metrics to Compare

1. **Final Accuracy** - Higher is better
   - Clean FL: ~75-85%
   - Byzantine Attack: ~40-60% (degraded)
   - With FoolsGold: ~70-80% (recovered)

2. **Aggregation Time** - Lower is better
   - FedAvg: Very fast
   - FoolsGold: Slightly slower due to similarity computation
   - With PQC: Additional encryption/decryption overhead

3. **Valid Updates** - Higher is better
   - Should be close to 10 (number of clients)

---

## 5. Modify Configuration

### Quick Configuration Changes

Edit `config.py` to customize:

#### Test on Fewer Rounds
```python
NUM_ROUNDS = 5  # Instead of 50
```

#### Use More Byzantine Clients
```python
BYZANTINE_CLIENTS = 3  # Instead of 1
```

#### Increase Poisoning Intensity
```python
POISON_RATIO = 0.5  # Instead of 0.3
BYZANTINE_SCALE = 5.0  # Instead of 3.0
```

#### Enable/Disable PQC
```python
PQC_ENABLED = False  # Faster but not secure
```

#### Change Defense
```python
DEFENSE_METHOD = 'foolsgold'  # or 'fedavg'
```

---

## 6. Common Issues & Solutions

### Issue: CIFAR-10 not downloading
```
Error: Failed to download CIFAR-10
```
**Solution:**
```bash
mkdir -p ./data
# Dataset auto-downloads on first run
```

### Issue: Out of memory error
```
RuntimeError: CUDA out of memory
```
**Solutions:**
1. Use CPU: Set `DEVICE = 'cpu'` in config.py
2. Reduce batch size: `BATCH_SIZE = 64`
3. Reduce clients: `NUM_CLIENTS = 5`

### Issue: liboqs not installed
```
ImportError: No module named 'oqs'
```
**Solution:** Set `PQC_ENABLED = False` in config.py

---

## 7. Experiment Workflow

### Recommended Workflow

1. **Start Simple**
   ```bash
   # Quick test with 5 rounds
   python basecode.py --experiment 1
   ```

2. **Compare Defenses**
   ```bash
   python basecode.py --experiment 2  # FedAvg
   python basecode.py --experiment 3  # FoolsGold
   ```

3. **Test Security**
   ```bash
   python basecode.py --experiment 6  # With PQC
   ```

4. **Full Benchmark**
   ```bash
   python basecode.py --all
   ```

---

## 8. Understanding the Architecture

### Federated Learning Loop
```
Round 1:
  1. Server sends model to all clients
  2. Each client trains locally for 4 epochs
  3. Compute update = local_model - global_model
  4. Apply attack (if enabled)
  5. Sign with Dilithium
  6. Encrypt with Kyber
  7. Send to server
  8. Server verifies and decrypts
  9. Aggregate using FedAvg or FoolsGold
  10. Update global model
  11. Repeat for next round
```

### Attack Types

**Byzantine Model Poisoning:**
- Malicious client sends: `poisoned_update = -SCALE * local_update`
- Effect: Pulls model away from correct weights

**Data Poisoning:**
- Malicious client flips labels: Cat → Dog
- Effect: Local model learns incorrect patterns

### Defense Types

**FedAvg (Baseline):**
- Simply averages all updates
- No defense against poisoning
- Fast but vulnerable

**FoolsGold:**
- Computes similarity between updates
- Identifies colluding clients
- Reduces weight of suspicious updates
- Slower but more robust

---

## 9. Next Steps

After understanding the framework:

1. **Extend Defenses**: Implement Trust Score or Secure Aggregation
2. **Modify Architecture**: Change CNN to ResNet or other models
3. **Different Dataset**: Use MNIST, Fashion-MNIST, or your own
4. **Advanced Attacks**: Add gradient inversion or model extraction attacks
5. **Real Deployment**: Deploy on actual distributed system

---

## 10. Support Resources

### File Organization
```
btp/
├── basecode.py          ← Start here
├── config.py            ← Modify parameters
├── model.py             ← Change architecture
├── dataset.py           ← New datasets
├── attacks.py           ← Add attacks
├── defenses.py          ← Add defenses
├── federated_learning.py ← Core FL logic
├── experiments.py       ← Experiment setup
└── metrics.py           ← Metrics tracking
```

### Key Concepts

1. **FedAvg**: Average all updates → global update
2. **Byzantine**: Attacker sends reversed updates
3. **FoolsGold**: Detect similar (colluding) updates
4. **ML-KEM**: Encrypt model updates
5. **ML-DSA**: Sign updates with digital signature

### Python Tips
```python
# Monitor progress
print(f"Round {round_num}: Accuracy = {accuracy:.2f}%")

# Save custom metrics
import json
with open('results/custom.json', 'w') as f:
    json.dump(metrics, f)

# Plot results
import matplotlib.pyplot as plt
plt.plot(accuracies)
plt.show()
```

---

## Summary

You now have a complete Post-Quantum Secure Federated Learning Framework!

**Next Action:** Run `python basecode.py --experiment 1` to start.

**Expected Time:**
- Single experiment: 10-15 minutes
- All 7 experiments: 30-60 minutes

**Success Indicator:** See accuracy plots in `results/` folder.

Good luck! 🚀
