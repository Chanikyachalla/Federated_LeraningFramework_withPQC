"""
quick_test.py — Scaled-down sanity check for the FL framework.

Overrides:
  - 3 clients, 5 rounds, 1 local epoch
  - 500 training samples total (50 per class subset)
  - Smaller 2-block CNN (faster forward pass)
  - No PQC encryption (mock mode only)
  - No attacks

Expected output:
  - Accuracy should climb above 25% (random=10%) by round 5
  - Update norm should be clearly non-zero every round
  - Runs in ~1-3 minutes on CPU
"""

import copy
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# ------------------------------------------------------------------ #
#  Tiny inline config — completely bypasses config.py                  #
# ------------------------------------------------------------------ #
CFG = dict(
    NUM_CLIENTS      = 3,
    NUM_ROUNDS       = 5,
    LOCAL_EPOCHS     = 1,
    BATCH_SIZE       = 32,
    LEARNING_RATE    = 0.01,
    SERVER_LR        = 1.0,
    WEIGHT_DECAY     = 1e-4,
    GRAD_CLIP        = 5.0,
    SAMPLES_PER_CLASS= 50,    # 50 * 10 classes = 500 train samples total
    NUM_CLASSES      = 10,
    DEVICE           = 'cpu',
)

# ------------------------------------------------------------------ #
#  Tiny 2-block CNN (no residual) — fast on CPU                       #
# ------------------------------------------------------------------ #
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

    def get_state_dict_flat(self):
        """Flatten full state_dict (params + BN buffers) to 1-D tensor."""
        return torch.cat([t.float().view(-1) for t in self.state_dict().values()])

    def set_state_dict_flat(self, flat):
        """Restore full state_dict from a 1-D tensor."""
        new_sd, offset = {}, 0
        for k, t in self.state_dict().items():
            n = t.numel()
            new_sd[k] = flat[offset:offset+n].view(t.shape).to(t.dtype)
            offset += n
        self.load_state_dict(new_sd)


# ------------------------------------------------------------------ #
#  Dataset helpers                                                     #
# ------------------------------------------------------------------ #
def get_balanced_subset(dataset, samples_per_class):
    """Return indices so each class has exactly `samples_per_class` samples."""
    targets = np.array(dataset.targets)
    indices = []
    for c in range(10):
        cls_idx = np.where(targets == c)[0]
        chosen  = np.random.choice(cls_idx, size=min(samples_per_class, len(cls_idx)),
                                   replace=False)
        indices.extend(chosen.tolist())
    np.random.shuffle(indices)
    return indices


def split_indices_among_clients(indices, num_clients):
    """Round-robin split of indices across clients (IID for quick test)."""
    np.random.shuffle(indices)
    return [indices[i::num_clients] for i in range(num_clients)]


# ------------------------------------------------------------------ #
#  One FL round                                                        #
# ------------------------------------------------------------------ #
def local_train(global_model, dataloader, cfg, round_num):
    """Train a local copy of the model; return it and average loss."""
    local = TinyCNN(cfg['NUM_CLASSES']).to(cfg['DEVICE'])
    local.load_state_dict(copy.deepcopy(global_model.state_dict()))
    local.train()

    cosine_lr = cfg['LEARNING_RATE'] * 0.5 * (
        1.0 + math.cos(math.pi * round_num / max(cfg['NUM_ROUNDS'], 1))
    )
    cosine_lr = max(cosine_lr, cfg['LEARNING_RATE'] * 0.01)

    opt = optim.SGD(local.parameters(), lr=cosine_lr,
                    momentum=0.9, weight_decay=cfg['WEIGHT_DECAY'], nesterov=True)
    criterion = nn.CrossEntropyLoss()

    total_loss, batches = 0.0, 0
    for _ in range(cfg['LOCAL_EPOCHS']):
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(cfg['DEVICE']), lbls.to(cfg['DEVICE'])
            opt.zero_grad()
            loss = criterion(local(imgs), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local.parameters(), cfg['GRAD_CLIP'])
            opt.step()
            total_loss += loss.item()
            batches += 1

    local.eval()
    avg_loss = total_loss / max(batches, 1)
    return local, avg_loss


def fedavg(updates):
    """Simple average of a list of flat update tensors."""
    return torch.stack(updates).mean(dim=0)


def evaluate(model, loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            total_loss += criterion(out, lbls).item()
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    cfg = CFG
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("  Quick FL Sanity Test")
    print(f"  Clients={cfg['NUM_CLIENTS']}  Rounds={cfg['NUM_ROUNDS']}  "
          f"LocalEpochs={cfg['LOCAL_EPOCHS']}")
    print(f"  SamplesPerClass={cfg['SAMPLES_PER_CLASS']}  BatchSize={cfg['BATCH_SIZE']}")
    print("=" * 60)

    # ------ Dataset ------
    print("\nLoading CIFAR-10 (small subset)...")
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_train = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=t_train)
    full_test  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=t_test)

    train_idx  = get_balanced_subset(full_train, cfg['SAMPLES_PER_CLASS'])
    client_idx = split_indices_among_clients(train_idx, cfg['NUM_CLIENTS'])

    client_loaders = [
        DataLoader(Subset(full_train, idx), batch_size=cfg['BATCH_SIZE'], shuffle=True)
        for idx in client_idx
    ]
    # Use only 1000 test samples for speed
    test_idx    = get_balanced_subset(full_test, 100)  # 100 per class = 1000 total
    test_loader = DataLoader(Subset(full_test, test_idx),
                             batch_size=cfg['BATCH_SIZE'], shuffle=False)

    print(f"  Train samples : {len(train_idx)} ({len(train_idx)//cfg['NUM_CLIENTS']} per client)")
    print(f"  Test  samples : {len(test_idx)}")

    # ------ Global model ------
    global_model = TinyCNN(cfg['NUM_CLASSES']).to(cfg['DEVICE'])
    init_loss, init_acc = evaluate(global_model, test_loader, cfg['DEVICE'])
    print(f"\nInitial  | Loss: {init_loss:.4f} | Acc: {init_acc:.2f}%  "
          f"(random ≈ 10%)\n")

    # ------ FL rounds ------
    history = []
    total_start = time.time()

    for rnd in range(cfg['NUM_ROUNDS']):
        rnd_start = time.time()

        # Broadcast global state
        global_model.eval()
        global_flat = global_model.get_state_dict_flat()

        # Local training + compute deltas
        updates, losses = [], []
        for cid, loader in enumerate(client_loaders):
            local_model, loss = local_train(global_model, loader, cfg, rnd)
            local_flat = local_model.get_state_dict_flat()
            delta = local_flat - global_flat          # full state_dict delta (BN included)
            updates.append(delta)
            losses.append(loss)

        # FedAvg aggregation
        aggregated = fedavg(updates)
        update_norm = aggregated.norm(2).item()

        # Apply to global model
        new_flat = global_flat + cfg['SERVER_LR'] * aggregated
        global_model.set_state_dict_flat(new_flat)

        # Evaluate
        test_loss, test_acc = evaluate(global_model, test_loader, cfg['DEVICE'])
        elapsed = time.time() - rnd_start

        history.append({'acc': test_acc, 'loss': test_loss, 'norm': update_norm})

        print(f"Round {rnd+1:2d}/{cfg['NUM_ROUNDS']} | "
              f"TrainLoss: {sum(losses)/len(losses):.4f} | "
              f"TestLoss: {test_loss:.4f} | "
              f"Acc: {test_acc:.2f}% | "
              f"UpdateNorm: {update_norm:.4f} | "
              f"Time: {elapsed:.1f}s")

        if update_norm < 1e-6:
            print("  ⚠️  WARNING: Update norm near zero — weights not updating!")

    # ------ Summary ------
    total_time = time.time() - total_start
    best_acc = max(h['acc'] for h in history)
    final_acc = history[-1]['acc']

    print("\n" + "=" * 60)
    print(f"  DONE in {total_time:.1f}s")
    print(f"  Initial accuracy : {init_acc:.2f}%")
    print(f"  Final  accuracy  : {final_acc:.2f}%")
    print(f"  Best   accuracy  : {best_acc:.2f}%")
    print(f"  Improvement      : +{final_acc - init_acc:.2f}%")

    if final_acc > init_acc + 5:
        print("\n  ✅ PASS: Model is learning — weights updating correctly!")
    else:
        print("\n  ❌ FAIL: Accuracy barely moved — check your setup.")

    print("=" * 60)


if __name__ == '__main__':
    main()
