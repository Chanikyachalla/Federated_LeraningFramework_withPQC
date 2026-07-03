"""
Trust-Based Client Scoring for Federated Learning

Improvements over v1 (single-round, single-metric):
  1. Stratified + rotating validation set — not a fixed 5-batch pool.
     Avoids systematically punishing honest non-IID clients whose local
     distribution differs from whatever classes happen to be in those 5 batches.

  2. Per-class loss decomposition — label-flipping/poisoning spikes loss on
     specific classes; an honest non-IID client is uniformly weaker or better.
     We flag concentrated loss spikes (poisoning signature) vs. spread increases.

  3. EMA-based historical trust (alpha=0.2) — smoothed over many rounds so a
     consistent attacker accumulates penalties while one noisy round cannot tank
     an honest client.

  4. Warm-up rounds — for the first TRUST_WARM_UP_ROUNDS rounds, trust scores
     are not applied (plain FedAvg). Round-1 loss deltas are pure noise.

  5. Trimmed-mean backstop — combined with trust weights so even if trust
     calibration is imperfect at high attack ratios, extreme updates are cut.

  6. Combined mode — geometric mean of cosine-based and loss-based trust scores,
     forcing an attacker to evade BOTH checks simultaneously.

Call ordering inside FLServer.aggregate_updates (each round):
    1. weights = trust_mgr.get_weights(valid_client_ids, round_num)   # BEFORE agg
    2. aggregated = defense.aggregate(updates, trust_weights=weights)
    3. trust_mgr.update_scores(...)                                    # AFTER agg
"""

import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from config import (
    TRUST_SCORE_ENABLED,
    TRUST_SCORING_METHOD,
    TRUST_WINDOW,
    TRUST_ALPHA,
    TRUST_MIN,
    TRUST_NORM_PENALTY,
    TRUST_WARM_UP_ROUNDS,
    NUM_CLIENTS,
    NUM_CLASSES,
)


# ---------------------------------------------------------------------------
# Helper: cosine similarity between two 1-D tensors
# ---------------------------------------------------------------------------
def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return cosine similarity in [-1, 1]; safe when either norm is zero."""
    a_flat = a.view(-1).float()
    b_flat = b.view(-1).float()
    norm_a = float(torch.norm(a_flat))
    norm_b = float(torch.norm(b_flat))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Stratified validation pool builder
# ---------------------------------------------------------------------------
def build_validation_pool(test_dataset, num_classes: int = NUM_CLASSES,
                          batches_per_class: int = 3) -> Dict[int, list]:
    """
    Build a stratified pool of batches, indexed by the most-common class in
    each batch.  Up to `batches_per_class` batches per class are stored.

    Args:
        test_dataset:      An iterable of (images, labels) batches.
        num_classes:       Number of target classes.
        batches_per_class: Max batches stored per class.

    Returns:
        Dict mapping class_id → list of (images, labels) tuples.
    """
    pool: Dict[int, list] = {c: [] for c in range(num_classes)}
    for batch in test_dataset:
        x, y = batch
        for c in torch.unique(y).tolist():
            c = int(c)
            if len(pool[c]) < batches_per_class:
                pool[c].append((x, y))
    return pool


def sample_validation_batches(pool: Dict[int, list],
                               round_seed: int,
                               k: int = 5) -> list:
    """
    Deterministically sample k batches from the pool, seeded by round number.
    Rotating the sample every round prevents an attacker from fitting an evasion
    to one static validation set.

    Args:
        pool:       Stratified pool from build_validation_pool.
        round_seed: Round number (used as RNG seed).
        k:          Number of batches to sample.

    Returns:
        List of (images, labels) tuples.
    """
    rng = random.Random(round_seed)
    flat = [b for batches in pool.values() for b in batches]
    if not flat:
        return []
    return rng.sample(flat, min(k, len(flat)))


# ---------------------------------------------------------------------------
# Per-class loss decomposition
# ---------------------------------------------------------------------------
def class_wise_loss(model, batches: list,
                    num_classes: int = NUM_CLASSES,
                    device: str = 'cpu') -> Dict[int, Optional[float]]:
    """
    Compute average cross-entropy loss per class over the given batches.

    Returns:
        Dict mapping class_id → mean loss (or None if class not present).
    """
    losses: Dict[int, list] = {c: [] for c in range(num_classes)}
    model.eval()
    with torch.no_grad():
        for x, y in batches:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            for c in range(num_classes):
                mask = (y == c)
                if mask.sum() > 0:
                    loss_c = F.cross_entropy(out[mask], y[mask]).item()
                    losses[c].append(loss_c)
    return {c: (sum(v) / len(v) if v else None) for c, v in losses.items()}


def _is_concentrated_spike(base_class_losses: Dict[int, Optional[float]],
                            client_class_losses: Dict[int, Optional[float]],
                            spike_threshold: float = 0.3,
                            spike_fraction: float = 0.25) -> bool:
    """
    Return True if the loss increase is concentrated in a small fraction of
    classes — the signature of targeted label-flipping poisoning.

    An honest non-IID client tends to shift loss uniformly; a label-flipping
    attacker spikes loss on exactly the targeted class(es).

    Args:
        base_class_losses:   Per-class loss before update.
        client_class_losses: Per-class loss after applying client's update.
        spike_threshold:     Min absolute increase to count as a spike.
        spike_fraction:      If spiked_classes / total_present > this, it's
                             considered spread (not poisoning).
    """
    deltas = []
    for c in range(NUM_CLASSES):
        b = base_class_losses.get(c)
        cl = client_class_losses.get(c)
        if b is not None and cl is not None:
            deltas.append(cl - b)

    if not deltas:
        return False

    num_spiked = sum(1 for d in deltas if d > spike_threshold)
    fraction_spiked = num_spiked / len(deltas)

    # Concentrated spike: only a few classes got much worse
    return 0 < fraction_spiked <= spike_fraction


# ---------------------------------------------------------------------------
# TrustManager
# ---------------------------------------------------------------------------
class TrustManager:
    """
    Manages per-client trust scores for trust-weighted aggregation.

    Attributes
    ----------
    scores : Dict[int, float]
        Current trust score for each client (initialised to 1.0).
    history : Dict[int, deque[float]]
        Sliding window (length TRUST_WINDOW) of per-round scores.
    _validation_pool : Optional[Dict]
        Stratified validation pool built on first use.
    """

    def __init__(self, client_ids: Optional[List[int]] = None):
        ids = client_ids if client_ids is not None else list(range(NUM_CLIENTS))
        self.scores: Dict[int, float] = {cid: 1.0 for cid in ids}
        self.history: Dict[int, deque] = {
            cid: deque(maxlen=TRUST_WINDOW) for cid in ids
        }
        self._validation_pool: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Validation pool management
    # ------------------------------------------------------------------

    def build_pool(self, test_dataset, num_classes: int = NUM_CLASSES,
                   batches_per_class: int = 3) -> None:
        """Build (or rebuild) the stratified validation pool from test_dataset."""
        self._validation_pool = build_validation_pool(
            test_dataset, num_classes, batches_per_class
        )

    def get_validation_batches(self, round_num: int, k: int = 5) -> list:
        """Return k deterministically-sampled batches for this round."""
        if self._validation_pool is None:
            return []
        return sample_validation_batches(self._validation_pool, round_num, k)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def update_scores(
        self,
        client_ids: List[int],
        updates: List[torch.Tensor],
        aggregated_update: Optional[torch.Tensor],
        client_losses: Optional[List[float]] = None,
        base_loss: Optional[float] = None,
        client_class_losses: Optional[List[Dict[int, Optional[float]]]] = None,
        base_class_losses: Optional[Dict[int, Optional[float]]] = None,
        round_num: int = 0,
    ) -> None:
        """
        Update trust scores after one FL round.

        Args:
            client_ids:           IDs of clients whose updates were accepted.
            updates:              Corresponding update tensors.
            aggregated_update:    Aggregated update (reference for cosine).
            client_losses:        Scalar loss after applying each client's update.
            base_loss:            Scalar loss before any update.
            client_class_losses:  Per-class loss dict for each client (optional).
            base_class_losses:    Per-class loss dict before updates (optional).
            round_num:            Current round number (0-indexed).
        """
        if not client_ids:
            return

        # During warm-up, accumulate history but don't penalise
        in_warmup = round_num < TRUST_WARM_UP_ROUNDS

        # Norm statistics for norm-ratio scoring
        norms = np.array([float(u.norm(2).cpu()) for u in updates], dtype=float)
        median_norm = float(np.median(norms)) if len(norms) > 0 else 1.0

        for idx, (cid, update) in enumerate(zip(client_ids, updates)):
            self._ensure_client(cid)

            method = TRUST_SCORING_METHOD

            # ---- Cosine score ----
            if aggregated_update is not None:
                cos_sim = _cosine_similarity(update, aggregated_update)
                cos_score = (cos_sim + 1.0) / 2.0
            else:
                cos_score = 0.5

            # Norm-ratio penalty
            client_norm = float(update.norm(2).cpu())
            ratio = (client_norm / median_norm) if median_norm > 1e-12 else 1.0
            norm_score = math.exp(-((ratio - 1.0) ** 2) / 0.5)
            cosine_raw = (
                (1.0 - TRUST_NORM_PENALTY) * cos_score
                + TRUST_NORM_PENALTY * norm_score
            )

            # ---- Loss score ----
            loss_raw = 1.0  # default (neutral) when loss data unavailable
            if client_losses is not None and base_loss is not None:
                client_loss = client_losses[idx]
                loss_diff = client_loss - base_loss
                relative_delta = loss_diff / max(base_loss, 1e-8)

                if relative_delta <= 0:
                    # Client helped — slight trust recovery
                    loss_raw = min(1.0, self.scores[cid] * 1.05)
                else:
                    # Scale-invariant exponential penalty
                    loss_raw = math.exp(-10.0 * relative_delta)

                # Extra penalty if the loss spike is concentrated (poisoning sig)
                if (client_class_losses is not None
                        and base_class_losses is not None
                        and idx < len(client_class_losses)):
                    if _is_concentrated_spike(base_class_losses,
                                              client_class_losses[idx]):
                        loss_raw *= 0.5   # halve trust score for targeted spike

            # ---- Combine ----
            if method == 'combined':
                # Geometric mean forces attacker to evade BOTH checks
                raw_score = math.sqrt(cosine_raw * loss_raw)
            elif method == 'loss':
                raw_score = loss_raw
            else:  # 'cosine' (default)
                raw_score = cosine_raw

            # During warm-up: don't apply the penalty, just observe
            if in_warmup:
                raw_score = max(raw_score, self.scores[cid])

            # EMA update  (alpha=0.2: slow to react to noise, slow to recover)
            old_score = self.scores[cid]
            new_score = (1.0 - TRUST_ALPHA) * old_score + TRUST_ALPHA * raw_score

            # Clamp to [TRUST_MIN, 1.0]
            new_score = max(TRUST_MIN, min(1.0, new_score))
            self.scores[cid] = new_score
            self.history[cid].append(new_score)

        # Clients absent this round: mild decay
        absent = set(self.scores.keys()) - set(client_ids)
        for cid in absent:
            old = self.scores[cid]
            decayed = max(TRUST_MIN, (1.0 - TRUST_ALPHA) * old + TRUST_ALPHA * TRUST_MIN)
            self.scores[cid] = decayed
            self.history[cid].append(decayed)

    # ------------------------------------------------------------------
    # Weight retrieval
    # ------------------------------------------------------------------

    def get_weights(self, client_ids: List[int],
                    round_num: int = 9999) -> List[float]:
        """
        Return normalised trust weights.  During warm-up rounds, return uniform
        weights (trust filtering not yet active).
        """
        if round_num < TRUST_WARM_UP_ROUNDS:
            n = len(client_ids)
            return [1.0 / n] * n

        raw = [self.scores.get(cid, TRUST_MIN) for cid in client_ids]
        total = sum(raw)
        if total < 1e-12:
            n = len(client_ids)
            return [1.0 / n] * n
        return [w / total for w in raw]

    def get_scores(self) -> Dict[int, float]:
        """Return a snapshot of all current trust scores."""
        return dict(self.scores)

    def get_score(self, client_id: int) -> float:
        """Return the current trust score for a single client."""
        return self.scores.get(client_id, TRUST_MIN)

    def reset(self, client_id: int) -> None:
        """Reset a client's trust score to the minimum."""
        self._ensure_client(client_id)
        self.scores[client_id] = TRUST_MIN
        self.history[client_id].append(TRUST_MIN)

    def get_krum_modifiers(self, client_ids: List[int],
                           round_num: int = 9999) -> List[float]:
        """
        Per-client Krum-score multipliers based on trust.
        Lower trust → higher multiplier → less likely to be selected.
        multiplier = 2 - trust_score ∈ [1, 2-TRUST_MIN]
        During warm-up returns uniform multipliers (1.0).
        """
        if round_num < TRUST_WARM_UP_ROUNDS:
            return [1.0] * len(client_ids)
        return [2.0 - self.scores.get(cid, TRUST_MIN) for cid in client_ids]

    def get_manhattan_weights(self, client_ids: List[int],
                              round_num: int = 9999) -> np.ndarray:
        """
        Per-client weight adjustments for Manhattan-distance defense.
        During warm-up returns uniform weights.
        """
        if round_num < TRUST_WARM_UP_ROUNDS:
            return np.ones(len(client_ids), dtype=float)
        return np.array(
            [self.scores.get(cid, TRUST_MIN) for cid in client_ids],
            dtype=float,
        )

    def print_scores(self, round_num: int) -> None:
        """Pretty-print the current trust scores table."""
        print(f"\n  [Trust Scores] Round {round_num + 1}:")
        print(f"  {'Client':>8} | {'Score':>8} | {'History (last {})'.format(TRUST_WINDOW)}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*20}")
        for cid, score in sorted(self.scores.items()):
            hist = list(self.history[cid])
            hist_str = " ".join(f"{s:.3f}" for s in hist[-5:])
            print(f"  {cid:>8} | {score:>8.4f} | {hist_str}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_client(self, client_id: int) -> None:
        """Lazily register a new client with initial trust = 1.0."""
        if client_id not in self.scores:
            self.scores[client_id] = 1.0
            self.history[client_id] = deque(maxlen=TRUST_WINDOW)
