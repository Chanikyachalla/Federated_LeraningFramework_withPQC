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

--- OptiGradTrust Enhancements (v3) ---
  7. Sign Consistency Score (OptiGradTrust Feature 2) — measures the fraction
     of gradient dimensions whose sign matches the reference (aggregated) update.
     Label-flipping attackers invert gradient directions on targeted features,
     producing low sign-consistency even when cosine similarity is moderate.
     sign_score = (matching_signs / total_dims)  ∈ [0, 1]

  8. Temporal Stability + Adaptive Alpha (OptiGradTrust Feature 5) — compares
     a client's current raw score against their recent historical average.
     If the drop is sudden and large (anomalous behaviour), alpha is boosted
     temporarily so the EMA penalty lands FAST rather than being smoothed away.
     This closes the "sleeper attacker" loophole in plain fixed-alpha EMA.

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
    TRUST_TEMPORAL_DROP_THRESHOLD,
    TRUST_TEMPORAL_BOOST_ALPHA,
    TRUST_TEMPORAL_SIGMA_MULT,
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
# OptiGradTrust Feature 2: Sign Consistency Score
# ---------------------------------------------------------------------------
def _sign_consistency_score(update: torch.Tensor,
                             reference: torch.Tensor) -> float:
    """
    Compute the fraction of gradient dimensions whose sign matches the
    reference update (typically the aggregated update).

    Honest clients' updates align in sign with the global gradient direction
    on most dimensions.  Label-flipping attackers push weights in the wrong
    direction on poisoned-class features, producing systematically inverted
    signs even when the cosine similarity is only mildly negative.

    Returns:
        float in [0, 1] — 1.0 means perfect sign agreement, 0.0 means
        every dimension is sign-flipped (adversarial).
    """
    u = update.view(-1).float()
    r = reference.view(-1).float()
    if u.numel() == 0:
        return 0.5  # neutral if empty

    # Only consider dimensions where the reference has a non-trivial magnitude
    # (avoids noise from near-zero reference components).
    mask = r.abs() > 1e-9
    if mask.sum() == 0:
        return 0.5  # no reference signal → neutral

    u_signs = torch.sign(u[mask])
    r_signs = torch.sign(r[mask])
    matching = (u_signs == r_signs).float().mean().item()
    return float(matching)


# ---------------------------------------------------------------------------
# OptiGradTrust Feature 5: Variance-Aware Adaptive Alpha (non-IID safe)
# ---------------------------------------------------------------------------
def _adaptive_alpha(raw_score: float,
                    history: deque,
                    base_alpha: float = 0.2,
                    base_drop_threshold: float = 0.25,
                    boost_alpha: float = 0.6,
                    sigma_multiplier: float = 2.0) -> float:
    """
    Return an adaptive EMA alpha that reacts faster when a client's current
    raw score drops *anomalously* relative to their own recent history.

    NON-IID SAFE design (key improvement over naive fixed-threshold):
    ---------------------------------------------------------------
    Honest non-IID clients have naturally HIGH variance in their round-to-round
    raw scores because their local data distribution differs from the global.
    A fixed threshold (e.g. drop > 0.25) would falsely flag them in rounds
    where their local class distribution happens to diverge from the aggregated
    update direction.

    Fix: the trigger threshold is personalised per client:
        effective_threshold = max(base_drop_threshold,
                                  sigma_multiplier × std(history))

    This means:
    - A stable honest IID client   (std ≈ 0.02) → threshold ≈ 0.25  (base)
    - A volatile honest non-IID    (std ≈ 0.12) → threshold ≈ 0.24  → same base
      BUT: the std check ensures the DROP must be > 2σ of THEIR OWN history.
    - A sleeper attacker            (std ≈ 0.04, then big drop) → flagged

    Args:
        raw_score:           Current round's raw trust signal ∈ [0, 1].
        history:             deque of past EMA-smoothed trust scores.
        base_alpha:          Normal EMA alpha (e.g., 0.2).
        base_drop_threshold: Minimum absolute drop to consider suspicious.
        boost_alpha:         Alpha when a suspicious drop is detected (0.6).
                             Softened from 0.7 → 0.6 to reduce false-positive
                             impact on honest non-IID clients.
        sigma_multiplier:    How many σ above history variance constitutes
                             an anomalous drop (default 2.0 → 2-sigma rule).

    Returns:
        float — the alpha to use for this round's EMA update.
    """
    if len(history) < 3:
        # Not enough history to compute meaningful variance.
        # Default to base_alpha to avoid penalising clients in early rounds.
        return base_alpha

    hist_list = list(history)
    hist_mean = float(np.mean(hist_list))
    hist_std  = float(np.std(hist_list))

    drop = hist_mean - raw_score  # positive = score got worse this round

    # Personalised threshold: must exceed BOTH the base absolute threshold
    # AND be more than sigma_multiplier standard deviations below the client's
    # own historical mean. This protects honest non-IID clients whose scores
    # naturally fluctuate, while still catching sudden attacker behaviour.
    variance_threshold = sigma_multiplier * hist_std
    effective_threshold = max(base_drop_threshold, variance_threshold)

    if drop > effective_threshold:
        return boost_alpha  # anomalous sudden drop → react fast
    return base_alpha


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

        # Pre-compute median update (element-wise) for sign consistency reference
        # Only used when aggregated_update is available; falls back to aggregated
        # update if not enough clients.
        if aggregated_update is not None and len(updates) >= 2:
            stacked = torch.stack([u.view(-1).float() for u in updates])
            reference_update = torch.median(stacked, dim=0).values
        elif aggregated_update is not None:
            reference_update = aggregated_update
        else:
            reference_update = None

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

            # ---- OptiGradTrust Feature 2: Sign Consistency Score (universal gate) ----
            # Computed regardless of scoring mode so ALL modes benefit.
            # Applied AFTER mode-selection as a multiplicative gate on raw_score:
            #   gate = 1.0  when sign_score >= 0.80  (honest client, no penalty)
            #   gate decays toward 0.5 as sign_score falls to 0.0 (adversarial)
            # Formula: gate = 0.5 + 0.5 * sign_score  => maps [0,1] -> [0.5, 1.0]
            # This is mode-agnostic: a poisoner with sign_score=0.40 suffers
            # a 20% penalty on top of whatever cosine/loss score they get.
            if reference_update is not None:
                sign_score = _sign_consistency_score(update, reference_update)
            else:
                sign_score = 0.5  # neutral when no reference available

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

            # ---- Apply Sign Consistency Gate (ALL modes) ----
            # gate maps sign_score ∈ [0,1] → multiplier ∈ [0.5, 1.0]
            # Honest clients (sign ≥ 0.80) get gate ≈ 0.90–1.0 (near-neutral)
            # Poisoners (sign ≈ 0.40) get gate ≈ 0.70 (additional 30% penalty)
            # Applied before warm-up guard so warm-up still prevents full penalty.
            sign_gate = 0.5 + 0.5 * sign_score
            raw_score = raw_score * sign_gate

            # During warm-up: don't apply the penalty, just observe
            if in_warmup:
                raw_score = max(raw_score, self.scores[cid])

            # ---- OptiGradTrust Feature 5: Temporal Stability / Adaptive Alpha ----
            # If score drops suddenly relative to this client's own history,
            # raise alpha temporarily so the EMA reacts fast (sleeper attacker
            # detection).  During warm-up we always use base_alpha.
            if in_warmup:
                ema_alpha = TRUST_ALPHA
            else:
                ema_alpha = _adaptive_alpha(
                    raw_score=raw_score,
                    history=self.history[cid],
                    base_alpha=TRUST_ALPHA,
                    base_drop_threshold=TRUST_TEMPORAL_DROP_THRESHOLD,
                    boost_alpha=TRUST_TEMPORAL_BOOST_ALPHA,
                    sigma_multiplier=TRUST_TEMPORAL_SIGMA_MULT,
                )

            # EMA update with (possibly boosted) alpha
            old_score = self.scores[cid]
            new_score = (1.0 - ema_alpha) * old_score + ema_alpha * raw_score

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
