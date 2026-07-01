"""
Checkpoint Manager — crash-recovery for federated learning experiments.

Saves after every CHECKPOINT_INTERVAL rounds so that a power loss / OOM
event loses at most CHECKPOINT_INTERVAL rounds of work.

Saved artefacts per experiment (under CHECKPOINT_DIR/<safe_name>/):
  model_round_XXXX.pt   — torch state_dict of the global model
  progress.json         — last completed round + full metrics snapshot
  progress.tmp          — atomic-write staging file (never left on disk)

On the next run, ExperimentRunner checks for a progress.json and, if
RESUME_FROM_CHECKPOINT is True, reloads the model and metrics and
continues from `last_completed_round + 1`.
"""

import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from config import (
    CHECKPOINT_DIR, CHECKPOINT_KEEP_LAST, RESUME_FROM_CHECKPOINT,
    NUM_ROUNDS, RESULTS_DIR, LOG_DIR
)

# ─── Absolute paths anchored to this source file ────────────────────────────
_PROJECT_ROOT = Path(__file__).parent
_CKPT_ROOT    = _PROJECT_ROOT / CHECKPOINT_DIR
_RESULTS_ROOT = _PROJECT_ROOT / RESULTS_DIR
_LOG_ROOT     = _PROJECT_ROOT / LOG_DIR


# ─── JSON serialisation helper ───────────────────────────────────────────────
def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars for json.dump."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


def _safe_name(experiment_name: str) -> str:
    """Convert an experiment name to a filesystem-safe string."""
    return (
        experiment_name
        .replace(':', '')
        .replace(' ', '_')
        .replace('/', '-')
        .replace('\\', '-')
    )


# ─── CheckpointManager ───────────────────────────────────────────────────────
class CheckpointManager:
    """
    Saves and restores experiment checkpoints.

    Usage
    -----
    ckpt = CheckpointManager(experiment_name)

    # at startup
    state = ckpt.load_latest()          # None if no checkpoint
    if state:
        server.model.load_state_dict(state['model_state_dict'])
        metrics_tracker.restore(state['metrics'])
        start_round = state['last_completed_round'] + 1

    # after each round
    if (round_num + 1) % CHECKPOINT_INTERVAL == 0:
        ckpt.save(round_num, server.model.state_dict(), metrics_tracker.metrics)
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.safe  = _safe_name(experiment_name)
        self.ckpt_dir = _CKPT_ROOT / self.safe
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.ckpt_dir / 'progress.json'

        # Per-experiment logger → checkpoints/<safe>/checkpoint.log
        log_path = self.ckpt_dir / 'checkpoint.log'
        self._log = logging.getLogger(f'checkpoint.{self.safe}')
        self._log.setLevel(logging.INFO)
        if not self._log.handlers:
            fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            fh.setFormatter(logging.Formatter(
                '%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self._log.addHandler(fh)

    # ── Public API ──────────────────────────────────────────────────────────

    def save(self,
             round_num: int,
             model_state_dict: dict,
             metrics: dict,
             experiment_config: dict = None) -> None:
        """
        Persist a checkpoint for the completed round.

        Steps
        -----
        1. Write model state dict to `model_round_XXXX.pt`
        2. Write progress.json atomically (write to .tmp, then rename)
        3. Prune old checkpoint files
        """
        # 1. Save model weights
        ckpt_file = self.ckpt_dir / f'model_round_{round_num:04d}.pt'
        torch.save(model_state_dict, ckpt_file)

        # 2. Save progress atomically
        progress: Dict[str, Any] = {
            'experiment_name':     self.experiment_name,
            'last_completed_round': round_num,
            'total_rounds':        NUM_ROUNDS,
            'checkpoint_file':     ckpt_file.name,
            'metrics':             _to_json_safe(metrics),
            'experiment_config':   experiment_config or {},
        }
        tmp = self.progress_file.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
        tmp.replace(self.progress_file)   # atomic on POSIX; near-atomic on Windows

        msg = f'Checkpoint saved -- round {round_num:04d} -> {ckpt_file.name}'
        self._log.info(msg)
        print(f'  [Checkpoint] {msg}')

        # 3. Prune old model files
        self._prune(keep=CHECKPOINT_KEEP_LAST, latest_round=round_num)

    def load_latest(self) -> Optional[Dict]:
        """
        Load the most recent checkpoint.

        Returns
        -------
        dict with keys:
            'last_completed_round' : int
            'model_state_dict'     : dict  (torch state_dict)
            'metrics'              : dict  (metrics data)
            'experiment_config'    : dict
        or None if no checkpoint exists or loading fails.
        """
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, encoding='utf-8') as f:
                progress = json.load(f)

            ckpt_file = self.ckpt_dir / progress['checkpoint_file']
            if not ckpt_file.exists():
                self._log.warning(
                    f'Progress file references missing model: {ckpt_file}'
                )
                return None

            model_state = torch.load(
                ckpt_file, map_location='cpu', weights_only=False
            )
            progress['model_state_dict'] = model_state

            last = progress['last_completed_round']
            self._log.info(f'Resuming from round {last} ({ckpt_file.name})')
            print(f'\n  [Checkpoint] Resuming "{self.experiment_name}" '
                  f'from round {last + 1}/{NUM_ROUNDS}')
            return progress

        except Exception as e:
            self._log.error(f'Failed to load checkpoint: {e}')
            print(f'  [Checkpoint] WARNING: Could not load checkpoint: {e}')
            return None

    def exists(self) -> bool:
        """Return True if a valid checkpoint file exists."""
        return self.progress_file.exists()

    def mark_complete(self) -> None:
        """Write a DONE marker so we know the experiment finished cleanly."""
        done_file = self.ckpt_dir / 'DONE'
        done_file.write_text(
            f'Experiment "{self.experiment_name}" completed successfully.\n'
        )
        self._log.info('Experiment marked as DONE')

    def is_complete(self) -> bool:
        """Return True if the experiment has already finished (has DONE marker)."""
        return (self.ckpt_dir / 'DONE').exists()

    # ── Internals ───────────────────────────────────────────────────────────

    def _prune(self, keep: int, latest_round: int) -> None:
        """Delete old model checkpoint files, keeping the `keep` most recent."""
        files = sorted(self.ckpt_dir.glob('model_round_*.pt'))
        to_delete = files[:-keep] if len(files) > keep else []
        for f in to_delete:
            try:
                f.unlink()
                self._log.info(f'Pruned old checkpoint: {f.name}')
            except Exception:
                pass


# ─── Ensure all output directories exist at import time ─────────────────────
def ensure_dirs() -> None:
    """Create results, logs, and checkpoint root directories."""
    for d in (_RESULTS_ROOT, _LOG_ROOT, _CKPT_ROOT):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
