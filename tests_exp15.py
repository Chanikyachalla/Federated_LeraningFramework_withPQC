"""
Unit Tests for Experiment 15 (Multi-Seed Statistical Evaluation)
Runs with NO GPU and NO dataset download required.
"""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model import create_model
from attacks import ModelPoisoningAttack
from trust import TrustManager
from federated_learning import FLServer
from checkpoint import CheckpointManager
from experiments_multiseed import (
    set_global_seed,
    SeededAttackManager,
    _SeededDataPoisoning,
    analyze_experiment_15,
    EXP15_SCENARIOS,
    EXP15_DEFENSES,
    _ci95,
)


class TestExp15Determinism(unittest.TestCase):
    def test_global_seed_model_weights(self):
        set_global_seed(42)
        m1 = create_model(device="cpu")
        w1 = list(m1.parameters())[0].detach().clone()

        set_global_seed(42)
        m2 = create_model(device="cpu")
        w2 = list(m2.parameters())[0].detach().clone()

        set_global_seed(99)
        m3 = create_model(device="cpu")
        w3 = list(m3.parameters())[0].detach().clone()

        self.assertTrue(torch.equal(w1, w2), "Same seed must produce identical weights")
        self.assertFalse(torch.equal(w1, w3), "Different seeds should produce different weights")

    def test_seeded_attack_manager(self):
        rng1 = np.random.RandomState(42)
        mgr1 = SeededAttackManager(
            num_clients=10, num_byzantine=3,
            byzantine_scale=10.0, poison_ratio=0.5, rng=rng1
        )
        byz1 = mgr1.get_byzantine_clients()

        rng2 = np.random.RandomState(42)
        mgr2 = SeededAttackManager(
            num_clients=10, num_byzantine=3,
            byzantine_scale=10.0, poison_ratio=0.5, rng=rng2
        )
        byz2 = mgr2.get_byzantine_clients()

        self.assertEqual(byz1, byz2, "Identical seed must pick identical Byzantine client IDs")
        self.assertEqual(len(byz1), 3)

        # Test model poisoning
        dummy_update = torch.tensor([1.0, -2.0, 3.0])
        byz_cid = byz1[0]
        honest_cid = next(cid for cid in range(10) if cid not in byz1)

        poisoned = mgr1.apply_model_poisoning(byz_cid, dummy_update)
        expected = -10.0 * dummy_update
        self.assertTrue(torch.equal(poisoned, expected))

        clean = mgr1.apply_model_poisoning(honest_cid, dummy_update)
        self.assertTrue(torch.equal(clean, dummy_update))

    def test_seeded_label_flipping(self):
        rng1 = np.random.RandomState(123)
        p1 = _SeededDataPoisoning(poison_ratio=0.5, rng=rng1)
        imgs = torch.zeros((10, 3, 32, 32))
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        _, p_labels1 = p1.poison_batch(imgs, labels)

        rng2 = np.random.RandomState(123)
        p2 = _SeededDataPoisoning(poison_ratio=0.5, rng=rng2)
        _, p_labels2 = p2.poison_batch(imgs, labels)

        self.assertTrue(torch.equal(p_labels1, p_labels2), "Label flipping must be deterministic with local RNG")


class TestExp15ExplicitParameters(unittest.TestCase):
    def test_trust_manager_scoring_method(self):
        tm = TrustManager(scoring_method="combined")
        self.assertEqual(tm.scoring_method, "combined")

    def test_fl_server_trust_scoring_method(self):
        model = create_model(device="cpu")
        server = FLServer(
            model=model,
            defense_name="fedavg",
            device="cpu",
            trust_enabled=True,
            trust_scoring_method="combined",
        )
        self.assertIsNotNone(server.trust_manager)
        self.assertEqual(server.trust_manager.scoring_method, "combined")


class TestExp15Checkpointing(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_unique_run_name_checkpoint(self):
        run_name = "Exp15_test_byzantine_optigradtrust_v4_seed42"
        ckpt = CheckpointManager(run_name)
        self.assertIn("Exp15_test_byzantine_optigradtrust_v4_seed42", str(ckpt.ckpt_dir))

        # Test save and load
        model = create_model(device="cpu")
        metrics = {"test_accuracy": [50.0, 55.0], "test_loss": [1.5, 1.2]}
        ckpt.save(1, model.state_dict(), metrics)

        self.assertTrue(ckpt.exists())
        loaded = ckpt.load_latest()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["last_completed_round"], 1)
        self.assertEqual(loaded["metrics"]["test_accuracy"], [50.0, 55.0])

        ckpt.mark_complete()
        self.assertTrue(ckpt.is_complete())


class TestExp15Analysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ci95_calculation(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = _ci95(vals)
        self.assertFalse(math.isnan(ci))
        self.assertGreater(ci, 0.0)

    def test_full_analysis_pipeline_with_mock_data(self):
        # Create mock run results in test_dir
        seeds = [42, 43]
        for sc in EXP15_SCENARIOS:
            for df in EXP15_DEFENSES:
                for seed in seeds:
                    run_name = f"Exp15_{sc}_{df}_seed{seed}"
                    rd = self.test_dir / run_name
                    rd.mkdir(parents=True, exist_ok=True)
                    acc = 85.0 if df == "optigradtrust_v4" else 75.0
                    if sc == "clean":
                        acc += 5.0
                    res = {
                        "config": {
                            "scenario": sc, "defense": df,
                            "seed": seed, "rounds": 5,
                            "num_clients": 10, "pqc_enabled": True
                        },
                        "seed": seed,
                        "malicious_client_ids": [0, 1, 2] if sc != "clean" else [],
                        "final_accuracy": acc + (seed - 42) * 0.5,
                        "last5_mean_accuracy": acc + (seed - 42) * 0.4,
                        "best_accuracy": acc + 1.0,
                        "per_round_accuracy": [acc - 4, acc - 2, acc - 1, acc, acc],
                        "per_round_loss": [2.0, 1.5, 1.2, 1.0, 0.9],
                        "mean_trust_honest": 0.95 if df == "optigradtrust_v4" else None,
                        "mean_trust_malicious": 0.25 if df == "optigradtrust_v4" else None,
                        "total_time_sec": 10.5
                    }
                    import json
                    (rd / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

        # Run analyze_experiment_15
        analysis = analyze_experiment_15(out_dir=self.test_dir)
        self.assertEqual(analysis["num_runs_found"], 12)
        self.assertEqual(len(analysis["summary"]), 6)
        self.assertEqual(len(analysis["paired_comparison"]), 3)

        # Check output artifacts
        self.assertTrue((self.test_dir / "summary_table.csv").exists())
        self.assertTrue((self.test_dir / "summary_table.md").exists())
        self.assertTrue((self.test_dir / "paired_comparison.csv").exists())
        self.assertTrue((self.test_dir / "paired_comparison.md").exists())
        self.assertTrue((self.test_dir / "bar_chart.png").exists())
        self.assertTrue((self.test_dir / "accuracy_curves.png").exists())
        self.assertTrue((self.test_dir / "paired_diff.png").exists())
        self.assertTrue((self.test_dir / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
