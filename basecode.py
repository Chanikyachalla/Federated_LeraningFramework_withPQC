"""
Post-Quantum Secure Federated Learning Framework
Main Entry Point

Usage:
    python basecode.py --experiment 1   (Run single experiment)
    python basecode.py --all            (Run all 12 experiments)
"""

import argparse
from pathlib import Path

from experiments import (
    run_experiment_1,  run_experiment_2,  run_experiment_3,
    run_experiment_4,  run_experiment_5,  run_experiment_6,
    run_experiment_7,  run_experiment_8,  run_experiment_9,
    run_experiment_10, run_experiment_11, run_experiment_12,
    run_experiment_13, run_experiment_14, run_all_experiments
)
from experiments_multiseed import run_experiment_15


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Post-Quantum Secure Federated Learning Framework"
    )

    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help='Run specific experiment (1-15)'
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 12 experiments"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of federated learning rounds"
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Number of clients"
    )

    parser.add_argument(
        "--Byzantine",
        type=int,
        default=1,
        help="Number of Byzantine clients"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for Experiment 15 (default: 42 43 44 45 46)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test for Experiment 15 (2 rounds, 2 seeds)"
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore cached checkpoints/results and run fresh"
    )

    args = parser.parse_args()

    # Create output directories
    Path("./results").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Post-Quantum Secure Federated Learning Framework               ║
║  with Byzantine and Data Poisoning Attack Detection             ║
║  + PQC + Trust-Based Aggregation                               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    if args.all:
        run_all_experiments()

    elif args.experiment:

        experiment_funcs = {
            1:  run_experiment_1,
            2:  run_experiment_2,
            3:  run_experiment_3,
            4:  run_experiment_4,
            5:  run_experiment_5,
            6:  run_experiment_6,
            7:  run_experiment_7,
            8:  run_experiment_8,
            9:  run_experiment_9,
            10: run_experiment_10,
            11: run_experiment_11,
            12: run_experiment_12,
            13: run_experiment_13,
            14: run_experiment_14,
            15: lambda: run_experiment_15(
                seeds=(42, 43) if args.quick else tuple(args.seeds),
                rounds=2 if args.quick else args.rounds,
                fresh=args.fresh
            ),
        }

        experiment_funcs[args.experiment]()

    else:
        print("Please specify an experiment to run:")
        print("  python basecode.py --experiment 1-15  (Run specific experiment)")
        print("  python basecode.py --all               (Run all 14 experiments)")
        print("\nExperiments:")
        print("  1:  Clean FL + FedAvg")
        print("  2:  Byzantine Attack + FedAvg")
        print("  3:  Byzantine Attack + Krum")
        print("  4:  Data Poisoning + FedAvg")
        print("  5:  Data Poisoning + Krum")
        print("  6:  Byzantine Attack + PQC + FedAvg")
        print("  7:  Byzantine Attack + PQC + Krum")
        print("  8:  Byzantine Attack + Manhattan Distance")
        print("  9:  Data Poisoning + Manhattan Distance")
        print("  10: Byzantine Attack + PQC + Manhattan Distance")
        print("  11: Data Poisoning + PQC + Loss-Based Trust")
        print("  12: Data Poisoning + PQC + Combined Trust (old)")
        print("  13: Data Poisoning + PQC + OptiGradTrust [Sign + Adaptive Alpha]")
        print("  14: Data Poisoning + PQC + OptiGradTrust v4 [Sharpening + Trust-Gated Trim]")
        print("  15: Multi-Seed Statistical Evaluation (Clean / Byzantine / LabelFlip x FedAvg / OptiGradTrust v4) ← NEW")


if __name__ == "__main__":
    main()