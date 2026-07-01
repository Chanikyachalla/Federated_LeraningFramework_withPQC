"""
Post-Quantum Secure Federated Learning Framework
Main Entry Point

Usage:
    python basecode.py --experiment 1  (Run single experiment)
    python basecode.py --all            (Run all 7 experiments)
"""

import sys
import argparse
from pathlib import Path
from experiments import (
    run_experiment_1, run_experiment_2, run_experiment_3,
    run_experiment_4, run_experiment_5, run_experiment_6,
    run_experiment_7, run_experiment_8, run_experiment_9,
    run_experiment_10, run_all_experiments
)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Post-Quantum Secure Federated Learning Framework'
    )
    parser.add_argument(
        '--experiment',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Run specific experiment (1-10)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=50,
        help='Number of federated learning rounds'
    )
    parser.add_argument(
        '--clients',
        type=int,
        default=10,
        help='Number of clients'
    )
    parser.add_argument(
        '--Byzantine',
        type=int,
        default=1,
        help='Number of Byzantine clients'
    )
    
    args = parser.parse_args()
    
    # Create results directory
    Path('./results').mkdir(exist_ok=True)
    Path('./logs').mkdir(exist_ok=True)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Post-Quantum Secure Federated Learning Framework               ║
    ║  with Byzantine and Data Poisoning Attack Detection             ║
    ║  + Manhattan Distance Defense                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run experiments
    if args.all:
        run_all_experiments()
    elif args.experiment:
        experiment_funcs = {
            1: run_experiment_1,
            2: run_experiment_2,
            3: run_experiment_3,
            4: run_experiment_4,
            5: run_experiment_5,
            6: run_experiment_6,
            7: run_experiment_7,
            8: run_experiment_8,
            9: run_experiment_9,
            10: run_experiment_10,
        }
        experiment_funcs[args.experiment]()
    else:
        print("Please specify an experiment to run:")
        print("  python basecode.py --experiment 1-10  (Run specific experiment)")
        print("  python basecode.py --all               (Run all 10 experiments)")
        print("\nExperiments:")
        print("  1: Clean FL + FedAvg")
        print("  2: Byzantine Attack + FedAvg")
        print("  3: Byzantine Attack + Krum")
        print("  4: Data Poisoning + FedAvg")
        print("  5: Data Poisoning + Krum")
        print("  6: Byzantine Attack + PQC + FedAvg")
        print("  7: Byzantine Attack + PQC + Krum")
        print("  8: Byzantine Attack + Manhattan Distance")
        print("  9: Data Poisoning + Manhattan Distance")
        print("  10: Byzantine Attack + PQC + Manhattan Distance")


if __name__ == '__main__':
    main()
