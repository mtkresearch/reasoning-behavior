#!/usr/bin/env python3
"""
Print metrics from experiment results in experiments/ directory.

Output format: model/dataset/flow/acc/correct/success

This script walks through all experiment directories and extracts metrics from results.json files.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


def load_results(results_path: Path) -> Optional[Dict[str, Any]]:
    """Load results.json file and return parsed data."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {results_path}: {e}", file=sys.stderr)
        return None


def extract_metrics(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract metrics from results data."""
    try:
        metadata = results.get('experiment_metadata', {})
        summary = results.get('summary', {})

        # Handle both old and new format
        # Old format uses 'successful', new format uses 'generation_successful'
        success = summary.get('generation_successful', summary.get('successful', 0))

        return {
            'model': metadata.get('model_type', 'unknown'),
            'dataset': metadata.get('dataset', 'unknown'),
            'flow': metadata.get('flow', 'unknown'),
            'acc': summary.get('accuracy', 0.0),
            'correct': summary.get('correct', 0),
            'success': success,
            'max_instances': summary.get('total_questions', 0),
        }
    except Exception as e:
        return None


def find_all_results(base_dir: str = 'experiments') -> list:
    """Find all results.json files in the experiments directory."""
    results_files = []
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory {base_dir} does not exist")
        return results_files

    # Walk through all subdirectories
    for results_path in base_path.rglob('results.json'):
        results_files.append(results_path)

    return sorted(results_files)


def print_metrics(base_dir: str = 'experiments', output_format: str = 'line'):
    """
    Print metrics from all experiment results.

    Args:
        base_dir: Base directory to search for results.json files
        output_format: Output format ('line' or 'table')
    """
    results_files = find_all_results(base_dir)

    if not results_files:
        print(f"No results.json files found in {base_dir}")
        return

    # Print header
    if output_format == 'table':
        print(f"{'Model':<15} {'Dataset':<25} {'Flow':<40} {'Acc':>8} {'Correct':>8} {'Success':>8} {'MaxInst':>8}")
        print("-" * 130)
    else:
        print("model/dataset/flow/acc/correct/success/max_instances")

    # Process each results file
    for results_path in results_files:
        results = load_results(results_path)
        if results is None:
            continue

        metrics = extract_metrics(results)
        if metrics is None:
            continue

        # Print metrics
        if output_format == 'table':
            print(f"{metrics['model']:<15} {metrics['dataset']:<25} {metrics['flow']:<40} "
                  f"{metrics['acc']:>8.4f} {metrics['correct']:>8} {metrics['success']:>8} {metrics['max_instances']:>8}")
        else:
            # Line format: model/dataset/flow/acc/correct/success/max_instances
            print(f"{metrics['model']}/{metrics['dataset']}/{metrics['flow']}/"
                  f"{metrics['acc']:.4f}/{metrics['correct']}/{metrics['success']}/{metrics['max_instances']}")


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Print metrics from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print all metrics in line format
  python run_print_metrics.py

  # Print in table format
  python run_print_metrics.py --format table

  # Specify custom experiments directory
  python run_print_metrics.py --dir experiments/exp/
"""
    )
    parser.add_argument(
        '--dir',
        default='experiments',
        help='Base directory to search for results.json files (default: experiments)'
    )
    parser.add_argument(
        '--format',
        choices=['line', 'table'],
        default='line',
        help='Output format: line or table (default: line)'
    )

    args = parser.parse_args()

    print_metrics(base_dir=args.dir, output_format=args.format)
