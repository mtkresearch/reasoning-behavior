#!/usr/bin/env python3
"""
Causal Behavior Analysis Script

This script analyzes the causal effect of reasoning behaviors on accuracy.
Given that Model, Prompt, and Problem are fixed (repeated 10 times per problem),
we can estimate the causal effect as:

    Effect(Behavior X) = P(ACC | Behavior X used) - P(ACC | Behavior X not used)

This controls for:
- Problem difficulty (fixed within each problem)
- Model capability (same model throughout)
- Prompt (same prompt throughout)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics


def load_data(data_dir: Path) -> Tuple[List[dict], Dict[str, bool], Dict[str, List[str]]]:
    """
    Load results, grades, and metrics data.

    Returns:
        results: List of result items
        grades: Dict mapping unique_id -> correctness
        metrics: Dict mapping unique_id -> list of behaviors found
    """
    with open(data_dir / "results.json") as f:
        results = json.load(f)

    with open(data_dir / "grades.json") as f:
        grades_data = json.load(f)
        grades = {g['unique_id']: g['correct'] for g in grades_data['grades']}

    with open(data_dir / "metrics.json") as f:
        metrics_data = json.load(f)
        metrics = {m['unique_id']: m['strategies_found'] for m in metrics_data['metrics']}

    return results, grades, metrics


def extract_problem_id(unique_id: str) -> str:
    """Extract problem ID from unique_id (e.g., 'aime2025-I-0-0' -> 'aime2025-I-0')"""
    return '-'.join(unique_id.split('-')[:-1])


def calculate_behavior_effect_per_problem(
    problem_id: str,
    unique_ids: List[str],
    grades: Dict[str, bool],
    metrics: Dict[str, List[str]],
    behavior: str
) -> Dict[str, float]:
    """
    Calculate the effect of a specific behavior on a specific problem.

    Returns:
        Dict with keys: 'used_acc', 'not_used_acc', 'effect', 'used_count', 'not_used_count'
    """
    used_correct = []
    not_used_correct = []

    for uid in unique_ids:
        behaviors_found = metrics.get(uid, [])
        is_correct = grades.get(uid, False)

        if behavior in behaviors_found:
            used_correct.append(1 if is_correct else 0)
        else:
            not_used_correct.append(1 if is_correct else 0)

    # Calculate accuracies
    used_acc = statistics.mean(used_correct) if used_correct else None
    not_used_acc = statistics.mean(not_used_correct) if not_used_correct else None

    # Calculate effect (only if both groups have data)
    if used_acc is not None and not_used_acc is not None:
        effect = used_acc - not_used_acc
    else:
        effect = None

    return {
        'used_acc': used_acc,
        'not_used_acc': not_used_acc,
        'effect': effect,
        'used_count': len(used_correct),
        'not_used_count': len(not_used_correct),
        'problem_id': problem_id
    }


def calculate_overall_behavior_effect(
    all_results: Dict[str, Dict],
    behavior: str
) -> Dict[str, float]:
    """
    Aggregate the behavior effect across all problems.

    Returns:
        Dict with overall statistics
    """
    effects = []
    used_accs = []
    not_used_accs = []
    valid_problems = 0

    for problem_id, result in all_results.items():
        if result['effect'] is not None:
            effects.append(result['effect'])
            used_accs.append(result['used_acc'])
            not_used_accs.append(result['not_used_acc'])
            valid_problems += 1

    if not effects:
        return {
            'behavior': behavior,
            'mean_effect': None,
            'median_effect': None,
            'std_effect': None,
            'mean_used_acc': None,
            'mean_not_used_acc': None,
            'valid_problems': 0,
            'total_problems': len(all_results)
        }

    return {
        'behavior': behavior,
        'mean_effect': statistics.mean(effects),
        'median_effect': statistics.median(effects),
        'std_effect': statistics.stdev(effects) if len(effects) > 1 else 0.0,
        'mean_used_acc': statistics.mean(used_accs),
        'mean_not_used_acc': statistics.mean(not_used_accs),
        'valid_problems': valid_problems,
        'total_problems': len(all_results)
    }


def analyze_all_behaviors(
    results: List[dict],
    grades: Dict[str, bool],
    metrics: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """
    Analyze causal effects for all behaviors.

    Returns:
        Dict mapping behavior -> effect statistics
    """
    # Group by problem
    problem_groups = defaultdict(list)
    for item in results:
        problem_id = extract_problem_id(item['unique_id'])
        problem_groups[problem_id].append(item['unique_id'])

    # Get all unique behaviors
    all_behaviors = set()
    for behaviors_list in metrics.values():
        all_behaviors.update(behaviors_list)

    # Get behavior names from metrics.json summary
    behavior_names = {}

    # Calculate effects for each behavior
    behavior_effects = {}

    for behavior in sorted(all_behaviors):
        problem_effects = {}

        for problem_id, unique_ids in problem_groups.items():
            result = calculate_behavior_effect_per_problem(
                problem_id, unique_ids, grades, metrics, behavior
            )
            problem_effects[problem_id] = result

        overall_effect = calculate_overall_behavior_effect(problem_effects, behavior)

        behavior_effects[behavior] = {
            'overall': overall_effect,
            'per_problem': problem_effects
        }

    return behavior_effects


def get_behavior_names(data_dir: Path) -> Dict[str, str]:
    """Load behavior names from metrics.json summary"""
    with open(data_dir / "metrics.json") as f:
        metrics_data = json.load(f)
        return {
            code: info['name']
            for code, info in metrics_data['summary']['strategies'].items()
        }


def print_summary(behavior_effects: Dict, behavior_names: Dict[str, str]):
    """Print a summary of behavior effects"""
    print("=" * 80)
    print("CAUSAL BEHAVIOR EFFECT ANALYSIS")
    print("=" * 80)
    print()
    print("Effect = P(Correct | Behavior Used) - P(Correct | Behavior Not Used)")
    print()
    print("-" * 80)
    print(f"{'Code':<5} {'Behavior Name':<40} {'Effect':<10} {'Used Acc':<10} {'Not Used Acc':<12} {'N Problems':<10}")
    print("-" * 80)

    # Sort by effect size (descending)
    sorted_behaviors = sorted(
        behavior_effects.items(),
        key=lambda x: x[1]['overall']['mean_effect'] if x[1]['overall']['mean_effect'] is not None else -999,
        reverse=True
    )

    for behavior, data in sorted_behaviors:
        overall = data['overall']
        behavior_name = behavior_names.get(behavior, "Unknown")

        if overall['mean_effect'] is not None:
            effect_str = f"{overall['mean_effect']:+.4f}"
            used_acc_str = f"{overall['mean_used_acc']:.4f}"
            not_used_acc_str = f"{overall['mean_not_used_acc']:.4f}"
            n_problems = f"{overall['valid_problems']}/{overall['total_problems']}"
        else:
            effect_str = "N/A"
            used_acc_str = "N/A"
            not_used_acc_str = "N/A"
            n_problems = f"0/{overall['total_problems']}"

        print(f"{behavior:<5} {behavior_name:<40} {effect_str:<10} {used_acc_str:<10} {not_used_acc_str:<12} {n_problems:<10}")

    print("-" * 80)
    print()


def save_detailed_results(behavior_effects: Dict, behavior_names: Dict[str, str], output_path: Path):
    """Save detailed results to JSON"""
    output = {}

    for behavior, data in behavior_effects.items():
        behavior_name = behavior_names.get(behavior, "Unknown")
        output[behavior] = {
            'name': behavior_name,
            'overall': data['overall'],
            'per_problem': data['per_problem']
        }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze causal effects of reasoning behaviors on accuracy"
    )
    parser.add_argument(
        'data_dir',
        type=Path,
        help='Directory containing results.json, grades.json, and metrics.json'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for detailed results JSON (default: <data_dir>/causal_analysis.json)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.data_dir}")
    results, grades, metrics = load_data(args.data_dir)
    behavior_names = get_behavior_names(args.data_dir)

    print(f"Loaded {len(results)} instances")
    print()

    # Analyze behaviors
    print("Analyzing causal effects...")
    behavior_effects = analyze_all_behaviors(results, grades, metrics)

    # Print summary
    print_summary(behavior_effects, behavior_names)

    # Save detailed results
    output_path = args.output or (args.data_dir / "causal_analysis.json")
    save_detailed_results(behavior_effects, behavior_names, output_path)


if __name__ == '__main__':
    main()
