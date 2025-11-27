#!/usr/bin/env python3
"""
Calculate P(shuffle=True|normal=True)

Given two result files (normal and shuffled), calculate the conditional probability
that a problem is answered correctly after shuffling, given it was correct before shuffling.

Usage:
    python calculate_shuffle_conditional_prob.py [normal_file] [shuffle_file]

Default:
    normal_file: data/baseline/remove_answer_after.json
    shuffle_file: data/baseline/remove_answer_after_then_shuffle.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def load_results(file_path: str) -> Dict[str, bool]:
    """
    Load results file and return mapping of unique_id -> is_correct.

    Args:
        file_path: Path to the results JSON file

    Returns:
        Dictionary mapping unique_id to is_correct boolean
    """
    print(f"Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    for item in data:
        unique_id = item['unique_id']
        is_correct = item.get('is_correct', False)
        results[unique_id] = is_correct

    print(f"  Loaded {len(results)} results")
    return results


def calculate_conditional_probability(
    normal_results: Dict[str, bool],
    shuffle_results: Dict[str, bool]
) -> Tuple[float, int, int]:
    """
    Calculate P(shuffle=True|normal=True).

    Args:
        normal_results: Mapping of unique_id -> is_correct for normal (before shuffle)
        shuffle_results: Mapping of unique_id -> is_correct for shuffled

    Returns:
        Tuple of (probability, correct_both_count, normal_correct_count)
    """
    # Find common unique_ids
    common_ids = set(normal_results.keys()) & set(shuffle_results.keys())
    print(f"\nCommon unique_ids: {len(common_ids)}")

    # Count cases where normal=True
    normal_correct = [uid for uid in common_ids if normal_results[uid]]
    print(f"Normal correct (normal=True): {len(normal_correct)}")

    # Count cases where both normal=True and shuffle=True
    both_correct = [
        uid for uid in normal_correct
        if shuffle_results[uid]
    ]
    print(f"Both correct (normal=True AND shuffle=True): {len(both_correct)}")

    # Calculate P(shuffle=True|normal=True)
    if len(normal_correct) == 0:
        print("\nWarning: No cases where normal=True")
        return 0.0, 0, 0

    probability = len(both_correct) / len(normal_correct)

    return probability, len(both_correct), len(normal_correct)


def print_statistics(
    normal_results: Dict[str, bool],
    shuffle_results: Dict[str, bool],
    common_ids: set
):
    """Print additional statistics."""
    print("\n" + "="*60)
    print("Additional Statistics")
    print("="*60)

    # Normal accuracy
    normal_correct_all = sum(1 for uid in common_ids if normal_results[uid])
    normal_accuracy = normal_correct_all / len(common_ids)
    print(f"Normal accuracy: {normal_correct_all}/{len(common_ids)} = {normal_accuracy:.4f}")

    # Shuffle accuracy
    shuffle_correct_all = sum(1 for uid in common_ids if shuffle_results[uid])
    shuffle_accuracy = shuffle_correct_all / len(common_ids)
    print(f"Shuffle accuracy: {shuffle_correct_all}/{len(common_ids)} = {shuffle_accuracy:.4f}")

    # Count all combinations
    both_correct = sum(
        1 for uid in common_ids
        if normal_results[uid] and shuffle_results[uid]
    )
    both_wrong = sum(
        1 for uid in common_ids
        if not normal_results[uid] and not shuffle_results[uid]
    )
    normal_only = sum(
        1 for uid in common_ids
        if normal_results[uid] and not shuffle_results[uid]
    )
    shuffle_only = sum(
        1 for uid in common_ids
        if not normal_results[uid] and shuffle_results[uid]
    )

    print(f"\nConfusion Matrix:")
    print(f"  Both correct:        {both_correct}")
    print(f"  Both wrong:          {both_wrong}")
    print(f"  Normal only correct: {normal_only}")
    print(f"  Shuffle only correct: {shuffle_only}")

    # P(normal=True|shuffle=True)
    if shuffle_correct_all > 0:
        p_normal_given_shuffle = both_correct / shuffle_correct_all
        print(f"\nP(normal=True|shuffle=True) = {p_normal_given_shuffle:.4f}")


def main():
    # Parse arguments or use defaults
    if len(sys.argv) >= 3:
        normal_file = sys.argv[1]
        shuffle_file = sys.argv[2]
    else:
        normal_file = "data/baseline/remove_answer_after.json"
        shuffle_file = "data/baseline/remove_answer_after_then_shuffle.json"

    # Convert to absolute paths
    normal_file = Path(normal_file).resolve()
    shuffle_file = Path(shuffle_file).resolve()

    print("="*60)
    print("Calculate P(shuffle=True|normal=True)")
    print("="*60)
    print(f"Normal file:  {normal_file}")
    print(f"Shuffle file: {shuffle_file}")
    print()

    # Load results
    normal_results = load_results(normal_file)
    shuffle_results = load_results(shuffle_file)

    # Calculate conditional probability
    prob, both_correct, normal_correct = calculate_conditional_probability(
        normal_results,
        shuffle_results
    )

    # Print main result
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"P(shuffle=True|normal=True) = {both_correct}/{normal_correct} = {prob:.4f}")
    print(f"                             = {prob*100:.2f}%")

    # Print additional statistics
    common_ids = set(normal_results.keys()) & set(shuffle_results.keys())
    print_statistics(normal_results, shuffle_results, common_ids)

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
