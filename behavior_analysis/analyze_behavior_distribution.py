#!/usr/bin/env python3
"""
Analyze behavior count distribution in metrics.json files
"""
import json
import sys
from collections import Counter
from pathlib import Path


def analyze_behavior_distribution(metrics_file):
    """Analyze and display behavior count distribution from metrics.json"""

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    # Count strategies (behaviors) per item
    behavior_counts = []
    for item in data['metrics']:
        if 'strategies_found' in item:
            behavior_counts.append(len(item['strategies_found']))

    # Get distribution
    distribution = Counter(behavior_counts)

    print(f"\nAnalyzing: {metrics_file}")
    print("=" * 70)
    print(f"{'Behavior Count':<20} {'Frequency':<15} {'Percentage'}")
    print("-" * 70)

    total = len(behavior_counts)
    for count in sorted(distribution.keys()):
        freq = distribution[count]
        pct = (freq / total * 100) if total > 0 else 0
        print(f"{count:<20} {freq:<15} {pct:.2f}%")

    print("-" * 70)
    print(f"Total items: {total}")

    if behavior_counts:
        print(f"\nStatistics:")
        print(f"  Min:  {min(behavior_counts)}")
        print(f"  Max:  {max(behavior_counts)}")
        print(f"  Mean: {sum(behavior_counts)/len(behavior_counts):.2f}")
        print(f"  Median: {sorted(behavior_counts)[len(behavior_counts)//2]}")

    return distribution, behavior_counts


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_behavior_distribution.py <metrics.json> [metrics2.json ...]")
        print("\nExample:")
        print("  python analyze_behavior_distribution.py data/AIME2025__R10/deepseek/p1/metrics.json")
        print("  python analyze_behavior_distribution.py data/*/deepseek/p1/metrics.json")
        sys.exit(1)

    for metrics_file in sys.argv[1:]:
        try:
            analyze_behavior_distribution(metrics_file)
            print()
        except FileNotFoundError:
            print(f"Error: File not found: {metrics_file}")
        except KeyError as e:
            print(f"Error: Missing key {e} in {metrics_file}")
        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")


if __name__ == "__main__":
    main()
