#!/usr/bin/env python3
"""
Benchmark Comparison Tool for Reasoning Behavior Analysis

This script generates comparative visualizations from multiple experiment directories,
allowing side-by-side comparison of different experimental runs.

Usage:
    python benchmark.py --out benchmark_fig.pdf dir1 dir2 dir3 ...
    Example: python benchmark.py --out comparison.pdf data/AIME2025__R10/deepseek/p1 data/AIME2025__R10/deepseek/p2
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Disable LaTeX rendering to avoid parsing errors
matplotlib.rcParams['text.usetex'] = False


def load_data(data_folder: Path) -> Tuple[Dict, Dict]:
    """Load metrics and grades data from the specified folder."""
    metrics_path = data_folder / "metrics.json"
    grades_path = data_folder / "grades.json"

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    with open(grades_path, 'r', encoding='utf-8') as f:
        grades = json.load(f)

    return metrics, grades


def calculate_strategy_accuracy(metrics: Dict, grades: Dict) -> Dict[str, Dict]:
    """Calculate accuracy for each thinking strategy."""
    strategy_stats = {}

    # Create a mapping from unique_id to correctness
    grade_map = {g['unique_id']: g['correct'] for g in grades['grades']}

    # Initialize strategy counters
    for strategy_key, strategy_info in metrics['summary']['strategies'].items():
        strategy_stats[strategy_key] = {
            'name': strategy_info['name'],
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'percentage': strategy_info['percentage']
        }

    # Count correct answers for each strategy
    if 'metrics' in metrics:
        for item in metrics['metrics']:
            unique_id = item['unique_id']
            is_correct = grade_map.get(unique_id, False)

            for strategy_key in item['strategies_found']:
                strategy_stats[strategy_key]['total'] += 1
                if is_correct:
                    strategy_stats[strategy_key]['correct'] += 1

    # Calculate accuracy
    for strategy_key in strategy_stats:
        total = strategy_stats[strategy_key]['total']
        if total > 0:
            strategy_stats[strategy_key]['accuracy'] = (
                strategy_stats[strategy_key]['correct'] / total * 100
            )

    return strategy_stats


def load_all_experiments(data_folders: List[Path]) -> Dict[str, Dict]:
    """Load data from all experiment folders."""
    all_data = {}

    for folder in data_folders:
        folder_name = str(folder)
        print(f"Loading data from {folder_name}...")

        try:
            metrics, grades = load_data(folder)
            strategy_stats = calculate_strategy_accuracy(metrics, grades)

            all_data[folder_name] = {
                'metrics': metrics,
                'grades': grades,
                'strategy_stats': strategy_stats,
                'overall_accuracy': grades['summary']['accuracy']
            }
        except Exception as e:
            print(f"Warning: Failed to load {folder_name}: {e}")

    return all_data


def create_comparison_chart(all_data: Dict[str, Dict], chart_type: str = 'distribution'):
    """Create a comparison chart for all experiments.

    Args:
        all_data: Dictionary mapping folder names to their data
        chart_type: Either 'distribution' or 'accuracy'
    """
    # Get all unique strategy keys from all experiments
    all_strategy_keys = set()
    for data in all_data.values():
        all_strategy_keys.update(data['strategy_stats'].keys())

    sorted_strategy_keys = sorted(all_strategy_keys)

    # Prepare data for plotting
    num_experiments = len(all_data)
    x = np.arange(len(sorted_strategy_keys))
    width = 0.8 / num_experiments  # Divide available space among experiments

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, num_experiments))

    # Plot bars for each experiment
    for i, (folder_name, data) in enumerate(all_data.items()):
        strategy_stats = data['strategy_stats']

        if chart_type == 'distribution':
            values = [strategy_stats.get(key, {}).get('percentage', 0)
                     for key in sorted_strategy_keys]
            ylabel = 'Distribution Percentage (%)'
            title = 'Thinking Strategy Distribution Comparison'
        else:  # accuracy
            values = [strategy_stats.get(key, {}).get('accuracy', 0)
                     for key in sorted_strategy_keys]
            ylabel = 'Accuracy (%)'
            title = 'Thinking Strategy Accuracy Comparison'

        offset = width * (i - num_experiments / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, alpha=0.8, color=colors[i],
                     label=folder_name)

        # Add value labels on bars (only if not too crowded)
        if num_experiments <= 3:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    # Set labels and title
    strategy_labels = []
    for key in sorted_strategy_keys:
        # Try to get strategy name from any experiment that has it
        name = key
        for data in all_data.values():
            if key in data['strategy_stats']:
                name = data['strategy_stats'][key]['name']
                break
        strategy_labels.append(f"{key}\n{name}")

    ax.set_xlabel('Thinking Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=8, ncol=1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()
    return fig


def create_overall_accuracy_chart(all_data: Dict[str, Dict]):
    """Create a bar chart comparing overall accuracy across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))

    folder_names = list(all_data.keys())
    accuracies = [data['overall_accuracy'] for data in all_data.values()]

    x = np.arange(len(folder_names))
    bars = ax.bar(x, accuracies, alpha=0.8, color='steelblue')

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(folder_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()
    return fig


def create_dual_axis_comparison_chart(all_data: Dict[str, Dict]):
    """Create a dual-axis chart comparing distribution (bars) and accuracy (lines) for all experiments."""
    # Get all unique strategy keys from all experiments
    all_strategy_keys = set()
    for data in all_data.values():
        all_strategy_keys.update(data['strategy_stats'].keys())

    sorted_strategy_keys = sorted(all_strategy_keys)

    # Prepare data
    num_experiments = len(all_data)
    x = np.arange(len(sorted_strategy_keys))
    width = 0.8 / num_experiments

    # Create figure with dual axis
    fig, ax1 = plt.subplots(figsize=(16, 9))

    # Color schemes
    bar_colors = plt.cm.Set2(np.linspace(0, 1, num_experiments))
    line_colors = plt.cm.Set1(np.linspace(0, 1, num_experiments))
    line_markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h']

    # Plot bars for distribution (left y-axis)
    bar_handles = []
    for i, (folder_name, data) in enumerate(all_data.items()):
        strategy_stats = data['strategy_stats']
        percentages = [strategy_stats.get(key, {}).get('percentage', 0)
                      for key in sorted_strategy_keys]

        offset = width * (i - num_experiments / 2 + 0.5)
        bars = ax1.bar(x + offset, percentages, width, alpha=0.7,
                      color=bar_colors[i], label=f'{folder_name} (Dist)')
        bar_handles.append(bars)

    ax1.set_xlabel('Thinking Strategy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distribution Percentage (%)', fontsize=12, fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Plot lines for accuracy (right y-axis)
    ax2 = ax1.twinx()
    line_handles = []
    for i, (folder_name, data) in enumerate(all_data.items()):
        strategy_stats = data['strategy_stats']
        accuracies = [strategy_stats.get(key, {}).get('accuracy', 0)
                     for key in sorted_strategy_keys]

        marker = line_markers[i % len(line_markers)]
        line = ax2.plot(x, accuracies, color=line_colors[i], marker=marker,
                       linewidth=2.5, markersize=8, label=f'{folder_name} (Acc)',
                       alpha=0.9)
        line_handles.extend(line)

    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(0, 105)

    # Set x-axis labels
    strategy_labels = []
    for key in sorted_strategy_keys:
        name = key
        for data in all_data.values():
            if key in data['strategy_stats']:
                name = data['strategy_stats'][key]['name']
                break
        strategy_labels.append(f"{key}\n{name}")

    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_labels, rotation=45, ha='right', fontsize=9)

    # Title
    plt.title('Thinking Strategy Distribution and Accuracy Comparison',
             fontsize=14, fontweight='bold', pad=20)

    # Combined legend
    all_handles = []
    all_labels = []
    for i, (folder_name, _) in enumerate(all_data.items()):
        # Add distribution handle
        dist_patch = mpatches.Patch(color=bar_colors[i], alpha=0.7,
                                    label=f'{folder_name} (Dist %)')
        all_handles.append(dist_patch)
        all_labels.append(f'{folder_name} (Dist %)')

        # Add accuracy handle
        acc_patch = mpatches.Patch(color=line_colors[i],
                                   label=f'{folder_name} (Acc %)')
        all_handles.append(acc_patch)
        all_labels.append(f'{folder_name} (Acc %)')

    ax1.legend(handles=all_handles, labels=all_labels, loc='upper left',
              fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


def create_summary_page(all_data: Dict[str, Dict]):
    """Create a summary page with key statistics."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Benchmark Comparison Summary', fontsize=16, fontweight='bold')

    ax = fig.add_subplot(111)
    ax.axis('off')

    summary_text = "EXPERIMENT COMPARISON SUMMARY\n"
    summary_text += "=" * 80 + "\n\n"

    for folder_name, data in all_data.items():
        summary_text += f"Experiment: {folder_name}\n"
        summary_text += "-" * 80 + "\n"
        summary_text += f"Overall Accuracy: {data['overall_accuracy']:.2f}%\n"
        summary_text += f"Total Problems: {data['grades']['summary']['total']}\n"
        summary_text += f"Correct: {data['grades']['summary']['correct']}\n"
        summary_text += f"Incorrect: {data['grades']['summary']['incorrect']}\n\n"

        summary_text += "Strategy Distribution:\n"
        for strategy_key, stats in sorted(data['strategy_stats'].items()):
            summary_text += f"  {strategy_key}: {stats['percentage']:.1f}% "
            summary_text += f"(Acc: {stats['accuracy']:.2f}%)\n"
        summary_text += "\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    return fig


def generate_benchmark_report(data_folders: List[Path], output_path: Path):
    """Generate a comprehensive benchmark comparison PDF report."""
    # Load all experiment data
    all_data = load_all_experiments(data_folders)

    if not all_data:
        print("Error: No valid data loaded from any folder.")
        return

    print(f"Generating benchmark comparison report to {output_path}...")

    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        fig = create_summary_page(all_data)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Overall accuracy comparison
        fig = create_overall_accuracy_chart(all_data)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 3: Dual-axis comparison (Distribution + Accuracy)
        fig = create_dual_axis_comparison_chart(all_data)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 4: Distribution comparison
        fig = create_comparison_chart(all_data, chart_type='distribution')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 5: Accuracy comparison
        fig = create_comparison_chart(all_data, chart_type='accuracy')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Benchmark Comparison Report'
        d['Author'] = 'Benchmark Comparison Tool'
        d['Subject'] = 'Comparison of multiple reasoning behavior experiments'
        d['Keywords'] = 'reasoning, benchmark, comparison, thinking strategies'

    print(f"✓ Benchmark report generated successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark comparison report from multiple experiment directories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python benchmark.py --out comparison.pdf data/AIME2025__R10/deepseek/p1 data/AIME2025__R10/deepseek/p2
        """
    )
    parser.add_argument('data_folders', type=str, nargs='+',
                       help='Paths to data folders containing metrics.json and grades.json')
    parser.add_argument('--out', type=str, required=True,
                       help='Output PDF path')

    args = parser.parse_args()

    data_folders = [Path(folder) for folder in args.data_folders]

    # Validate all folders exist
    for folder in data_folders:
        if not folder.exists():
            print(f"Error: Data folder '{folder}' does not exist.")
            return

    output_path = Path(args.out)

    generate_benchmark_report(data_folders, output_path)


if __name__ == '__main__':
    main()
