#!/usr/bin/env python3
"""
Summary Report Generator for Reasoning Behavior Analysis

This script generates a comprehensive PDF report from experiment data, including:
- Basic experimental setup
- Model accuracy metrics
- Thinking mode distribution and accuracy
- Random examples from each thinking mode
- Dual-axis visualization (bar chart for distribution, line chart for accuracy)

Usage:
    python summary.py <data_folder>
    Example: python summary.py ./data/MATH500/deepseek/p2/
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import numpy as np

# Disable LaTeX rendering to avoid parsing errors
matplotlib.rcParams['text.usetex'] = False


def load_data(data_folder: Path) -> Tuple[Dict, Dict, List]:
    """Load metrics, grades, and results data from the specified folder."""
    metrics_path = data_folder / "metrics.json"
    grades_path = data_folder / "grades.json"
    results_path = data_folder / "results.json"

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    with open(grades_path, 'r', encoding='utf-8') as f:
        grades = json.load(f)

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return metrics, grades, results


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
    # The structure is metrics['metrics'] not metrics['items']
    if 'metrics' in metrics:
        for item in metrics['metrics']:
            unique_id = item['unique_id']
            is_correct = grade_map.get(unique_id, False)

            # Use 'strategies_found' field
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


def get_random_correct_examples(metrics: Dict, grades: Dict, results: List,
                                strategy_stats: Dict) -> Dict[str, Dict]:
    """Get one random correct example for each thinking strategy."""
    # Create mappings
    grade_map = {g['unique_id']: g['correct'] for g in grades['grades']}
    result_map = {r['unique_id']: r for r in results}

    # Collect correct examples for each strategy
    strategy_examples = {key: [] for key in strategy_stats.keys()}

    if 'metrics' in metrics:
        for item in metrics['metrics']:
            unique_id = item['unique_id']
            if grade_map.get(unique_id, False):  # Only correct answers
                for strategy_key in item['strategies_found']:
                    strategy_examples[strategy_key].append(unique_id)

    # Select random examples
    selected_examples = {}
    for strategy_key, example_ids in strategy_examples.items():
        if example_ids:
            selected_id = random.choice(example_ids)
            selected_examples[strategy_key] = {
                'unique_id': selected_id,
                'data': result_map.get(selected_id, {})
            }

    return selected_examples


def create_dual_axis_chart(strategy_stats: Dict, output_path: Path):
    """Create a dual-axis chart with bar chart for percentage and line chart for accuracy."""
    # Sort strategies by key for consistent ordering
    sorted_strategies = sorted(strategy_stats.items(), key=lambda x: x[0])

    strategy_keys = [f"{k}\n{v['name']}" for k, v in sorted_strategies]
    percentages = [v['percentage'] for _, v in sorted_strategies]
    accuracies = [v['accuracy'] for _, v in sorted_strategies]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Bar chart for percentage (left y-axis)
    x = np.arange(len(strategy_keys))
    width = 0.6
    bars = ax1.bar(x, percentages, width, alpha=0.7, color='steelblue', label='Distribution %')
    ax1.set_xlabel('Thinking Strategy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distribution Percentage (%)', fontsize=12, fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_keys, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, max(percentages) * 1.2)

    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    # Line chart for accuracy (right y-axis)
    ax2 = ax1.twinx()
    line = ax2.plot(x, accuracies, color='crimson', marker='o', linewidth=2,
                    markersize=8, label='Accuracy %')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(0, 105)

    # Add value labels on line points
    for xi, acc in zip(x, accuracies):
        ax2.text(xi, acc + 2, f'{acc:.1f}%', ha='center', va='bottom',
                fontsize=8, color='crimson')

    # Title and legend
    plt.title('Thinking Strategy Distribution and Accuracy',
              fontsize=14, fontweight='bold', pad=20)

    # Combined legend
    bars_patch = mpatches.Patch(color='steelblue', alpha=0.7, label='Distribution %')
    line_patch = mpatches.Patch(color='crimson', label='Accuracy %')
    ax1.legend(handles=[bars_patch, line_patch], loc='upper left', fontsize=10)

    fig.tight_layout()

    return fig


def sanitize_text(text: str) -> str:
    """Remove or escape special characters that might cause rendering issues."""
    if not isinstance(text, str):
        return str(text)
    # Replace problematic characters
    text = text.replace('\\', '\\\\')
    text = text.replace('$', '\\$')
    return text


def generate_pdf_report(data_folder: Path, output_path: Path = None):
    """Generate a comprehensive PDF report."""
    if output_path is None:
        output_path = data_folder / "summary.pdf"

    # Load data
    print(f"Loading data from {data_folder}...")
    metrics, grades, results = load_data(data_folder)

    # Calculate statistics
    print("Calculating strategy accuracy...")
    strategy_stats = calculate_strategy_accuracy(metrics, grades)

    # Get random examples
    print("Selecting random correct examples...")
    examples = get_random_correct_examples(metrics, grades, results, strategy_stats)

    # Create PDF
    print(f"Generating PDF report to {output_path}...")
    with PdfPages(output_path) as pdf:
        # Page 1: Summary Information
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Reasoning Behavior Analysis Report', fontsize=16, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.axis('off')

        # Basic experimental setup
        setup_text = f"""
EXPERIMENTAL SETUP
{'='*80}
Data Folder: {data_folder}
Total Items: {metrics['summary']['total_items']}
Total Strategies: {len(metrics['summary']['strategies'])}

OVERALL ACCURACY
{'='*80}
Total Problems: {grades['summary']['total']}
Correct: {grades['summary']['correct']}
Incorrect: {grades['summary']['incorrect']}
Overall Accuracy: {grades['summary']['accuracy']:.2f}%

THINKING STRATEGY DISTRIBUTION
{'='*80}
"""

        for strategy_key, stats in sorted(strategy_stats.items()):
            setup_text += f"\n{strategy_key}: {stats['name']}\n"
            setup_text += f"  Distribution: {stats['percentage']:.1f}%\n"
            setup_text += f"  Accuracy: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})\n"

        ax.text(0.05, 0.95, setup_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Visualization
        fig = create_dual_axis_chart(strategy_stats, output_path)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Pages 3+: Examples for each strategy
        for strategy_key, example_data in sorted(examples.items()):
            if not example_data:
                continue

            strategy_name = strategy_stats[strategy_key]['name']
            data = example_data['data']

            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'Example for Strategy {strategy_key}: {strategy_name}',
                        fontsize=14, fontweight='bold')

            ax = fig.add_subplot(111)
            ax.axis('off')

            # Format the response text
            problem = sanitize_text(data.get('problem', 'N/A'))
            solution = sanitize_text(data.get('solution', 'N/A'))
            result = data.get('result', {})
            response = sanitize_text(result.get('answer', 'N/A'))
            cot = sanitize_text(result.get('traj', 'N/A'))

            example_text = f"""
PROBLEM
{'='*80}
{problem[:500]}{'...' if len(problem) > 500 else ''}

GROUND TRUTH ANSWER
{'='*80}
{sanitize_text(data.get('answer', 'N/A'))}

MODEL RESPONSE
{'='*80}
{response[:800]}{'...' if len(response) > 800 else ''}

CHAIN OF THOUGHT (First 1500 chars)
{'='*80}
{cot[:1500]}{'...' if len(cot) > 1500 else ''}
"""

            ax.text(0.05, 0.95, example_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    wrap=True)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Reasoning Behavior Analysis Report'
        d['Author'] = 'Summary Report Generator'
        d['Subject'] = f'Analysis of {data_folder}'
        d['Keywords'] = 'reasoning, thinking strategies, accuracy'

    print(f"✓ Report generated successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a comprehensive PDF report from reasoning behavior data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python summary.py ./data/MATH500/deepseek/p2/
        """
    )
    parser.add_argument('data_folder', type=str,
                       help='Path to the data folder containing metrics.json, grades.json, and results.json')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output PDF path (default: <data_folder>/summary.pdf)')

    args = parser.parse_args()

    data_folder = Path(args.data_folder)

    if not data_folder.exists():
        print(f"Error: Data folder '{data_folder}' does not exist.")
        return

    output_path = Path(args.output) if args.output else None

    generate_pdf_report(data_folder, output_path)


if __name__ == '__main__':
    main()
