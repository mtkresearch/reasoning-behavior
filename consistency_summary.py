import json
import sys
from pathlib import Path
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Disable LaTeX rendering to avoid parsing errors
matplotlib.rcParams['text.usetex'] = False

# All possible methods
all_methods = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
method_names = {
    'A': 'Planning and Execution',
    'B': 'Problem Decomposition',
    'C': 'Formulate Hypotheses and Test Them',
    'D': 'Guess the Answer First, Then Verify',
    'E': 'Option Judgment and Elimination',
    'F': 'Reverse Thinking',
    'G': 'Divergent and Convergent Thinking',
    'H': 'Counterexample Testing',
    'I': 'Association Method'
}

def load_model_data(model_name: str, base_dir: str = 'consistency_data') -> list:
    """Load consistency data for a given model"""
    file_path = Path(base_dir) / f'consistency_{model_name}.json'
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


def get_method_labels(entry, all_methods):
    """Convert method_types list to binary labels for each method"""
    method_types = set(entry['method_types'])
    return [1 if method in method_types else 0 for method in all_methods]


def cohen_kappa(y1, y2):
    """Calculate Cohen's Kappa score manually"""
    n = len(y1)

    # Calculate observed agreement (P_o)
    agreements = sum(1 for a, b in zip(y1, y2) if a == b)
    p_o = agreements / n

    # Calculate expected agreement (P_e)
    # Count YES and NO for each rater
    yes_1 = sum(y1)
    no_1 = n - yes_1
    yes_2 = sum(y2)
    no_2 = n - yes_2

    p_yes_1 = yes_1 / n
    p_no_1 = no_1 / n
    p_yes_2 = yes_2 / n
    p_no_2 = no_2 / n

    p_e = p_yes_1 * p_yes_2 + p_no_1 * p_no_2

    # Calculate kappa
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0

    kappa = (p_o - p_e) / (1 - p_e)

    return kappa, p_o, p_e


def pearson_correlation(y1, y2):
    """Calculate Pearson correlation coefficient"""
    n = len(y1)

    # Calculate means
    mean_1 = sum(y1) / n
    mean_2 = sum(y2) / n

    # Calculate covariance and standard deviations
    cov = sum((a - mean_1) * (b - mean_2) for a, b in zip(y1, y2)) / n
    std_1 = (sum((a - mean_1) ** 2 for a in y1) / n) ** 0.5
    std_2 = (sum((b - mean_2) ** 2 for b in y2) / n) ** 0.5

    # Calculate correlation
    if std_1 == 0 or std_2 == 0:
        return 0.0

    correlation = cov / (std_1 * std_2)

    return correlation

def get_kappa_interpretation(kappa):
    """Get interpretation for Cohen's Kappa value"""
    if kappa < 0:
        return "Worse than random"
    elif kappa < 0.2:
        return "Slight agreement"
    elif kappa < 0.4:
        return "Fair agreement"
    elif kappa < 0.6:
        return "Moderate agreement"
    elif kappa < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def compare_two_models(model1_name: str, model1_data: list, model2_name: str, model2_data: list, verbose: bool = True):
    """Compare two models and return comparison statistics"""

    # Verify same unique_ids
    assert len(model1_data) == len(model2_data), f"Data length mismatch: {len(model1_data)} vs {len(model2_data)}"

    # Collect all labels
    all_model1_labels = []
    all_model2_labels = []

    for entry1, entry2 in zip(model1_data, model2_data):
        assert entry1['unique_id'] == entry2['unique_id'], f"ID mismatch: {entry1['unique_id']} vs {entry2['unique_id']}"

        labels1 = get_method_labels(entry1, all_methods)
        labels2 = get_method_labels(entry2, all_methods)

        all_model1_labels.extend(labels1)
        all_model2_labels.extend(labels2)

    # Calculate overall Kappa and Correlation
    overall_kappa, overall_po, overall_pe = cohen_kappa(all_model1_labels, all_model2_labels)
    overall_corr = pearson_correlation(all_model1_labels, all_model2_labels)

    # Calculate per-method metrics
    per_method_stats = []
    for method in all_methods:
        # Collect labels for this method only
        method_model1_labels = []
        method_model2_labels = []

        for entry1, entry2 in zip(model1_data, model2_data):
            label1 = 1 if method in entry1['method_types'] else 0
            label2 = 1 if method in entry2['method_types'] else 0

            method_model1_labels.append(label1)
            method_model2_labels.append(label2)

        kappa, po, pe = cohen_kappa(method_model1_labels, method_model2_labels)
        corr = pearson_correlation(method_model1_labels, method_model2_labels)

        # Count YES/NO
        model1_yes = sum(method_model1_labels)
        model2_yes = sum(method_model2_labels)
        total = len(method_model1_labels)

        # Confusion matrix
        both_yes = sum(1 for a, b in zip(method_model1_labels, method_model2_labels) if a == 1 and b == 1)
        both_no = sum(1 for a, b in zip(method_model1_labels, method_model2_labels) if a == 0 and b == 0)
        m1_yes_m2_no = sum(1 for a, b in zip(method_model1_labels, method_model2_labels) if a == 1 and b == 0)
        m1_no_m2_yes = sum(1 for a, b in zip(method_model1_labels, method_model2_labels) if a == 0 and b == 1)

        per_method_stats.append({
            'method': method,
            'method_name': method_names[method],
            'model1_yes': model1_yes,
            'model2_yes': model2_yes,
            'total': total,
            'kappa': kappa,
            'po': po,
            'pe': pe,
            'corr': corr,
            'both_yes': both_yes,
            'both_no': both_no,
            'm1_yes_m2_no': m1_yes_m2_no,
            'm1_no_m2_yes': m1_no_m2_yes
        })

    result = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'num_entries': len(model1_data),
        'overall_kappa': overall_kappa,
        'overall_po': overall_po,
        'overall_pe': overall_pe,
        'overall_corr': overall_corr,
        'per_method_stats': per_method_stats
    }

    # Print to console if verbose
    if verbose:
        print("="*80)
        print(f"COMPARISON: {model1_name.upper()} vs {model2_name.upper()}")
        print("="*80)
        print()
        print("OVERALL METRICS (All methods combined)")
        print("-"*80)
        print(f"Total judgments: {len(all_model1_labels)} ({len(model1_data)} entries × 9 methods)")
        print(f"Observed Agreement (P_o): {overall_po:.4f}")
        print(f"Expected Agreement (P_e): {overall_pe:.4f}")
        print(f"Cohen's Kappa: {overall_kappa:.4f}")
        print(f"Pearson Correlation: {overall_corr:.4f}")
        print(f"Kappa Interpretation: {get_kappa_interpretation(overall_kappa)}")
        print()

        print("-"*80)
        print("PER-METHOD METRICS")
        print("-"*80)

        for stats in per_method_stats:
            print(f"\nMethod {stats['method']}: {stats['method_name']}")
            print(f"  {model1_name} YES: {stats['model1_yes']}/{stats['total']} ({stats['model1_yes']/stats['total']*100:.1f}%)")
            print(f"  {model2_name} YES: {stats['model2_yes']}/{stats['total']} ({stats['model2_yes']/stats['total']*100:.1f}%)")
            print(f"  P_o: {stats['po']:.4f}, P_e: {stats['pe']:.4f}")
            print(f"  Cohen's Kappa: {stats['kappa']:.4f}")
            print(f"  Pearson Correlation: {stats['corr']:.4f}")
            print(f"  Agreement: Both YES={stats['both_yes']}, Both NO={stats['both_no']}")
            print(f"  Disagreement: {model1_name}=YES,{model2_name}=NO: {stats['m1_yes_m2_no']}, {model1_name}=NO,{model2_name}=YES: {stats['m1_no_m2_yes']}")

        print("\n" + "="*80 + "\n")

    return result


def create_summary_page(all_results: list):
    """Create a summary page with overall statistics"""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Model Consistency Comparison Summary', fontsize=16, fontweight='bold')

    ax = fig.add_subplot(111)
    ax.axis('off')

    summary_text = "MODEL CONSISTENCY COMPARISON SUMMARY\n"
    summary_text += "=" * 80 + "\n\n"

    for result in all_results:
        m1 = result['model1_name']
        m2 = result['model2_name']
        summary_text += f"Comparison: {m1.upper()} vs {m2.upper()}\n"
        summary_text += "-" * 80 + "\n"
        summary_text += f"Total entries: {result['num_entries']}\n"
        summary_text += f"Overall Cohen's Kappa: {result['overall_kappa']:.4f} ({get_kappa_interpretation(result['overall_kappa'])})\n"
        summary_text += f"Overall Pearson Correlation: {result['overall_corr']:.4f}\n"
        summary_text += f"Observed Agreement (P_o): {result['overall_po']:.4f}\n"
        summary_text += f"Expected Agreement (P_e): {result['overall_pe']:.4f}\n"
        summary_text += "\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    return fig


def create_kappa_comparison_chart(all_results: list):
    """Create a bar chart comparing Cohen's Kappa across method strategies for all model pairs"""
    num_comparisons = len(all_results)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(all_methods))
    width = 0.8 / num_comparisons

    colors = plt.cm.tab10(np.linspace(0, 1, num_comparisons))

    for i, result in enumerate(all_results):
        kappas = [stat['kappa'] for stat in result['per_method_stats']]
        offset = width * (i - num_comparisons / 2 + 0.5)

        label = f"{result['model1_name']} vs {result['model2_name']}"
        bars = ax.bar(x + offset, kappas, width, alpha=0.8, color=colors[i], label=label)

        # Add value labels on bars if not too crowded
        if num_comparisons <= 3:
            for bar, val in zip(bars, kappas):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # Set labels
    method_labels = [f"{method}\n{method_names[method]}" for method in all_methods]
    ax.set_xlabel('Reasoning Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel("Cohen's Kappa", fontsize=12, fontweight='bold')
    ax.set_title("Cohen's Kappa by Reasoning Strategy", fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.5)

    fig.tight_layout()
    return fig


def create_correlation_comparison_chart(all_results: list):
    """Create a bar chart comparing Pearson Correlation across method strategies for all model pairs"""
    num_comparisons = len(all_results)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(all_methods))
    width = 0.8 / num_comparisons

    colors = plt.cm.tab10(np.linspace(0, 1, num_comparisons))

    for i, result in enumerate(all_results):
        corrs = [stat['corr'] for stat in result['per_method_stats']]
        offset = width * (i - num_comparisons / 2 + 0.5)

        label = f"{result['model1_name']} vs {result['model2_name']}"
        bars = ax.bar(x + offset, corrs, width, alpha=0.8, color=colors[i], label=label)

        # Add value labels on bars if not too crowded
        if num_comparisons <= 3:
            for bar, val in zip(bars, corrs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # Set labels
    method_labels = [f"{method}\n{method_names[method]}" for method in all_methods]
    ax.set_xlabel('Reasoning Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax.set_title('Pearson Correlation by Reasoning Strategy', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='black', linewidth=0.5)

    fig.tight_layout()
    return fig


def create_agreement_heatmap(result: dict):
    """Create a heatmap showing agreement rates for each method"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data
    methods = [stat['method'] for stat in result['per_method_stats']]
    data = []

    for stat in result['per_method_stats']:
        total = stat['total']
        agreement_rate = (stat['both_yes'] + stat['both_no']) / total * 100
        both_yes_rate = stat['both_yes'] / total * 100
        both_no_rate = stat['both_no'] / total * 100
        m1_yes_m2_no_rate = stat['m1_yes_m2_no'] / total * 100
        m1_no_m2_yes_rate = stat['m1_no_m2_yes'] / total * 100

        data.append([both_yes_rate, both_no_rate, m1_yes_m2_no_rate, m1_no_m2_yes_rate])

    data = np.array(data)

    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([f"{m}\n{method_names[m]}" for m in methods], fontsize=9)
    ax.set_yticklabels(['Both YES', 'Both NO',
                        f'{result["model1_name"]}=YES\n{result["model2_name"]}=NO',
                        f'{result["model1_name"]}=NO\n{result["model2_name"]}=YES'], fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(4):
            text = ax.text(i, j, f'{data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title(f'Agreement Patterns: {result["model1_name"].upper()} vs {result["model2_name"].upper()}',
                fontsize=12, fontweight='bold', pad=10)

    fig.tight_layout()
    return fig


def create_detailed_stats_page(result: dict):
    """Create a detailed statistics page for a comparison"""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f'Detailed Statistics: {result["model1_name"].upper()} vs {result["model2_name"].upper()}',
                 fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    ax.axis('off')

    m1 = result['model1_name']
    m2 = result['model2_name']

    stats_text = f"OVERALL METRICS\n"
    stats_text += "=" * 80 + "\n"
    stats_text += f"Total entries: {result['num_entries']}\n"
    stats_text += f"Cohen's Kappa: {result['overall_kappa']:.4f} ({get_kappa_interpretation(result['overall_kappa'])})\n"
    stats_text += f"Pearson Correlation: {result['overall_corr']:.4f}\n"
    stats_text += f"Observed Agreement (P_o): {result['overall_po']:.4f}\n"
    stats_text += f"Expected Agreement (P_e): {result['overall_pe']:.4f}\n\n"

    stats_text += "PER-METHOD STATISTICS\n"
    stats_text += "=" * 80 + "\n\n"

    for stat in result['per_method_stats']:
        stats_text += f"Method {stat['method']}: {stat['method_name']}\n"
        stats_text += f"  {m1} YES: {stat['model1_yes']}/{stat['total']} ({stat['model1_yes']/stat['total']*100:.1f}%)\n"
        stats_text += f"  {m2} YES: {stat['model2_yes']}/{stat['total']} ({stat['model2_yes']/stat['total']*100:.1f}%)\n"
        stats_text += f"  Cohen's Kappa: {stat['kappa']:.4f}\n"
        stats_text += f"  Pearson Correlation: {stat['corr']:.4f}\n"
        stats_text += f"  Agreement: Both YES={stat['both_yes']}, Both NO={stat['both_no']}\n"
        stats_text += f"  Disagreement: {m1}=YES,{m2}=NO={stat['m1_yes_m2_no']}, {m1}=NO,{m2}=YES={stat['m1_no_m2_yes']}\n\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace')

    return fig


def generate_pdf_report(all_results: list, output_path: str):
    """Generate a comprehensive PDF report"""
    print(f"Generating PDF report to {output_path}...")

    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        fig = create_summary_page(all_results)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Overall Kappa comparison
        fig = create_kappa_comparison_chart(all_results)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 3: Overall Correlation comparison
        fig = create_correlation_comparison_chart(all_results)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Detailed pages for each comparison
        for result in all_results:
            # Detailed stats page
            fig = create_detailed_stats_page(result)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Agreement heatmap
            fig = create_agreement_heatmap(result)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Model Consistency Comparison Report'
        d['Author'] = 'Consistency Analysis Tool'
        d['Subject'] = 'Comparison of reasoning strategy judgments across models'
        d['Keywords'] = 'consistency, kappa, correlation, reasoning strategies'

    print(f"✓ PDF report generated successfully: {output_path}")


if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python consistency_summary.py [--out output.pdf] model1 model2 [model3 ...]")
        print("Example: python consistency_summary.py deepseek gpt5")
        print("Example: python consistency_summary.py --out report.pdf deepseek gpt5 claude")
        sys.exit(1)

    # Check for --out flag
    output_pdf = None
    args = sys.argv[1:]

    if args[0] == '--out':
        if len(args) < 2:
            print("Error: --out requires a filename")
            sys.exit(1)
        output_pdf = args[1]
        model_names = args[2:]
    else:
        model_names = args

    if len(model_names) < 2:
        print("Error: At least 2 models required for comparison")
        sys.exit(1)

    # Load all model data
    models_data = {}
    print("Loading model data...")
    for model_name in model_names:
        try:
            models_data[model_name] = load_model_data(model_name)
            print(f"  ✓ Loaded {model_name}: {len(models_data[model_name])} entries")
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            sys.exit(1)

    print()

    # Perform pairwise comparisons
    model_pairs = list(combinations(model_names, 2))
    print(f"Performing {len(model_pairs)} pairwise comparison(s)...\n")

    all_results = []
    for model1, model2 in model_pairs:
        result = compare_two_models(model1, models_data[model1], model2, models_data[model2], verbose=True)
        all_results.append(result)

    # Generate PDF report if requested
    if output_pdf:
        generate_pdf_report(all_results, output_pdf)
    else:
        # Auto-generate filename
        if len(model_names) == 2:
            output_pdf = f"consistency_{model_names[0]}_vs_{model_names[1]}.pdf"
        else:
            output_pdf = f"consistency_comparison_{len(model_names)}models.pdf"
        generate_pdf_report(all_results, output_pdf)
