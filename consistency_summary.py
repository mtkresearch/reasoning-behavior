import json

# Load data
deepseek = json.load(open('consistency_data/consistency_deepseek.json'))
gpt5 = json.load(open('consistency_data/consistency_gpt5.json'))

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

# Convert method_types to YES/NO for each method
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

# Collect all labels
all_ds_labels = []
all_gpt_labels = []

for ds_entry, gpt_entry in zip(deepseek, gpt5):
    assert ds_entry['unique_id'] == gpt_entry['unique_id']

    ds_labels = get_method_labels(ds_entry, all_methods)
    gpt_labels = get_method_labels(gpt_entry, all_methods)

    all_ds_labels.extend(ds_labels)
    all_gpt_labels.extend(gpt_labels)

# Calculate overall Kappa and Correlation
overall_kappa, overall_po, overall_pe = cohen_kappa(all_ds_labels, all_gpt_labels)
overall_corr = pearson_correlation(all_ds_labels, all_gpt_labels)

print("="*80)
print("OVERALL METRICS (All methods combined)")
print("="*80)
print(f"Total judgments: {len(all_ds_labels)} (2000 entries × 9 methods)")
print(f"Observed Agreement (P_o): {overall_po:.4f}")
print(f"Expected Agreement (P_e): {overall_pe:.4f}")
print(f"Cohen's Kappa: {overall_kappa:.4f}")
print(f"Pearson Correlation: {overall_corr:.4f}")
print()

# Interpret kappa
if overall_kappa < 0:
    interpretation = "Worse than random"
elif overall_kappa < 0.2:
    interpretation = "Slight agreement"
elif overall_kappa < 0.4:
    interpretation = "Fair agreement"
elif overall_kappa < 0.6:
    interpretation = "Moderate agreement"
elif overall_kappa < 0.8:
    interpretation = "Substantial agreement"
else:
    interpretation = "Almost perfect agreement"

print(f"Kappa Interpretation: {interpretation}")
print()

# Calculate per-method Kappa and Correlation
print("="*80)
print("PER-METHOD METRICS")
print("="*80)

for method in all_methods:
    # Collect labels for this method only by iterating through entries
    method_ds_labels = []
    method_gpt_labels = []

    for ds_entry, gpt_entry in zip(deepseek, gpt5):
        assert ds_entry['unique_id'] == gpt_entry['unique_id']

        # Check if this method is in method_types
        ds_label = 1 if method in ds_entry['method_types'] else 0
        gpt_label = 1 if method in gpt_entry['method_types'] else 0

        method_ds_labels.append(ds_label)
        method_gpt_labels.append(gpt_label)

    kappa, po, pe = cohen_kappa(method_ds_labels, method_gpt_labels)
    corr = pearson_correlation(method_ds_labels, method_gpt_labels)

    # Count YES/NO
    ds_yes = sum(method_ds_labels)
    gpt_yes = sum(method_gpt_labels)

    print(f"\nMethod {method}: {method_names[method]}")
    print(f"  DeepSeek YES: {ds_yes}/2000 ({ds_yes/20:.1f}%)")
    print(f"  GPT-5 YES:    {gpt_yes}/2000 ({gpt_yes/20:.1f}%)")
    print(f"  P_o: {po:.4f}, P_e: {pe:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print(f"  Pearson Correlation: {corr:.4f}")

    # Confusion matrix
    both_yes = sum(1 for a, b in zip(method_ds_labels, method_gpt_labels) if a == 1 and b == 1)
    both_no = sum(1 for a, b in zip(method_ds_labels, method_gpt_labels) if a == 0 and b == 0)
    ds_yes_gpt_no = sum(1 for a, b in zip(method_ds_labels, method_gpt_labels) if a == 1 and b == 0)
    ds_no_gpt_yes = sum(1 for a, b in zip(method_ds_labels, method_gpt_labels) if a == 0 and b == 1)

    print(f"  Agreement: Both YES={both_yes}, Both NO={both_no}")
    print(f"  Disagreement: DS=YES,GPT=NO: {ds_yes_gpt_no}, DS=NO,GPT=YES: {ds_no_gpt_yes}")

print("\n" + "="*80)
