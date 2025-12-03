#!/usr/bin/env python3
"""
Noise Impact Analysis Tool

This script analyzes the impact of noise answers on the distribution by calculating
statistics about how noise answers rank and their probabilities.

Usage:
    python run_noise_analysis.py \\
        --result exp/insert_fix_sentence_Thus_answer_123_count_100_of_answer/shuffle_word/answer_retrieval/distributions.json \\
        --noise_answer 123

Features:
    - Average noise answer probability across all questions
    - Expected rank (mean) of noise answer
    - Standard deviation of noise answer rank
    - Per-question noise analysis
"""

import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from logger_config import setup_logger

# Setup logger
logger = setup_logger(__name__, log_file='logs/run_noise_analysis.log')


# =============================================================================
# Answer Normalization (copied from run_distribution.py)
# =============================================================================

def normalize_answer(answer: str) -> Optional[str]:
    """
    Normalize mathematical answer using sympy for equivalence checking.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string, or original if normalization fails
    """
    import sympy as sp
    import re

    if not answer:
        return None

    # Strip whitespace
    answer = answer.strip()

    try:
        # Try to parse as sympy expression
        # Handle LaTeX fractions
        answer_processed = answer.replace('\\frac', 'Rational')
        answer_processed = re.sub(r'Rational\{(\d+)\}\{(\d+)\}', r'Rational(\1, \2)', answer_processed)

        expr = sp.sympify(answer_processed)

        # Simplify and convert to string
        simplified = sp.simplify(expr)

        # Convert floats to integers if they are whole numbers
        if simplified.is_Float:
            if simplified % 1 == 0:
                normalized = str(int(simplified))
            else:
                normalized = str(simplified)
        else:
            normalized = str(simplified)

        return normalized

    except (sp.SympifyError, ValueError, TypeError, AttributeError) as e:
        # If parsing fails, return original string
        logger.debug(f"Failed to normalize '{answer}': {e}")
        return answer


# =============================================================================
# Noise Analysis Functions
# =============================================================================

def analyze_noise_in_distribution(
    distribution: List[Dict],
    noise_answer: str
) -> Tuple[Optional[float], Optional[int]]:
    """
    Analyze noise answer in a single question's distribution.

    Args:
        distribution: Distribution list for one question
        noise_answer: The noise answer to look for (normalized)

    Returns:
        Tuple of (noise_probability, noise_rank)
        Returns (0.0, max_rank+1) if noise not found in distribution
        Returns (None, None) if distribution is empty
    """
    if not distribution:
        return None, None

    # Normalize noise answer for comparison
    normalized_noise = normalize_answer(noise_answer)

    # Find noise answer in distribution
    noise_prob = None
    noise_rank = None
    max_rank = 0

    for entry in distribution:
        rank = entry.get('rank', 0)
        max_rank = max(max_rank, rank)

        normalized_entry_answer = entry.get('normalized_answer')
        if normalized_entry_answer == normalized_noise:
            noise_prob = entry.get('percentage', 0.0)
            noise_rank = rank
            break

    # If noise not found, assign probability 0 and rank = max_rank + 1
    if noise_prob is None:
        noise_prob = 0.0
        noise_rank = max_rank + 1

    return noise_prob, noise_rank


def calculate_noise_statistics(results: List[Dict], noise_answer: str) -> Dict:
    """
    Calculate comprehensive noise answer statistics from distribution results.

    Computes:
    - avg_noise_prob: Average probability of noise answer across all questions
    - mean_noise_rank: Average rank of noise answer
    - std_noise_rank: Standard deviation of noise answer ranks
    - noise_appearance_rate: Proportion of questions where noise appears in distribution
    - questions_with_noise: Number of questions where noise appears in distribution

    Args:
        results: List of result dicts, each containing a 'distribution' field
        noise_answer: The noise answer to analyze (e.g., "123")

    Returns:
        Dictionary with noise statistics
    """
    noise_probs = []
    noise_ranks = []
    questions_with_noise = 0

    for result in results:
        distribution = result.get('distribution', [])

        # Skip empty distributions
        if not distribution:
            continue

        noise_prob, noise_rank = analyze_noise_in_distribution(distribution, noise_answer)

        if noise_prob is None or noise_rank is None:
            continue

        noise_probs.append(noise_prob)
        noise_ranks.append(noise_rank)

        # Count if noise appears in distribution (prob > 0)
        if noise_prob > 0:
            questions_with_noise += 1

    # Calculate averages
    avg_noise_prob = sum(noise_probs) / len(noise_probs) if noise_probs else None

    # Calculate rank statistics
    if noise_ranks:
        mean_noise_rank = sum(noise_ranks) / len(noise_ranks)

        if len(noise_ranks) == 1:
            std_noise_rank = 0.0
        else:
            variance = sum((r - mean_noise_rank) ** 2 for r in noise_ranks) / len(noise_ranks)
            std_noise_rank = math.sqrt(variance)

        # Calculate noise appearance rate
        noise_appearance_rate = questions_with_noise / len(noise_ranks)
    else:
        mean_noise_rank = None
        std_noise_rank = None
        noise_appearance_rate = None

    return {
        'noise_answer': noise_answer,
        'normalized_noise_answer': normalize_answer(noise_answer),
        'total_questions': len(noise_probs),
        'avg_noise_prob': avg_noise_prob,
        'mean_noise_rank': mean_noise_rank,
        'std_noise_rank': std_noise_rank,
        'questions_with_noise': questions_with_noise,
        'noise_appearance_rate': noise_appearance_rate
    }


def analyze_per_question(results: List[Dict], noise_answer: str) -> List[Dict]:
    """
    Analyze noise answer for each question individually.

    Args:
        results: List of result dicts
        noise_answer: The noise answer to analyze

    Returns:
        List of per-question analysis dictionaries
    """
    per_question_analysis = []

    for result in results:
        unique_id = result.get('unique_id')
        question_id = result.get('question_id')
        distribution = result.get('distribution', [])

        if not distribution:
            continue

        noise_prob, noise_rank = analyze_noise_in_distribution(distribution, noise_answer)

        per_question_analysis.append({
            'unique_id': unique_id,
            'question_id': question_id,
            'noise_prob': noise_prob,
            'noise_rank': noise_rank,
            'noise_in_distribution': noise_prob > 0 if noise_prob is not None else False
        })

    return per_question_analysis


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze impact of noise answers on distribution'
    )
    parser.add_argument(
        '--result',
        type=str,
        required=True,
        help='Path to distributions.json file'
    )
    parser.add_argument(
        '--noise_answer',
        type=str,
        required=True,
        help='The noise answer to analyze (e.g., "123")'
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Path to output analysis JSON (default: same directory with suffix _noise_analysis.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-question analysis'
    )

    args = parser.parse_args()

    # Set default output path
    if args.out is None:
        result_path = Path(args.result)
        args.out = str(result_path.parent / f'{result_path.stem}_noise_analysis.json')

    print(f"\nNoise Impact Analysis")
    print(f"=" * 60)
    print(f"Input:        {args.result}")
    print(f"Output:       {args.out}")
    print(f"Noise Answer: {args.noise_answer}")
    print(f"=" * 60)

    # Load distributions.json
    print(f"\nLoading distributions from {args.result}...")
    with open(args.result, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    metadata = data.get('metadata', {})
    summary = data.get('summary', {})

    print(f"Total questions: {len(results)}")

    # Calculate noise statistics
    print(f"\n=== Analyzing noise answer: {args.noise_answer} ===")
    noise_stats = calculate_noise_statistics(results, args.noise_answer)

    # Print results
    print(f"\nNoise Statistics:")
    print(f"-" * 60)
    print(f"Noise Answer (normalized): {noise_stats['normalized_noise_answer']}")
    print(f"Total Questions:           {noise_stats['total_questions']}")
    print(f"Questions with Noise:      {noise_stats['questions_with_noise']}")
    print(f"Noise Appearance Rate:     {noise_stats['noise_appearance_rate']:.2%}" if noise_stats['noise_appearance_rate'] is not None else "N/A")
    print(f"-" * 60)
    print(f"Average Noise Probability: {noise_stats['avg_noise_prob']:.4f}" if noise_stats['avg_noise_prob'] is not None else "N/A")
    print(f"Mean Noise Rank:           {noise_stats['mean_noise_rank']:.2f}" if noise_stats['mean_noise_rank'] is not None else "N/A")
    print(f"Std Dev Noise Rank:        {noise_stats['std_noise_rank']:.2f}" if noise_stats['std_noise_rank'] is not None else "N/A")
    print(f"=" * 60)

    # Per-question analysis
    per_question_analysis = analyze_per_question(results, args.noise_answer)

    if args.verbose:
        print(f"\nPer-Question Analysis:")
        print(f"-" * 60)
        for analysis in per_question_analysis[:10]:  # Show first 10
            print(f"Question {analysis['question_id']} ({analysis['unique_id']}): "
                  f"prob={analysis['noise_prob']:.4f}, rank={analysis['noise_rank']}, "
                  f"in_dist={analysis['noise_in_distribution']}")
        if len(per_question_analysis) > 10:
            print(f"... and {len(per_question_analysis) - 10} more questions")

    # Save output
    output_data = {
        'analysis_metadata': {
            'source_file': args.result,
            'noise_answer': args.noise_answer,
            'normalized_noise_answer': noise_stats['normalized_noise_answer'],
            'source_metadata': metadata,
            'source_summary': summary
        },
        'noise_statistics': noise_stats,
        'per_question_analysis': per_question_analysis
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis saved to: {output_path}")


if __name__ == '__main__':
    main()
