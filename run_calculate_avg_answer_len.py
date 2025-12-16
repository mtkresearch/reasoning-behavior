"""
Calculate average answer length statistics from results.json

This script analyzes generated_answer fields in results.json and computes
token-level statistics (mean and standard deviation) using GPT-OSS tokenizer.

Usage:
    python run_calculate_avg_answer_len.py --results_path <path_to_results.json> [--only_correct]

Examples:
    # Calculate for all answers
    python run_calculate_avg_answer_len.py --results_path exp/e178799c/results.json

    # Calculate only for correct answers
    python run_calculate_avg_answer_len.py --results_path exp/e178799c/results.json --only_correct
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Dict

from transformers import AutoTokenizer

from core import load_existing_results


# Global tokenizer cache
_TOKENIZER_CACHE = {}


def load_tokenizer(model_name: str = "openai/gpt-oss-120b"):
    """Load and cache tokenizer"""
    if model_name not in _TOKENIZER_CACHE:
        print(f"正在載入 tokenizer: {model_name}...")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return _TOKENIZER_CACHE[model_name]


def calculate_answer_length_stats(results_path: str, tokenizer_model: str = "openai/gpt-oss-120b", only_correct: bool = False):
    """
    Calculate token length statistics for generated answers

    Args:
        results_path: Path to results.json file
        tokenizer_model: Model name for tokenizer (default: GPT-OSS)
        only_correct: If True, only calculate statistics for correct answers (default: False)

    Returns:
        Dictionary containing statistics and details
    """
    # Load results
    print(f"正在載入結果檔案: {results_path}")
    data = load_existing_results(results_path)

    # Extract results list
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError("無法從結果檔案中找到 'results' 欄位")

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_model)

    # Calculate lengths
    lengths = []
    total_processed = 0
    skipped_count = 0

    filter_msg = " (僅正確答案)" if only_correct else ""
    print(f"\n正在計算 {len(results)} 個答案的 token 長度{filter_msg}...")

    for i, result in enumerate(results):
        # Filter by correctness if requested
        if only_correct:
            if 'is_correct' not in result:
                print(f"警告: 結果 {i} 缺少 'is_correct' 欄位，跳過")
                skipped_count += 1
                continue
            if not result['is_correct']:
                skipped_count += 1
                continue

        if 'generated_answer' not in result:
            print(f"警告: 結果 {i} 缺少 'generated_answer' 欄位")
            skipped_count += 1
            continue

        answer = result['generated_answer']
        if answer is None:
            print(f"警告: 結果 {i} 的 'generated_answer' 為 None")
            skipped_count += 1
            continue

        # Tokenize and count
        tokens = tokenizer.encode(answer)
        length = len(tokens)
        lengths.append(length)
        total_processed += 1

    # Calculate statistics
    if not lengths:
        raise ValueError("沒有找到任何有效的 generated_answer")

    mean_length = statistics.mean(lengths)
    stdev_length = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    min_length = min(lengths)
    max_length = max(lengths)
    median_length = statistics.median(lengths)

    if only_correct:
        print(f"已處理: {total_processed} 個正確答案，跳過: {skipped_count} 個（錯誤或無效）")
    else:
        print(f"已處理: {total_processed} 個答案，跳過: {skipped_count} 個（無效）")

    stats = {
        'tokenizer_model': tokenizer_model,
        'only_correct': only_correct,
        'total_results': len(results),
        'total_answers': len(lengths),
        'skipped': skipped_count,
        'mean': mean_length,
        'stdev': stdev_length,
        'min': min_length,
        'max': max_length,
        'median': median_length,
        'lengths': lengths,
    }

    return stats


def print_statistics(stats: Dict):
    """Print statistics in a readable format"""
    print("\n" + "="*60)
    print("答案長度統計 (Token 數)")
    print("="*60)
    print(f"Tokenizer 模型: {stats['tokenizer_model']}")
    print(f"過濾條件:      {'僅正確答案' if stats.get('only_correct', False) else '所有答案'}")
    print(f"總結果數:      {stats.get('total_results', 'N/A')}")
    print(f"已計算數:      {stats['total_answers']}")
    if stats.get('skipped', 0) > 0:
        print(f"跳過數:        {stats['skipped']}")
    print(f"平均值:        {stats['mean']:.2f} tokens")
    print(f"標準差:        {stats['stdev']:.2f} tokens")
    print(f"中位數:        {stats['median']:.2f} tokens")
    print(f"最小值:        {stats['min']} tokens")
    print(f"最大值:        {stats['max']} tokens")
    print("="*60)


def save_statistics(stats: Dict, output_path: str):
    """Save statistics to JSON file"""
    # Create a copy without the full lengths array for summary
    summary = {k: v for k, v in stats.items() if k != 'lengths'}
    summary['sample_size'] = len(stats['lengths'])

    output_data = {
        'summary': summary,
        'lengths': stats['lengths']
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n統計結果已儲存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='計算 results.json 中 generated_answer 的 token 長度統計',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 計算所有答案的統計
  python run_calculate_avg_answer_len.py --results_path exp/e178799c/results.json

  # 僅計算正確答案的統計
  python run_calculate_avg_answer_len.py --results_path exp/e178799c/results.json --only_correct

  # 指定輸出檔案位置
  python run_calculate_avg_answer_len.py --results_path exp/e178799c/results.json --output exp/e178799c/answer_length_stats.json
        """
    )

    parser.add_argument(
        '--results_path',
        type=str,
        required=True,
        help='results.json 檔案路徑'
    )

    parser.add_argument(
        '--tokenizer_model',
        type=str,
        default='openai/gpt-oss-120b',
        help='Tokenizer 模型名稱 (預設: openai/gpt-oss-120b)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='輸出 JSON 檔案路徑 (選填，不指定則只顯示統計)'
    )

    parser.add_argument(
        '--only_correct',
        action='store_true',
        help='僅計算答案正確的結果 (根據 is_correct 欄位過濾)'
    )

    args = parser.parse_args()

    # Calculate statistics
    stats = calculate_answer_length_stats(args.results_path, args.tokenizer_model, args.only_correct)

    # Print to console
    print_statistics(stats)

    # Save to file if output path specified
    if args.output:
        save_statistics(stats, args.output)
    else:
        # Auto-generate output path based on input
        input_path = Path(args.results_path)
        output_path = input_path.parent / 'answer_length_stats.json'
        save_statistics(stats, str(output_path))


if __name__ == '__main__':
    main()
