#!/usr/bin/env python3
"""
生成 AIME2025 Baseline Results

這個腳本從 datasets/AIME2025/data.json 加載數學題目，使用 LLM 生成推理和答案，
並保存為 data/AIME2025/gpt-oss/p1/results.json

特性：
- 使用 JSONL 中繼檔案 (.jsonl) 記錄處理進度
- 支援斷點續傳：重新執行時會自動跳過已處理的題目
- 每處理完一個題目就立即寫入 JSONL，避免因中斷而遺失結果

檔案結構：
- results.jsonl: 中繼檔案，每行記錄一個題目的結果
- results.json: 最終輸出檔案，包含所有題目的結果

Usage:
    # 基本執行
    python generate_math_baseline.py --limit 5

    # 使用 R10 格式（重複 10 次）生成完整數據集
    python generate_math_baseline.py \
        --repeat_num 10 \
        --output_path data/AIME2025__R10/olmo/p1/results.json \
        --model_type olmo

    # 配置特定模型和並發數
    python generate_math_baseline.py \
        --model_type gpt-oss \
        --output_path data/AIME2025/gpt-oss/p1/results.json \
        --max_workers 8

    # 重新執行時會自動跳過已完成的題目
    python generate_math_baseline.py --repeat_num 10
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

from baseline_utils import generate_baseline_core, save_results
from logger_config import setup_logger

# Setup logger
logger = setup_logger(__name__, log_file='logs/generate_math_baseline.log')

# Default paths
DEFAULT_PROBLEMS_PATH = 'datasets/AIME2025/data.json'
DEFAULT_OUTPUT_PATH = 'data/AIME2025/gpt-oss/p1/results.json'


def load_problems(json_path: str = DEFAULT_PROBLEMS_PATH, repeat_num: int = 1) -> List[Dict]:
    """
    加載題目數據並重複指定次數

    Args:
        json_path: 題目 JSON 文件路徑
        repeat_num: 重複次數（默認 1）

    Returns:
        題目列表（重複後）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    if repeat_num <= 1:
        return problems

    # Expand problems by repeat_num times, adding repetition index to unique_id
    # Order: for each problem, repeat repeat_num times before moving to next problem
    expanded = []
    for problem in problems:
        for rep in range(repeat_num):
            new_problem = {
                'unique_id': f"{problem['unique_id']}-{rep}",
                'question': problem['question'],
                'answer': problem['answer']
            }
            expanded.append(new_problem)

    return expanded


def build_prompt(problem: Dict) -> str:
    """
    構建數學題目的 prompt

    Args:
        problem: 題目字典（包含 question 和 answer）

    Returns:
        完整的 prompt 字符串
    """
    question = problem['question']
    return f"{question}"




def generate_baseline(
    problems: List[Dict],
    output_path: str,
    model_type: str = 'gpt-oss',
    mode: str = 'openrouter',
    limit: int = None,
    max_workers: int = 16,
    repeat_num: int = 1
) -> List[Dict]:
    """
    生成 baseline results，使用 JSONL 中繼檔案支援斷點續傳

    Uses shared baseline_utils.generate_baseline_core for common workflow.

    Args:
        problems: 題目列表
        output_path: 輸出路徑（用於生成 JSONL 檔名）
        model_type: 模型類型
        mode: LLM 客戶端模式
        limit: 限制生成數量（None 為全部）
        max_workers: 最大並發數
        repeat_num: 重複次數（已在 load_problems 中處理，此處作為記錄）

    Returns:
        結果列表
    """
    instruction = "You are a helpful assistant"

    logger.info(f"Generating baseline with repeat_num={repeat_num}")

    use_complete_api = model_type == 'deepseek'

    return generate_baseline_core(
        problems=problems,
        output_path=output_path,
        build_prompt_fn=build_prompt,
        format_result_fn=lambda problem, response, sys_prompt: {
            'unique_id': problem['unique_id'],
            'question': problem['question'],
            'answer': problem['answer'],
            'result': {
                'traj': response.reasoning_content or '',
                'answer': response.content or '',
                'sys_prompt': sys_prompt,
                'elapsed_seconds': response.elapsed_seconds
            }
        },
        system_prompt=instruction,
        id_field='unique_id',
        model_type=model_type,
        mode=mode,
        limit=limit,
        max_workers=max_workers,
        use_complete_api=use_complete_api
    )




def main():
    parser = argparse.ArgumentParser(
        description='Generate AIME2025 baseline results'
    )
    parser.add_argument(
        '--problems_path',
        type=str,
        default=DEFAULT_PROBLEMS_PATH,
        help=f'Path to problems JSON file (default: {DEFAULT_PROBLEMS_PATH})'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f'Path to save results (default: {DEFAULT_OUTPUT_PATH})'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='gpt-oss',
        help='Model type (default: gpt-oss)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='openrouter',
        choices=['openrouter', 'local'],
        help='LLM client mode (default: openrouter)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of problems to process (for testing)'
    )

    DEFAULT_MAX_WORKERS = 4
    parser.add_argument(
        '--max_workers',
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f'Maximum concurrent workers (default: {DEFAULT_MAX_WORKERS})'
    )
    parser.add_argument(
        '--repeat_num',
        type=int,
        default=1,
        help='Number of times to repeat the dataset (default: 1). E.g., use 10 for R10'
    )

    args = parser.parse_args()

    # Load problems
    print(f"Loading problems from {args.problems_path}...")
    problems = load_problems(args.problems_path, repeat_num=args.repeat_num)
    print(f"Loaded {len(problems)} problems")

    # Generate baseline
    results = generate_baseline(
        problems,
        output_path=args.output_path,
        model_type=args.model_type,
        mode=args.mode,
        limit=args.limit,
        max_workers=args.max_workers,
        repeat_num=args.repeat_num
    )

    # Save results
    save_results(results, args.output_path)

    # Print summary
    successful = sum(1 for r in results if r['result']['answer'])
    failed = len(results) - successful
    print(f"\nSummary:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


if __name__ == '__main__':
    main()
