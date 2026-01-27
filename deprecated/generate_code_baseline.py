#!/usr/bin/env python3
"""
生成 CodeElo Baseline Results

這個腳本從 datasets/CodeElo/data/test.json 加載題目，使用 LLM 生成 C++ 代碼，
並保存為 data/CodeElo/gpt-oss/p1/results.json

特性：
- 使用 JSONL 中繼檔案 (.jsonl) 記錄處理進度
- 支援斷點續傳：重新執行時會自動跳過已處理的題目
- 每處理完一個題目就立即寫入 JSONL，避免因中斷而遺失結果

檔案結構：
- results.jsonl: 中繼檔案，每行記錄一個題目的結果
- results.json: 最終輸出檔案，包含所有題目的結果

Usage:
    # 首次執行或斷點續傳
    python generate_code_baseline.py --limit 5
    python generate_code_baseline.py --model_type gpt-oss --output_path data/CodeElo/gpt-oss/p1/results.json

    # 重新執行時會自動跳過已完成的題目
    python generate_code_baseline.py --output_path data/CodeElo/gpt-oss/p1/results.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

from baseline_utils import generate_baseline_core, save_results
from logger_config import setup_logger

# Setup logger
logger = setup_logger(__name__, log_file='logs/generate_code_baseline.log')

# Default paths
DEFAULT_PROBLEMS_PATH = 'datasets/CodeElo/data/test.json'
DEFAULT_OUTPUT_PATH = 'data/CodeElo/gpt-oss/p1/results.json'


def load_problems(json_path: str = DEFAULT_PROBLEMS_PATH) -> List[Dict]:
    """
    加載題目數據

    Args:
        json_path: 題目 JSON 文件路徑

    Returns:
        題目列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_html_problem(problem: Dict) -> str:
    """
    構建 HTML 格式的完整題目（類似 datasets/CodeElo/main.py:9-20）

    Args:
        problem: 題目字典

    Returns:
        HTML 格式的完整題目
    """
    html_output = '<html><body>'

    # Title
    html_output += f"<h1>{problem['title']}</h1>"

    # Time and memory limits
    html_output += f"<div><b>Time limit:</b> {problem['time_limit_ms']} ms</div>"
    html_output += f"<div><b>Memory limit:</b> {problem['memory_limit_mb']} MB</div>"
    html_output += f"<div><b>Rating:</b> {problem['rating']}</div>"
    html_output += "<br>"

    # Description
    html_output += "<h2>Description</h2>"
    html_output += f"<div>{problem['description']}</div>"

    # Input format
    html_output += "<h2>Input</h2>"
    html_output += f"<div>{problem['input']}</div>"

    # Output format
    html_output += "<h2>Output</h2>"
    html_output += f"<div>{problem['output']}</div>"

    # Examples
    if 'examples' in problem and problem['examples']:
        html_output += "<h2>Examples</h2>"
        for i, (input_text, output_text) in enumerate(problem['examples'], 1):
            html_output += f"<h3>Example {i}</h3>"
            html_output += f"<div><b>Input:</b><pre>{input_text}</pre></div>"
            html_output += f"<div><b>Output:</b><pre>{output_text}</pre></div>"

    # Note
    if problem.get('note'):
        html_output += "<h2>Note</h2>"
        html_output += f"<div>{problem['note']}</div>"

    html_output += '</body></html>'
    return html_output


def build_prompt(problem: Dict) -> str:
    """
    構建 C++ 代碼生成 prompt（遵循 datasets/CodeElo/main.py:59 的 instruction）

    Args:
        problem: 題目字典

    Returns:
        完整的 prompt 字符串
    """
    instruction = """You are a coding expert. Given a competition-level coding problem, you need to write a C++ program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```."""

    html_problem = make_html_problem(problem)

    return f"{instruction}\n\n{html_problem}"






def generate_baseline(
    problems: List[Dict],
    output_path: str,
    model_type: str = 'gpt-oss',
    mode: str = 'openrouter',
    limit: int = None,
    max_workers: int = 16
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

    Returns:
        結果列表
    """
    instruction = """You are a coding expert. Given a competition-level coding problem, you need to write a C++ program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```."""

    use_complete_api = model_type == 'deepseek'

    return generate_baseline_core(
        problems=problems,
        output_path=output_path,
        build_prompt_fn=build_prompt,
        format_result_fn=lambda problem, response, sys_prompt: {
            'unique_id': f"codeelo-{problem['problem_id']}-0",
            'question': make_html_problem(problem),
            'test_cases': problem['examples'],
            'result': {
                'traj': response.reasoning_content or '',
                'answer': response.content or '',
                'sys_prompt': sys_prompt,
                'elapsed_seconds': response.elapsed_seconds
            }
        },
        system_prompt=instruction,
        id_field='problem_id',
        model_type=model_type,
        mode=mode,
        limit=limit,
        max_workers=max_workers,
        use_complete_api=use_complete_api
    )




def main():
    parser = argparse.ArgumentParser(
        description='Generate CodeElo baseline results'
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

    args = parser.parse_args()

    # Load problems
    print(f"Loading problems from {args.problems_path}...")
    problems = load_problems(args.problems_path)
    print(f"Loaded {len(problems)} problems")

    # Generate baseline
    results = generate_baseline(
        problems,
        output_path=args.output_path,
        model_type=args.model_type,
        mode=args.mode,
        limit=args.limit,
        max_workers=args.max_workers
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
