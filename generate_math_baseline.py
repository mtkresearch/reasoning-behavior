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
from tqdm import tqdm
import random

from llm_client import LLMClient, Request, Task
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


def load_existing_jsonl_results(jsonl_path: Path) -> Dict[str, Dict]:
    """
    從 JSONL 檔案載入已處理的結果

    Args:
        jsonl_path: JSONL 檔案路徑

    Returns:
        Dict mapping problem_id to result
    """
    existing_results = {}

    if jsonl_path.exists():
        logger.info(f"Loading existing results from {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    # Extract problem_id from unique_id
                    unique_id = result.get('unique_id', '')
                    existing_results[unique_id] = result
        logger.info(f"Loaded {len(existing_results)} existing results")

    return existing_results


def append_result_to_jsonl(result: Dict, jsonl_path: Path):
    """
    將單一結果追加到 JSONL 檔案

    Args:
        result: 結果字典
        jsonl_path: JSONL 檔案路徑
    """
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


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
    # Store the instruction for later use
    instruction = "You are a helpful assistant"

    logger.info(f"Generating baseline with repeat_num={repeat_num}")

    # Setup JSONL cache path
    output_path_obj = Path(output_path)
    jsonl_path = output_path_obj.parent / f"{output_path_obj.stem}.jsonl"

    # Ensure output directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results from JSONL
    existing_results = load_existing_jsonl_results(jsonl_path)

    # Filter out already processed problems
    problems_to_process = [
        p for p in problems
        if p['unique_id'] not in existing_results
    ]

    # Limit problems if specified (apply after filtering)
    if limit:
        problems_to_process = problems_to_process[:limit]
        logger.info(f"Limited to {limit} problems")

    print(f"Total problems: {len(problems)}")
    print(f"Already processed: {len(existing_results)}")
    print(f"To process: {len(problems_to_process)}")

    if not problems_to_process:
        logger.info("All problems already processed, skipping generation")
        print("\nAll problems already processed!")
        # Return existing results
        return list(existing_results.values())

    # Initialize LLM client
    client = LLMClient(mode=mode, timeout=3600)

    # Prepare tasks
    tasks = []
    for i, problem in enumerate(problems_to_process):
        prompt = build_prompt(problem)

        task = Task(
            index=i,
            request=Request(
                queries=[prompt],
                model_type=model_type,
                temperature=0.01
            ),
            metadata={
                'unique_id': problem['unique_id'],
                'problem': problem,
                'instruction': instruction
            }
        )
        tasks.append(task)
    random.shuffle(tasks)

    # Generate answers
    new_results = []
    use_complete_api = model_type == 'deepseek'
    for completed_task in tqdm(client.generate_concurrent(tasks, max_workers=max_workers, use_complete_api=use_complete_api), total=len(tasks)):
        problem = completed_task.metadata['problem']
        instruction = completed_task.metadata['instruction']
        response = completed_task.response

        # Check if generation was successful
        success = response.success and bool(response.content)

        # Extract reasoning and answer from OpenRouter response
        reasoning_traj = response.reasoning_content or ''
        final_answer = response.content or ''

        # IMPORTANT: traj must not be empty - if it is, this indicates a bug
        if not reasoning_traj:
            error_msg = f"BUG: reasoning_content (traj) is empty for problem {problem['unique_id']}. This indicates the LLM client or OpenRouter API is not returning reasoning content properly."
            logger.error(error_msg)
            continue
        if not final_answer:
            error_msg = f"BUG: final_answer is empty for problem {problem['unique_id']}. This indicates the LLM client or OpenRouter API is not returning reasoning content properly."
            logger.error(error_msg)
            continue

        # Build result structure
        result = {
            'unique_id': problem['unique_id'],
            'question': problem['question'],
            'answer': problem['answer'],
            'result': {
                'traj': reasoning_traj,  # 模型的 reasoning 過程
                'answer': final_answer,  # 模型的最終答案
                'sys_prompt': instruction,  # Complete instruction text
                'elapsed_seconds': response.elapsed_seconds
            }
        }

        # Append to JSONL immediately
        append_result_to_jsonl(result, jsonl_path)
        new_results.append(result)

        # Log success/failure
        if success:
            logger.info(f"Generated solution for {problem['unique_id']}")
        else:
            logger.error(f"Failed to generate solution for {problem['unique_id']}: {response.err_message}")

    # Combine existing and new results
    all_results = list(existing_results.values()) + new_results

    return all_results


def save_results(results: List[Dict], output_path: str):
    """
    保存結果到 JSON 文件

    Args:
        results: 結果列表
        output_path: 輸出路徑
    """
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(results)} results to {output_path}")
    print(f"\nSaved {len(results)} results to {output_path}")


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
