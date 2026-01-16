#!/usr/bin/env python3
"""
生成 CodeElo Baseline Results

這個腳本從 CodeElo/data/test.json 加載題目，使用 LLM 生成 C++ 代碼，
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
from tqdm import tqdm

from llm_client import LLMClient, Request, Task
from logger_config import setup_logger
from core import extract_code_blocks  # Import from core.py (has fallback implementation)

# Setup logger
logger = setup_logger(__name__, log_file='logs/generate_code_baseline.log')

# Default paths
DEFAULT_PROBLEMS_PATH = 'CodeElo/data/test.json'
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
    構建 HTML 格式的完整題目（類似 CodeElo/main.py:9-20）

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
    構建 C++ 代碼生成 prompt（遵循 CodeElo/main.py:59 的 instruction）

    Args:
        problem: 題目字典

    Returns:
        完整的 prompt 字符串
    """
    instruction = """You are a coding expert. Given a competition-level coding problem, you need to write a C++ program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```."""

    html_problem = make_html_problem(problem)

    return f"{instruction}\n\n{html_problem}"




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
                    # Extract problem_id from unique_id (format: "codeforces-{problem_id}-0")
                    unique_id = result.get('unique_id', '')
                    if unique_id.startswith('codeforces-'):
                        problem_id = unique_id.split('-')[1]
                        existing_results[problem_id] = result
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
    max_workers: int = 16
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

    Returns:
        結果列表
    """
    # Store the instruction for later use
    instruction = """You are a coding expert. Given a competition-level coding problem, you need to write a C++ program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```."""

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
        if p['problem_id'] not in existing_results
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
    client = LLMClient(mode=mode, timeout=60)

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
                'problem_id': problem['problem_id'],
                'problem': problem,
                'instruction': instruction
            }
        )
        tasks.append(task)

    # Generate answers
    new_results = []
    for completed_task in tqdm(client.generate_concurrent(tasks, max_workers=max_workers, use_complete_api=True), total=len(tasks)):
        problem = completed_task.metadata['problem']
        instruction = completed_task.metadata['instruction']
        response = completed_task.response

        # Check if generation was successful
        success = response.success and bool(response.content)

        # Extract reasoning and answer from OpenRouter response
        # OpenRouter returns (in single call):
        # - reasoning_content: the model's reasoning process (traj)
        # - content: the model's final answer (answer)
        reasoning_traj = response.reasoning_content or ''
        final_answer = response.content or ''



        # IMPORTANT: traj must not be empty - if it is, this indicates a bug
        if not reasoning_traj:
            error_msg = f"BUG: reasoning_content (traj) is empty for problem {problem['problem_id']}. This indicates the LLM client or OpenRouter API is not returning reasoning content properly."
            logger.error(error_msg)
            continue
        if not final_answer:
            error_msg = f"BUG: final_answer is empty for problem {problem['problem_id']}. This indicates the LLM client or OpenRouter API is not returning reasoning content properly."
            logger.error(error_msg)
            continue

        # Build result structure
        # Note: 'test_cases' field contains test cases in format [[input, output], ...]
        # Each element is a pair where:
        #   - First element: input string for the test case
        #   - Second element: expected output string for the test case
        result = {
            'unique_id': f"codeforces-{problem['problem_id']}-0",
            'question': make_html_problem(problem),  # Complete HTML-formatted problem
            'test_cases': problem['examples'],  # Test cases: [[input, output], ...]
            'result': {
                'traj': reasoning_traj,  # 模型的 reasoning 過程
                'answer': final_answer,  # 模型的最終答案（包含 code）
                'sys_prompt': instruction,  # Complete instruction text
                'elapsed_seconds': response.elapsed_seconds
            }
        }

        # Append to JSONL immediately
        append_result_to_jsonl(result, jsonl_path)
        new_results.append(result)

        # Log success/failure
        if success:
            logger.info(f"Generated code for {problem['problem_id']}")
        else:
            logger.error(f"Failed to generate code for {problem['problem_id']}: {response.err_message}")

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
