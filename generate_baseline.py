#!/usr/bin/env python3
"""
Unified Baseline Generation Script

Supports three task types:
- code: CodeElo C++ programming problems
- math: AIME2025 mathematical problems
- science: GPQA-Diamond multiple-choice science questions

This unified script eliminates code duplication across different baseline generators
while maintaining task-specific prompt construction and result formatting.

Usage:
    # CodeElo baseline (code type)
    python generate_baseline.py --task_type code \\
        --problems_path datasets/CodeElo/data/test.json \\
        --output_path data/CodeElo/gpt-oss/p1/results.json

    # AIME2025 baseline with R10 (10 repetitions)
    python generate_baseline.py --task_type math \\
        --problems_path datasets/AIME2025/data.json \\
        --output_path data/AIME2025__R10/gpt-oss/p1/results.json \\
        --repeat_num 10 \\
        --model_type gpt-oss

    # GPQA-Diamond baseline (NEW)
    python generate_baseline.py --task_type science \\
        --problems_path datasets/GPQA-Diamond/test/gpqa_diamond.parquet \\
        --output_path data/GPQA-Diamond/gpt-oss/p1/results.json \\
        --model_type gpt-oss

    # Test with limit
    python generate_baseline.py --task_type math --limit 2

All scripts support:
- Incremental saving with JSONL cache for resume capability
- Concurrent generation with configurable max_workers
- Automatic skipping of already-processed problems
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import sys

from baseline_utils import generate_baseline_core, save_results
from logger_config import setup_logger

logger = setup_logger(__name__, log_file='logs/generate_baseline.log')


# ============================================================================
# CodeElo Task Support
# ============================================================================

def load_problems_code(path: str) -> List[Dict]:
    """Load CodeElo problems from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_html_problem(problem: Dict) -> str:
    """Convert problem dict to HTML format."""
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


def build_prompt_code(problem: Dict) -> str:
    """Build C++ code generation prompt."""
    html_problem = make_html_problem(problem)
    return html_problem


def format_result_code(problem: Dict, response, system_prompt: str) -> Dict:
    """Format CodeElo result."""
    return {
        'unique_id': f"codeelo-{problem['problem_id']}-0",
        'question': make_html_problem(problem),
        'test_cases': problem['examples'],
        'result': {
            'traj': response.reasoning_content or '',
            'answer': response.content or '',
            'sys_prompt': system_prompt,
            'elapsed_seconds': response.elapsed_seconds
        }
    }


# ============================================================================
# AIME2025 Task Support
# ============================================================================

def load_problems_math(path: str, repeat_num: int = 1) -> List[Dict]:
    """Load AIME2025 problems from JSON, optionally repeated."""
    with open(path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    if repeat_num <= 1:
        return problems

    # Expand problems by repeat_num times
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


def build_prompt_math(problem: Dict) -> str:
    """Build math problem prompt."""
    return problem['question']


def format_result_math(problem: Dict, response, system_prompt: str) -> Dict:
    """Format AIME2025 result."""
    return {
        'unique_id': problem['unique_id'],
        'question': problem['question'],
        'answer': problem['answer'],
        'result': {
            'traj': response.reasoning_content or '',
            'answer': response.content or '',
            'sys_prompt': system_prompt,
            'elapsed_seconds': response.elapsed_seconds
        }
    }


# ============================================================================
# GPQA-Diamond Task Support
# ============================================================================

def load_problems_science(path: str) -> List[Dict]:
    """Load GPQA-Diamond problems from Parquet file."""
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required for GPQA-Diamond support")
        print("Install with: pip install pandas pyarrow")
        sys.exit(1)

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Error loading Parquet file {path}: {e}")
        sys.exit(1)

    problems = []
    for idx, row in df.iterrows():
        problem = {
            'unique_id': f"gpqa-diamond-{idx}",
            'question': row['question'],
            'answer': row['answer']  # A/B/C/D
        }
        problems.append(problem)

    return problems


def build_prompt_science(problem: Dict) -> str:
    """Build science question prompt."""
    return problem['question']


def format_result_science(problem: Dict, response, system_prompt: str) -> Dict:
    """Format GPQA-Diamond result."""
    return {
        'unique_id': problem['unique_id'],
        'question': problem['question'],
        'answer': problem['answer'],
        'result': {
            'traj': response.reasoning_content or '',
            'answer': response.content or '',
            'sys_prompt': system_prompt,
            'elapsed_seconds': response.elapsed_seconds
        }
    }


# ============================================================================
# Task Configuration
# ============================================================================

def get_unique_id_code(problem: Dict) -> str:
    """Construct unique_id for CodeElo problems."""
    return f"codeelo-{problem['problem_id']}-0"


TASK_CONFIGS = {
    'code': {
        'load_fn': load_problems_code,
        'prompt_fn': build_prompt_code,
        'result_fn': format_result_code,
        'system_prompt': "You are a coding expert. Given a competition-level coding problem, you need to write a C++ program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ```cpp ```.",
        'id_field': 'problem_id',
        'get_unique_id_fn': get_unique_id_code,
        'default_path': 'datasets/CodeElo/data/test.json',
        'default_output': 'data/CodeElo/gpt-oss/p1/results.json',
        'supports_repeat': False
    },
    'math': {
        'load_fn': load_problems_math,
        'prompt_fn': build_prompt_math,
        'result_fn': format_result_math,
        'system_prompt': "You are a helpful assistant",
        'id_field': 'unique_id',
        'default_path': 'datasets/AIME2025/data.json',
        'default_output': 'data/AIME2025/gpt-oss/p1/results.json',
        'supports_repeat': True
    },
    'science': {
        'load_fn': load_problems_science,
        'prompt_fn': build_prompt_science,
        'result_fn': format_result_science,
        'system_prompt': "You are a science expert. Given a multiple-choice question, select the most appropriate answer (A, B, C, or D). ",
        'id_field': 'unique_id',
        'default_path': 'datasets/GPQA-Diamond/test/gpqa_diamond.parquet',
        'default_output': 'data/GPQA-Diamond/gpt-oss/p1/results.json',
        'supports_repeat': False
    }
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified baseline generation for code, math, and science tasks'
    )

    parser.add_argument(
        '--task_type',
        type=str,
        required=True,
        choices=['code', 'math', 'science'],
        help='Task type (code, math, or science)'
    )

    parser.add_argument(
        '--problems_path',
        type=str,
        default=None,
        help='Path to problems file (JSON for code/math, Parquet for science)'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save results JSON'
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

    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='Maximum concurrent workers (default: 4)'
    )

    parser.add_argument(
        '--repeat_num',
        type=int,
        default=1,
        help='Number of times to repeat dataset (for math task with R10 format, default: 1)'
    )

    args = parser.parse_args()

    # Get task configuration
    if args.task_type not in TASK_CONFIGS:
        print(f"Error: Unknown task type {args.task_type}")
        sys.exit(1)

    config = TASK_CONFIGS[args.task_type]

    # Use defaults if not specified
    problems_path = args.problems_path or config['default_path']

    # Generate output_path based on model_type if not explicitly provided
    if args.output_path:
        output_path = args.output_path
    else:
        # Replace 'gpt-oss' in default_output with actual model_type
        default_output = config['default_output']
        output_path = default_output.replace('/gpt-oss/', f'/{args.model_type}/')

    # Check if problems file exists
    if not Path(problems_path).exists():
        print(f"Error: Problems file not found: {problems_path}")
        sys.exit(1)

    # Load problems
    print(f"Loading problems from {problems_path}...")
    try:
        if config.get('supports_repeat') and args.repeat_num > 1:
            problems = config['load_fn'](problems_path, repeat_num=args.repeat_num)
        else:
            problems = config['load_fn'](problems_path)
    except Exception as e:
        print(f"Error loading problems: {e}")
        sys.exit(1)

    print(f"Loaded {len(problems)} problems")

    # Determine if we should use complete API
    use_complete_api = args.model_type == 'deepseek'

    # Generate baseline
    try:
        results = generate_baseline_core(
            problems=problems,
            output_path=output_path,
            build_prompt_fn=config['prompt_fn'],
            format_result_fn=config['result_fn'],
            system_prompt=config['system_prompt'],
            id_field=config['id_field'],
            model_type=args.model_type,
            mode=args.mode,
            limit=args.limit,
            max_workers=args.max_workers,
            use_complete_api=use_complete_api,
            get_unique_id_fn=config.get('get_unique_id_fn')
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        logger.exception("Generation failed")
        sys.exit(1)

    # Save results
    save_results(results, output_path)

    # Print summary
    successful = sum(1 for r in results if r.get('result', {}).get('answer'))
    failed = len(results) - successful
    print(f"\nSummary:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


if __name__ == '__main__':
    main()
