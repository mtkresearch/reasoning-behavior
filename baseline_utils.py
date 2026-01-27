#!/usr/bin/env python3
"""
Shared utilities for baseline generation across different task types.

This module provides common functions used by code, math, and science baseline generators:
- JSONL cache management for incremental saving and resume
- Result formatting and validation
- Core generation workflow

The module is designed to eliminate code duplication while supporting task-specific
prompt building and result formatting via callbacks.
"""

import json
from pathlib import Path
from typing import List, Dict, Callable, Optional
from tqdm import tqdm
import random

from llm_client import LLMClient, Request, Task, Response
from logger_config import setup_logger

logger = setup_logger(__name__, log_file='logs/baseline_utils.log')


def load_existing_jsonl_results(jsonl_path: Path) -> Dict[str, Dict]:
    """
    Load already processed results from JSONL file.

    JSONL format: one JSON object per line. Each result must have a 'unique_id' field
    that uniquely identifies it.

    Args:
        jsonl_path: Path to JSONL cache file

    Returns:
        Dict mapping unique_id to result dict
    """
    existing_results = {}

    if jsonl_path.exists():
        logger.info(f"Loading existing results from {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    unique_id = result.get('unique_id')
                    if unique_id:
                        existing_results[unique_id] = result
        logger.info(f"Loaded {len(existing_results)} existing results")

    return existing_results


def append_result_to_jsonl(result: Dict, jsonl_path: Path):
    """
    Append a single result to JSONL file.

    This is called immediately after processing each problem to ensure incremental
    saving and resume capability.

    Args:
        result: Result dictionary (must have 'unique_id' field)
        jsonl_path: Path to JSONL cache file
    """
    # Ensure parent directory exists
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def save_results(results: List[Dict], output_path: str):
    """
    Save results to JSON file.

    Args:
        results: List of result dicts
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(results)} results to {output_path}")
    print(f"\nSaved {len(results)} results to {output_path}")


def generate_baseline_core(
    problems: List[Dict],
    output_path: str,
    build_prompt_fn: Callable[[Dict], str],
    format_result_fn: Callable[[Dict, Response, str], Dict],
    system_prompt: str,
    id_field: str = 'unique_id',
    model_type: str = 'gpt-oss',
    mode: str = 'openrouter',
    limit: Optional[int] = None,
    max_workers: int = 16,
    use_complete_api: bool = False,
    get_unique_id_fn: Optional[Callable[[Dict], str]] = None
) -> List[Dict]:
    """
    Core baseline generation workflow.

    This function encapsulates the common logic used by all baseline generators:
    1. Setup JSONL cache
    2. Load existing results (from both JSON and JSONL)
    3. Filter unprocessed problems
    4. Prepare LLM tasks
    5. Generate with concurrent execution
    6. Format and save results incrementally

    Args:
        problems: List of problem dictionaries
        output_path: Path to save results JSON
        build_prompt_fn: Callable(problem: Dict) -> str. Builds prompt from problem
        format_result_fn: Callable(problem: Dict, response: Response, system_prompt: str) -> Dict.
                         Formats result from problem and response
        system_prompt: System prompt for LLM
        id_field: Field in problem dict to use as unique identifier (default: 'unique_id')
        model_type: Model type for LLM (default: 'gpt-oss')
        mode: LLM client mode (default: 'openrouter')
        limit: Limit number of problems to process (None = all)
        max_workers: Maximum concurrent workers (default: 16)
        use_complete_api: Use complete API endpoint for specific models (default: False)
        get_unique_id_fn: Optional callable to construct unique_id from problem.
                         If None, uses problem.get(id_field) directly.
                         Required for tasks where unique_id format differs from id_field.

    Returns:
        List of result dicts (existing + newly generated)
    """
    # Setup JSONL cache path
    output_path_obj = Path(output_path)
    jsonl_path = output_path_obj.parent / f"{output_path_obj.stem}.jsonl"

    # Ensure output directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results from final JSON output file first
    # Note: All results have 'unique_id' field, regardless of id_field setting
    existing_results = {}
    if output_path_obj.exists():
        logger.info(f"Loading existing results from {output_path_obj}")
        with open(output_path_obj, 'r', encoding='utf-8') as f:
            results_list = json.load(f)
            for result in results_list:
                unique_id = result.get('unique_id')  # Always use 'unique_id' from results
                if unique_id:
                    existing_results[unique_id] = result
        logger.info(f"Loaded {len(existing_results)} existing results from JSON")
        print(f"Loaded {len(existing_results)} existing results from {output_path_obj}")

    # Also load from JSONL cache (may have additional results not yet in final JSON)
    jsonl_results = load_existing_jsonl_results(jsonl_path)
    for unique_id, result in jsonl_results.items():
        if unique_id not in existing_results:
            existing_results[unique_id] = result

    # Extract unique IDs from existing results
    existing_ids = set(existing_results.keys())

    # Filter out already processed problems
    # Use get_unique_id_fn if provided, otherwise use id_field directly
    if get_unique_id_fn:
        problems_to_process = [
            p for p in problems
            if get_unique_id_fn(p) not in existing_ids
        ]
    else:
        problems_to_process = [
            p for p in problems
            if p.get(id_field) not in existing_ids
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
        return list(existing_results.values())

    # Initialize LLM client
    client = LLMClient(mode=mode, timeout=3600)

    # Prepare tasks
    tasks = []
    for i, problem in enumerate(problems_to_process):
        prompt = build_prompt_fn(problem)

        task = Task(
            index=i,
            request=Request(
                queries=[prompt],
                system_prompt=system_prompt,
                model_type=model_type,
                temperature=0.01,
                reasoning_on=True
            ),
            metadata={
                'id_field': id_field,
                'problem_id': problem.get(id_field),
                'problem': problem,
                'system_prompt': system_prompt
            }
        )
        tasks.append(task)

    # Shuffle tasks for better distribution
    random.shuffle(tasks)

    # Generate answers
    new_results = []
    for completed_task in tqdm(
        client.generate_concurrent(tasks, max_workers=max_workers, use_complete_api=use_complete_api),
        total=len(tasks)
    ):
        problem = completed_task.metadata['problem']
        system_prompt = completed_task.metadata['system_prompt']
        id_field = completed_task.metadata['id_field']
        response = completed_task.response

        # Check if generation was successful
        success = response.success and bool(response.content)

        # Extract reasoning and answer from response
        reasoning_traj = response.reasoning_content or ''
        final_answer = response.content or ''

        # Validate response
        if not reasoning_traj:
            error_msg = f"BUG: reasoning_content (traj) is empty for problem {problem.get(id_field)}. " \
                       "This indicates the LLM client or API is not returning reasoning content properly."
            logger.error(error_msg)
            continue

        if not final_answer:
            error_msg = f"BUG: final_answer is empty for problem {problem.get(id_field)}. " \
                       "This indicates the LLM client or API is not returning answer content properly."
            logger.error(error_msg)
            continue

        # Format result using task-specific function
        result = format_result_fn(problem, response, system_prompt)

        # Append to JSONL immediately for incremental saving
        append_result_to_jsonl(result, jsonl_path)
        new_results.append(result)

        # Log success/failure
        if success:
            logger.info(f"Generated result for {problem.get(id_field)}")
        else:
            logger.error(f"Failed for {problem.get(id_field)}: {response.err_message}")

    # Combine existing and new results
    all_results = list(existing_results.values()) + new_results

    return all_results
