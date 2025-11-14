#!/usr/bin/env python3
"""
Reasoning Processing Experiment with Pipeline Architecture

This script processes reasoning text through a configurable pipeline of transformations
(masking, truncating, shuffling) and evaluates model performance on the processed reasoning.

=============================================================================
USAGE
=============================================================================

Basic syntax:
    python mask_experiment.py --flow "<pipeline_steps>"

The --flow parameter accepts a comma-separated sequence of processing steps.

-----------------------------------------------------------------------------
Available Processors
-----------------------------------------------------------------------------

1. mask(mode, mask_char='█', ...)
   Mask numbers, answers, or alphabetic characters in reasoning.

   Modes:
   - 'number': Mask all numbers (0-9)
   - 'answer': Mask only the answer number
   - 'line': Mask all numbers in lines containing the answer
   - 'n-lines': Mask numbers in answer line and N previous lines (specify num_prev_lines=N)
   - 'number-advance': Mask computational numbers, preserve algebraic notation (e.g., x_1, 3x)
   - 'alphabet': Mask all alphabetic characters (A-Z, a-z)
   - 'alphabet-and-answer': Mask alphabet AND answer number

   Optional parameters:
   - mask_char: Character to use for masking (default: '█')
   - num_prev_lines: For 'n-lines' mode (default: 1)

2. truncate(mode, ...)
   Remove lines from reasoning.

   Modes:
   - 'answer_and_after': Remove answer line and all lines after it
   - 'before_answer': Remove all lines before answer line (answer line kept)
   - 'last_n_lines': Remove last N lines (specify n=N)
   - 'last_ratio': Remove last X% of lines (specify ratio=X, e.g., 0.3 for 30%)

3. shuffle(mode, seed=None, ...)
   Shuffle reasoning content.

   Modes:
   - 'line': Shuffle lines
   - 'word': Shuffle words
   - 'token': Shuffle tokens (specify tokenizer_model='gpt2' or other)

   Optional parameters:
   - seed: Random seed for reproducibility

4. insert(mode, sentence='...', count=1, seed=None)
   Insert text into reasoning chain at random positions.

   Modes:
   - 'fix': Insert fixed text at random positions

   Parameters:
   - sentence: Text to insert (default: 'Maybe the answer is 123.')
   - count: Number of times to insert the text (default: 1)
   - seed: Random seed for reproducibility

-----------------------------------------------------------------------------
Examples
-----------------------------------------------------------------------------

# Example 1: Basic masking
python mask_experiment.py --flow "mask('number')"

# Example 2: Mask and shuffle
python mask_experiment.py --flow "mask('number'),shuffle('line')"

# Example 3: Full pipeline - truncate, mask, shuffle
python mask_experiment.py --flow "truncate('answer_and_after'),mask('number'),shuffle('line')"

# Example 4: Custom mask character
python mask_experiment.py --flow "mask('number',mask_char='*')"

# Example 5: Ratio-based truncation
python mask_experiment.py --flow "truncate('last_ratio',ratio=0.3),mask('alphabet')"

# Example 6: Word-level shuffle with specific seed
python mask_experiment.py --flow "mask('number'),shuffle('word',seed=42)"

# Example 7: N-lines masking
python mask_experiment.py --flow "mask('n-lines',num_prev_lines=2)"

# Example 8: Complex combination
python mask_experiment.py --flow "truncate('last_ratio',ratio=0.3),mask('number-advance'),shuffle('line',seed=123)"

# Example 9: Insert noise at random positions
python mask_experiment.py --flow "insert('fix',sentence='Maybe the answer is 123.',count=5)"

# Example 10: Combine insert with other processors
python mask_experiment.py --flow "insert('fix',sentence='Thus answer: 123.',count=3),shuffle('line')"

# Example 11: Insert with specific seed for reproducibility
python mask_experiment.py --flow "insert('fix',sentence='Answer: 456.',count=5,seed=42)"

-----------------------------------------------------------------------------
Other Parameters
-----------------------------------------------------------------------------

--results_path: Path to input results.json (default: data/AIME2025__R10/gpt-oss/p1/results.json)
--output_path: Path to save output (default: data/baseline/mask_numbers_experiment.json)
--model_type: Model to use (gpt-oss, deepseek, qwen3)
--mode: LLM client mode (openrouter, local)
--limit: Limit number of questions for testing

=============================================================================
"""

import json
import re
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import logging
import tempfile
import shutil

from llm_client import LLMClient, Task, Request, CompletionRequest
from logger_config import setup_logger
from core import (
    parse_answer_from_completion,
    parse_yes_no_response,
    build_gpt_oss_prompt_with_reasoning,
    GRADING_PROMPT,
    mask_numbers_in_reasoning,
    mask_answer_only_in_reasoning,
    mask_numbers_in_lines_with_answer,
    mask_numbers_in_nlines_with_answer,
    mask_alphabet_in_reasoning,
    mask_alphabet_and_answer_in_reasoning,
    mask_numbers_advance,
    remove_answer_and_after,
    shuffle_lines
)
from pipeline import parse_flow, Pipeline

# Default concurrency (can be overridden by --max_workers)
DEFAULT_MAX_WORKERS = 10
# Default max retry (can be overridden by --max_retry)
DEFAULT_MAX_RETRY = 3

# Setup logger
logger = setup_logger(__name__, log_file='logs/run_experiment.log')


# =============================================================================
# Safe File Operations - JSONL Strategy
# =============================================================================

def append_to_jsonl(filepath: Path, data: dict):
    """
    Append single result to JSONL file (append-only, never corrupts)

    Args:
        filepath: Path to JSONL file
        data: Single result dictionary
    """
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def load_from_jsonl(filepath: Path) -> List[dict]:
    """
    Load all results from JSONL file

    Args:
        filepath: Path to JSONL file

    Returns:
        List of result dictionaries
    """
    results = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSONL line: {e}")
                        continue
    return results


def atomic_save_json(filepath: Path, data: dict):
    """
    Atomic write to JSON file (write to temp, then rename)
    Ensures original file is never corrupted during write

    Args:
        filepath: Path to JSON file
        data: Data to save
    """
    filepath = Path(filepath)

    # Create temp file in same directory (required for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=f'.{filepath.name}.tmp',
        suffix='.json'
    )

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic rename (replaces old file safely)
        shutil.move(temp_path, filepath)
    except Exception as e:
        # Clean up temp file on error
        if Path(temp_path).exists():
            os.unlink(temp_path)
        raise e


def rebuild_json_from_jsonl(
    stage2_jsonl: Path,
    output_json: Path,
    experiment_metadata: dict
):
    """
    Rebuild results.json from stage2.jsonl with metadata and summary

    Args:
        stage2_jsonl: Path to stage 2 JSONL file (with grading)
        output_json: Path to final JSON output
        experiment_metadata: Experiment metadata dictionary
    """
    # Load all results from stage 2
    results = load_from_jsonl(stage2_jsonl)

    # Sort by question_id for consistency
    results.sort(key=lambda x: x.get('question_id', 0))

    # Calculate summary statistics
    total = len(results)
    generation_successful = sum(1 for r in results if r.get('generation_success', False))
    generation_failed = total - generation_successful
    grading_successful = sum(1 for r in results if r.get('success', False))
    grading_failed = generation_successful - grading_successful
    correct = sum(1 for r in results if r.get('is_correct', False))
    accuracy = correct / grading_successful if grading_successful > 0 else 0.0

    summary = {
        'total_questions': total,
        'generation_successful': generation_successful,
        'generation_failed': generation_failed,
        'grading_successful': grading_successful,
        'grading_failed': grading_failed,
        'correct': correct,
        'accuracy': accuracy
    }

    # Build final structure
    output_data = {
        'experiment_metadata': experiment_metadata,
        'summary': summary,
        'results': results
    }

    # Atomic write to JSON
    atomic_save_json(output_json, output_data)
    logger.info(f"Rebuilt {output_json} from {stage2_jsonl}")


# =============================================================================
# Path Generation
# =============================================================================

def generate_output_path_from_flow(results_path: str, flow: str) -> str:
    """
    Generate output path automatically based on flow string

    Creates path: exp/<processor1>/<processor2>/.../<processor_n>/results.json

    Args:
        results_path: Original results.json path
        flow: Flow string

    Returns:
        Generated output path

    Examples:
        flow = "mask('number'),shuffle('line')"
        -> exp/mask_number/shuffle_line/results.json

        flow = "insert('fix',sentence='Answer: 123.',count=5)"
        -> exp/insert_fix_sentence_Answer_123_count_5/results.json
    """
    import re
    from pathlib import Path

    if not flow:
        return "exp/no_processing/results.json"

    # Parse flow to extract processor steps
    processors = parse_flow(flow)

    # Build path segments by sanitizing the flow string directly
    segments = []

    # Split flow by comma to get individual processor calls
    flow_steps = re.split(r',(?![^()]*\))', flow)

    for step in flow_steps:
        step = step.strip()
        # Sanitize: replace non-alphanumeric chars with underscore, collapse multiple underscores
        sanitized = re.sub(r'[^\w]+', '_', step)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        segments.append(sanitized)

    # Construct path: exp/<seg1>/<seg2>/.../<segN>/results.json
    output_path = Path("exp") / Path(*segments) / "results.json"

    return str(output_path)


# =============================================================================
# Masking Strategy Registry
# =============================================================================

MASK_STRATEGIES = {
    'number': mask_numbers_in_reasoning,
    'answer': mask_answer_only_in_reasoning,
    'line': mask_numbers_in_lines_with_answer,
    'n-lines': mask_numbers_in_nlines_with_answer,
    'number-advance': mask_numbers_advance,
    'alphabet': mask_alphabet_in_reasoning,
    'alphabet-and-answer': mask_alphabet_and_answer_in_reasoning,
}


def apply_mask_strategy(
    reasoning: str,
    answer: str,
    mask_mode: str,
    mask_char: str = '█',
    num_prev_lines: int = 1
) -> str:
    """
    Apply masking strategy to reasoning text

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer
        mask_mode: Masking mode (number, answer, line, n-lines, number-advance, alphabet, alphabet-and-answer)
        mask_char: Character to use for masking (default: '█')
        num_prev_lines: Number of previous non-empty lines to mask (used with 'n-lines' mode)

    Returns:
        Masked reasoning content

    Raises:
        ValueError: If mask_mode is not recognized
    """
    strategy = MASK_STRATEGIES.get(mask_mode)
    if not strategy:
        raise ValueError(f"Unknown mask mode: {mask_mode}")

    # Apply strategy with appropriate parameters based on mode
    if mask_mode == 'n-lines':
        # n-lines mode needs num_prev_lines parameter
        return strategy(reasoning, answer, num_prev_lines, mask_char)
    elif mask_mode in ['number', 'alphabet']:
        # These modes don't need answer parameter
        return strategy(reasoning, mask_char)
    elif mask_mode == 'number-advance':
        # number-advance mode uses answer as keyword argument
        return strategy(reasoning, answer=answer, mask_char=mask_char)
    else:
        # Other modes (answer, line, alphabet-and-answer) use standard parameters
        return strategy(reasoning, answer, mask_char)


def load_results_json(results_path: str) -> List[Dict]:
    """
    Load results.json file

    Returns:
        List of result items with unique_id, question, answer, result
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_task(
    item: Dict,
    model_type: str,
    flow: str = None,
) -> Task:
    """
    Prepare a Task for reasoning processing (masking, truncating, shuffling, etc.)

    Note: This function supports both the new --flow syntax.

    Args:
        item: Result item from results.json
        model_type: Model type
        flow: Flow string (e.g., "mask('number'),shuffle('line')").
        seed_base: Base seed for shuffling (used when flow doesn't specify seed)
    """
    unique_id = item['unique_id']
    question = item['question']
    ground_truth = item['answer']
    original_reasoning = item['result']['traj']

    # Extract index from unique_id (e.g., "aime2025-I-0-2" -> 2)
    try:
        index = int(unique_id.split('-')[-1])
    except (ValueError, IndexError, AttributeError) as e:
        logger.debug(f"Failed to extract index from {unique_id}: {e}")
        index = hash(unique_id) % 10000

    # Determine flow string
    flow_str = flow

    # Process reasoning using Pipeline
    if flow_str:
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)
        context = {
            'question': question,
            'answer': ground_truth,
            'ground_truth': ground_truth
        }

        processed_reasoning, processing_metadata = pipeline.execute(original_reasoning, context)
        # Use question from context (may be modified by processors like truncate_question)
        final_question = context['question']
    else:
        # No processing
        processed_reasoning = original_reasoning
        processing_metadata = []
        final_question = question

    # Build prompt with processed reasoning
    prompt = build_gpt_oss_prompt_with_reasoning(final_question, processed_reasoning)

    # Create CompletionRequest
    request = CompletionRequest(
        prompt=prompt,
        model_type=model_type,
        temperature=0.5,
        max_tokens=5000
        # Note: min_tokens not supported by OpenRouter API
    )

    # Create Task with metadata
    task = Task(
        index=index,
        request=request,
        metadata={
            'unique_id': unique_id,
            'question': question,
            'ground_truth': ground_truth,
            'original_reasoning': original_reasoning,
            'processed_reasoning': processed_reasoning,
            'flow': flow_str,
            'processing_metadata': processing_metadata,
        }
    )

    return task


def task_to_result(task: Task) -> Dict:
    """
    Convert completed Task to result dict
    """
    metadata = task.metadata
    response = task.response

    # Check if response is successful and has non-empty content
    success = response.success and bool(response.content)

    # Determine error message
    if not response.success:
        error = response.err_message
    elif not response.content:
        error = 'Empty generated_answer from API'
    else:
        error = None

    # Build result dict
    result = {
        'unique_id': metadata['unique_id'],
        'question_id': task.index,
        'question': metadata['question'],
        'ground_truth': metadata['ground_truth'],
        'original_reasoning': metadata['original_reasoning'],
        'processed_reasoning': metadata.get('processed_reasoning', metadata.get('masked_reasoning', '')),
        'flow': metadata.get('flow', ''),
        'processing_metadata': metadata.get('processing_metadata', []),
        'generated_answer': response.content if success else None,
        'is_correct': None,
        'grading_reasoning': None,
        'generation_success': success,
        'success': False,
        'error': error,
        'retry_count': 0  # Track retry attempts
    }

    return result


def _save_results_with_metadata(
    output_path: str,
    results: List[Dict],
    results_path: str,
    model_type: str,
    flow_str: str
):
    """
    Save results in the new format with experiment_metadata, summary, and results

    Args:
        output_path: Path to save output
        results: List of result dictionaries
        results_path: Original results path (for metadata)
        model_type: Model type used
        flow_str: Flow string used
    """
    from datetime import datetime
    from pathlib import Path
    from pipeline import parse_flow

    # Parse flow to get flow_config
    try:
        processors = parse_flow(flow_str)
        flow_config = []
        for i, processor in enumerate(processors, 1):
            metadata = processor.get_metadata()
            flow_config.append({
                'step': i,
                'processor': metadata.get('processor', 'unknown'),
                'params': {k: v for k, v in metadata.items()
                          if k not in ['processor', 'input_stats', 'output_stats']}
            })
    except Exception as e:
        # If flow parsing fails, create minimal config
        logger.warning(f"Flow parsing failed for '{flow_str}': {e}")
        flow_config = [{'step': 1, 'processor': 'unknown', 'params': {'flow': flow_str}}]

    # Extract dataset name from results_path
    # e.g., "data/AIME2025__R10/gpt-oss/p1/results.json" -> "AIME2025__R10"
    dataset_name = "unknown"
    try:
        path_parts = Path(results_path).parts
        if 'data' in path_parts:
            data_index = path_parts.index('data')
            if len(path_parts) > data_index + 1:
                dataset_name = path_parts[data_index + 1]
    except (ValueError, IndexError, AttributeError) as e:
        logger.debug(f"Failed to extract dataset name from '{results_path}': {e}")

    # Calculate summary statistics
    # Support both new format (generation_success) and legacy format (success)
    generation_successful = [r for r in results if r.get('generation_success', r.get('success', False))]
    grading_successful = [r for r in results if r.get('success', False)]
    correct_count = sum(1 for r in grading_successful if r.get('is_correct', False))

    summary = {
        'total_questions': len(results),
        'generation_successful': len(generation_successful),
        'generation_failed': len(results) - len(generation_successful),
        'grading_successful': len(grading_successful),
        'grading_failed': len(generation_successful) - len(grading_successful),
        'correct': correct_count,
        'accuracy': correct_count / len(grading_successful) if grading_successful else 0
    }

    # Build experiment metadata
    experiment_metadata = {
        'experiment_name': Path(output_path).parent.name,
        'experiment_date': datetime.now().isoformat(),
        'dataset': dataset_name,
        'model_type': model_type,
        'flow': flow_str,
        'flow_config': flow_config
    }

    # Sort results by question_id
    results_sorted = sorted(results, key=lambda x: x['question_id'])

    # Build final output structure
    output = {
        'experiment_metadata': experiment_metadata,
        'summary': summary,
        'results': results_sorted
    }

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def create_grading_tasks(results: List[Dict], judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create grading tasks for generated answers"""
    tasks = []

    for result in results:
        if not result.get('generation_success', False):
            continue

        tasks.append(Task(
            index=result['question_id'],
            request=Request(
                queries=[GRADING_PROMPT.format(
                    problem=result['question'],
                    ground_truth=result['ground_truth'],
                    model_answer=result['generated_answer']
                )],
                model_type=judge_model_type,
                system_prompt="You are a helpful mathematical grading assistant.",
                reasoning_on=False,
                temperature=0.01
                # Note: min_tokens not supported by OpenRouter API
            ),
            metadata={'result_id': result['unique_id']}
        ))

    return tasks


def run_experiment(
    results_path: str,
    output_path: str,
    model_type: str = 'gpt-oss',
    mode: str = 'openrouter',
    flow: str = None,
    limit: int = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retry: int = DEFAULT_MAX_RETRY
):
    """
    Run the mask numbers experiment using JSONL strategy

    Args:
        results_path: Path to results.json
        output_path: Path to save output (e.g., exp/mask_number/results.json)
        model_type: Model type
        mode: 'openrouter' or 'local'
        flow: Flow string
        limit: Limit number of questions (for testing)
        max_workers: Maximum concurrent workers
        max_retry: Maximum retry attempts

    File Strategy:
        - Stage 1 (generation): Write to {output_path}_stage1.jsonl
        - Stage 2 (grading): Write to {output_path}_stage2.jsonl
        - Final: Rebuild {output_path}.json from stage2.jsonl
    """
    print(f"Loading results from {results_path}")
    data = load_results_json(results_path)

    if limit:
        data = data[:limit]
        print(f"Limited to {limit} questions")

    # Prepare file paths
    output_path = Path(output_path)
    stage1_jsonl = output_path.parent / f"{output_path.stem}_stage1.jsonl"
    stage2_jsonl = output_path.parent / f"{output_path.stem}_stage2.jsonl"
    output_json = output_path  # e.g., results.json

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Stage 1 JSONL: {stage1_jsonl}")
    print(f"Stage 2 JSONL: {stage2_jsonl}")
    print(f"Final JSON: {output_json}")

    # Load existing Stage 1 results from JSONL
    existing_stage1 = load_from_jsonl(stage1_jsonl)
    completed_stage1_ids = {
        r['unique_id'] for r in existing_stage1
        if r.get('generation_success', False)
    }

    successful_count = len(completed_stage1_ids)
    failed_count = len(existing_stage1) - successful_count
    logger.info(f"Found {len(existing_stage1)} Stage 1 results: {successful_count} successful, {failed_count} to retry")
    print(f"Found {len(existing_stage1)} Stage 1 results: {successful_count} successful, {failed_count} to retry")

    # Filter out only successfully completed items
    data = [item for item in data if item['unique_id'] not in completed_stage1_ids]
    print(f"Tasks to process: {len(data)}")

    # Determine flow string
    flow_str = flow
    print(f"Using flow: {flow_str}")

    print(f"Total questions: {len(data)}")

    # Initialize LLM client
    client = LLMClient(mode=mode)

    # Prepare tasks
    print("Preparing processing tasks...")
    tasks = []
    for item in data:
        task = prepare_task(item, model_type, flow=flow_str)
        tasks.append(task)

    # Phase 1: Generate answers with processed reasoning
    # Build in-memory map from existing Stage 1 JSONL
    stage1_results_map = {r['unique_id']: r for r in existing_stage1}

    if len(tasks) > 0:
        print(f"\n=== Phase 1: Generating answers with processed reasoning ({len(tasks)} tasks) ===")

        # Execute generation tasks
        for completed_task in tqdm(client.complete_concurrent(tasks, max_workers=max_workers), total=len(tasks)):
            new_result = task_to_result(completed_task)

            # Update in-memory map
            stage1_results_map[new_result['unique_id']] = new_result

            # Append to Stage 1 JSONL (append-only, never corrupts)
            append_to_jsonl(stage1_jsonl, new_result)

        # Unified retry logic for failed tasks
        for retry_attempt in range(max_retry):
            # Identify ALL failed tasks from in-memory map
            failed_results = [
                r for r in stage1_results_map.values()
                if (not r.get('generation_success')) or  # API failure
                   (r.get('generation_success') and not r.get('generated_answer'))  # Empty answer
            ]

            if not failed_results:
                break

            print(f"\n=== Retry attempt {retry_attempt + 1}/{max_retry}: {len(failed_results)} failed tasks ===")

            # Prepare retry tasks
            retry_tasks = []
            for result in failed_results:
                # Find original task
                original_task = next(
                    (t for t in tasks if t.metadata['unique_id'] == result['unique_id']),
                    None
                )
                if original_task:
                    retry_tasks.append(original_task)

            # Execute retry tasks with error handling
            try:
                for completed_task in tqdm(client.complete_concurrent(retry_tasks, max_workers=max_workers),
                                          total=len(retry_tasks)):
                    # Update with new response
                    unique_id = completed_task.metadata['unique_id']
                    response = completed_task.response

                    if response.success and response.content:
                        stage1_results_map[unique_id]['generated_answer'] = response.content
                        stage1_results_map[unique_id]['generation_success'] = True
                        stage1_results_map[unique_id]['error'] = None
                        stage1_results_map[unique_id]['retry_count'] = retry_attempt + 1
                    else:
                        stage1_results_map[unique_id]['retry_count'] = retry_attempt + 1
                        stage1_results_map[unique_id]['generation_success'] = response.success
                        stage1_results_map[unique_id]['error'] = response.err_message if response.err_message else stage1_results_map[unique_id].get('error')
                        print(f"Retry {retry_attempt + 1} failed for {unique_id}: {stage1_results_map[unique_id]['error']}")

                    # Append updated result to Stage 1 JSONL
                    append_to_jsonl(stage1_jsonl, stage1_results_map[unique_id])

            except Exception as e:
                logger.error(f"Retry batch failed with exception: {e}", exc_info=True)
                print(f"\nERROR: Retry batch failed with exception: {e}")
                print("Continuing with remaining retries...")
                continue  # Don't crash, try next retry round

        # Final warning for still-failed tasks
        still_failed = [
            r for r in stage1_results_map.values()
            if not r.get('generation_success') or (r.get('generation_success') and not r.get('generated_answer'))
        ]
        if still_failed:
            logger.warning(f"{len(still_failed)} tasks still failed after {max_retry} retries")
            print(f"\nWarning: {len(still_failed)} tasks still failed after {max_retry} retries")
            for result in still_failed:
                error_msg = result.get('error', 'Unknown error')
                logger.warning(f"Failed task: {result['unique_id']}: {error_msg}")
                print(f"  - {result['unique_id']}: {error_msg}")
    else:
        print(f"\n=== Phase 1: All tasks already completed, skipping generation ===")

    # Phase 2: Grade answers
    print(f"\n=== Phase 2: Grading answers ===")

    # Load existing Stage 2 results from JSONL
    existing_stage2 = load_from_jsonl(stage2_jsonl)
    stage2_results_map = {r['unique_id']: r for r in existing_stage2}

    # Get all successful Stage 1 results
    stage1_successful = [r for r in stage1_results_map.values() if r.get('generation_success', False)]

    # Identify which need grading
    need_grading = [
        r for r in stage1_successful
        if r['unique_id'] not in stage2_results_map or not stage2_results_map[r['unique_id']].get('success', False)
    ]

    print(f"Total successful generations: {len(stage1_successful)}")
    print(f"Already graded: {len(stage1_successful) - len(need_grading)}")
    print(f"Need to grade: {len(need_grading)}")

    grading_tasks = create_grading_tasks(need_grading, judge_model_type=model_type)

    if len(grading_tasks) > 0:
        print(f"Grading {len(grading_tasks)} answers...")
        for grading_task in tqdm(client.generate_concurrent(grading_tasks, max_workers=max_workers),
                                 total=len(grading_tasks)):
            if not grading_task.response.success:
                logger.error(f"Error in grading task {grading_task.index}: {grading_task.response.err_message}")
                print(f"\nError in grading task {grading_task.index}: {grading_task.response.err_message}")
                continue

            result_id = grading_task.metadata['result_id']

            # Get result from Stage 1 and add grading info
            if result_id in stage1_results_map:
                graded_result = stage1_results_map[result_id].copy()
                graded_result['is_correct'] = parse_yes_no_response(grading_task.response.content)
                graded_result['grading_reasoning'] = grading_task.response.content
                graded_result['success'] = True

                # Update Stage 2 map
                stage2_results_map[result_id] = graded_result

                # Append to Stage 2 JSONL
                append_to_jsonl(stage2_jsonl, graded_result)
    else:
        print(f"All results already graded, skipping grading phase")

    # Rebuild final JSON from Stage 2 JSONL
    print(f"\n=== Rebuilding final JSON from Stage 2 JSONL ===")

    # Prepare experiment metadata
    from datetime import datetime
    from pipeline import parse_flow

    # Parse flow to get flow_config
    try:
        processors = parse_flow(flow_str)
        flow_config = []
        for i, processor in enumerate(processors, 1):
            metadata = processor.get_metadata()
            flow_config.append({
                'step': i,
                'processor': metadata.get('processor', 'unknown'),
                'params': {k: v for k, v in metadata.items()
                          if k not in ['processor', 'input_stats', 'output_stats']}
            })
    except Exception as e:
        logger.warning(f"Flow parsing failed for '{flow_str}': {e}")
        flow_config = [{'step': 1, 'processor': 'unknown', 'params': {'flow': flow_str}}]

    # Extract dataset name from results_path
    dataset_name = "unknown"
    try:
        path_parts = Path(results_path).parts
        if 'data' in path_parts:
            data_index = path_parts.index('data')
            if len(path_parts) > data_index + 1:
                dataset_name = path_parts[data_index + 1]
    except (ValueError, IndexError, AttributeError) as e:
        logger.debug(f"Failed to extract dataset name from '{results_path}': {e}")

    experiment_metadata = {
        'experiment_name': output_json.parent.name,
        'experiment_date': datetime.now().isoformat(),
        'dataset': dataset_name,
        'model_type': model_type,
        'flow': flow_str,
        'flow_config': flow_config
    }

    # Rebuild JSON from Stage 2 JSONL
    rebuild_json_from_jsonl(stage2_jsonl, output_json, experiment_metadata)

    # Load final JSON to print statistics
    with open(output_json, 'r', encoding='utf-8') as f:
        final_output = json.load(f)

    stats = final_output['summary']

    # Print experiment summary
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Flow:                         {flow_str}")
    print(f"Model:                        {model_type}")
    print(f"Total Questions:              {stats['total_questions']}")
    print(f"Generation Successful:        {stats['generation_successful']}")
    print(f"Generation Failed:            {stats['generation_failed']}")
    print(f"Grading Successful:           {stats['grading_successful']}")
    print(f"Grading Failed:               {stats['grading_failed']}")
    print(f"Correct Answers:              {stats['correct']}")
    print(f"Accuracy:                     {stats['accuracy']:.2%}")
    print("="*60)

    print(f"\nStage 1 JSONL: {stage1_jsonl}")
    print(f"Stage 2 JSONL: {stage2_jsonl}")
    print(f"Final JSON:    {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description='Mask Numbers Experiment on result.traj'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default='data/AIME2025__R10/gpt-oss/p1/results.json',
        help='Path to results.json'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save results. If not specified, auto-generated based on --flow in exp/ directory'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='gpt-oss',
        choices=['gpt-oss', 'deepseek', 'qwen3'],
        help='Model type'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='openrouter',
        choices=['openrouter', 'local'],
        help='LLM client mode'
    )
    parser.add_argument(
        '--flow',
        type=str,
        default="",
        help='Processing flow string (e.g., "mask(\'number\'),shuffle(\'line\')"). '
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions to process (for testing)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f'Maximum number of concurrent workers (default: {DEFAULT_MAX_WORKERS})'
    )
    parser.add_argument(
        '--max_retry',
        type=int,
        default=DEFAULT_MAX_RETRY,
        help=f'Maximum number of retry attempts for failed tasks (default: {DEFAULT_MAX_RETRY})'
    )

    args = parser.parse_args()

    # Determine flow string
    flow_str = args.flow

    # Auto-generate output path if not specified
    if args.output_path is None:
        args.output_path = generate_output_path_from_flow(args.results_path, flow_str)
        print(f"Auto-generated output path: {args.output_path}")

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    run_experiment(
        results_path=args.results_path,
        output_path=args.output_path,
        model_type=args.model_type,
        mode=args.mode,
        flow=args.flow,
        limit=args.limit,
        max_workers=args.max_workers,
        max_retry=args.max_retry
    )


if __name__ == '__main__':
    main()
