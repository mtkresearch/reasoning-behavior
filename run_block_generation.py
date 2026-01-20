#!/usr/bin/env python3
"""
Block Generation Experiment with Parallel Processing

This script generates reasoning in 3 parallel blocks with padding:
- Block 1: Generate N//3 tokens without padding
- Block 2: Generate N//3 tokens with N//3 tokens padding
- Block 3: Generate N//3 tokens with 2*N//3 tokens padding

Then merges the blocks and uses retrieval to generate the final answer.

Usage:
    python run_block_generation.py --limit 3
    python run_block_generation.py --results_path data/AIME2025__R10/gpt-oss/p1/results.json
    python run_block_generation.py --output_path experiments/exp_block_generation/results.json
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
from datetime import datetime

from transformers import AutoTokenizer

from llm_client import LLMClient, Task, Request, CompletionRequest
from logger_config import setup_logger
from core import (
    parse_answer_from_completion,
    parse_yes_no_response,
    build_gpt_oss_prompt_with_reasoning_prefilled_answer,
    GRADING_PROMPT,
    append_to_jsonl,
    load_from_jsonl
)

# Default parameters
DEFAULT_RESULTS_PATH = "data/AIME2025__R10/gpt-oss/p1/results.json"
DEFAULT_OUTPUT_PATH = "experiments/exp_block_generation/results.json"
TOKENIZER_MODEL = "openai/gpt-oss-120b"
MODEL_TYPE = "gpt-oss"
DEFAULT_MAX_WORKERS = 4  # Higher for better parallelism
DEFAULT_MAX_RETRIES = 2  # Number of retries on failure
PREFILL_TEXT = "Thus, the answer is"

# Setup logger
logger = setup_logger(__name__, log_file='logs/run_block_generation.log')

# Global tokenizer cache
_TOKENIZER_CACHE = {}


# =============================================================================
# Retry Helper Functions
# =============================================================================

def retry_on_failure(func, max_retries: int = DEFAULT_MAX_RETRIES, *args, **kwargs):
    """
    Retry a function call on failure

    Args:
        func: Function to call
        max_retries: Maximum number of retries (default: DEFAULT_MAX_RETRIES)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Function result on success, or raises the last exception
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying...")
                time.sleep(1)  # Brief delay before retry
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")

    raise last_exception


# =============================================================================
# Tokenizer Functions
# =============================================================================

def get_tokenizer(model_name: str = TOKENIZER_MODEL) -> AutoTokenizer:
    """
    Get or create cached tokenizer

    Args:
        model_name: Tokenizer model name

    Returns:
        AutoTokenizer instance
    """
    if model_name not in _TOKENIZER_CACHE:
        logger.info(f"Loading tokenizer: {model_name}")
        print(f"Loading tokenizer: {model_name}...")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    return _TOKENIZER_CACHE[model_name]


def count_tokens(text: str, tokenizer: AutoTokenizer = None) -> int:
    """
    Count tokens in text

    Args:
        text: Text to count tokens
        tokenizer: Tokenizer instance (optional, will use default if not provided)

    Returns:
        Number of tokens
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def find_single_token_char(tokenizer: AutoTokenizer) -> str:
    """
    Find a character that encodes to exactly 1 token

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Single-token character
    """
    candidates = ["█", "▓", "▒", "░", "_", "-", ".", " ", "X", "0"]

    for char in candidates:
        tokens = tokenizer.encode(char, add_special_tokens=False)
        if len(tokens) == 1:
            logger.debug(f"Found single-token char: '{char}'")
            return char

    # Fallback: use space
    logger.warning("No single-token char found, using space")
    return " "


def generate_padding_text(num_tokens: int, tokenizer: AutoTokenizer = None) -> str:
    """
    Generate padding text with exactly num_tokens tokens

    Args:
        num_tokens: Target number of tokens
        tokenizer: Tokenizer instance (optional)

    Returns:
        Padding text
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # Strategy: Check if "█" is single token
    test_tokens = tokenizer.encode("█", add_special_tokens=False)

    if len(test_tokens) == 1:
        # "█" is single token, repeat it
        return "█" * num_tokens
    else:
        # Find alternative single-token character
        single_token_char = find_single_token_char(tokenizer)
        return single_token_char * num_tokens


# =============================================================================
# Prompt Building Functions
# =============================================================================

def build_gpt_oss_block_prompt(
    question: str,
    padding_tokens: int = 0,
    system_prompt: str = "You are a helpful assistant",
    reasoning_effort: str = "high"
) -> str:
    """
    Build GPT-OSS prompt for block generation with optional padding

    Args:
        question: The math question
        padding_tokens: Number of padding tokens to add (0 for block1)
        system_prompt: System message
        reasoning_effort: Reasoning effort level

    Returns:
        Formatted prompt string for text completion
    """
    # System message
    system_msg = f"You are a helpful assistant. Approach mathematical problems with {reasoning_effort} reasoning effort."

    # Build prompt using GPT-OSS chat template
    prompt = f"<|start|>system<|message|>{system_msg}<|end|>\n"
    prompt += f"<|start|>user<|message|>{question}<|end|>\n"
    prompt += "<|start|>assistant<|channel|>analysis<|message|>"

    # Add padding if needed
    if padding_tokens > 0:
        padding_text = generate_padding_text(padding_tokens)
        prompt += padding_text

    return prompt


# =============================================================================
# Block Task Preparation
# =============================================================================

def prepare_all_block_tasks(
    data: List[Dict],
    model_type: str = MODEL_TYPE
) -> Tuple[List[Task], Dict]:
    """
    Prepare all block tasks for all instances (3*N tasks total)

    Args:
        data: List of instances from results.json
        model_type: Model type

    Returns:
        Tuple of (all_block_tasks, instance_metadata)
        where instance_metadata maps instance_idx to original data
    """
    all_block_tasks = []
    instance_metadata = {}
    tokenizer = get_tokenizer()

    print("Preparing block tasks...")
    for idx, item in enumerate(tqdm(data)):
        unique_id = item['unique_id']
        question = item['question']
        ground_truth = item['answer']
        original_traj = item['result']['traj']

        # Count tokens in original trajectory
        total_tokens = count_tokens(original_traj, tokenizer)
        block_size = total_tokens // 3

        # Store metadata
        instance_metadata[idx] = {
            'unique_id': unique_id,
            'question': question,
            'ground_truth': ground_truth,
            'original_traj': original_traj,
            'total_tokens': total_tokens,
            'block_size': block_size
        }

        # Create 3 block tasks for this instance
        # Block 1: no padding
        block1_task = Task(
            index=f"{idx}_block1",
            request=CompletionRequest(
                prompt=build_gpt_oss_block_prompt(question, padding_tokens=0),
                model_type=model_type,
                max_tokens=block_size
            ),
            metadata={
                'instance_idx': idx,
                'block_id': 1,
                'unique_id': unique_id
            }
        )

        # Block 2: padding N//3 tokens
        block2_task = Task(
            index=f"{idx}_block2",
            request=CompletionRequest(
                prompt=build_gpt_oss_block_prompt(question, padding_tokens=block_size),
                model_type=model_type,
                max_tokens=block_size
            ),
            metadata={
                'instance_idx': idx,
                'block_id': 2,
                'unique_id': unique_id
            }
        )

        # Block 3: padding 2*N//3 tokens
        block3_task = Task(
            index=f"{idx}_block3",
            request=CompletionRequest(
                prompt=build_gpt_oss_block_prompt(question, padding_tokens=2*block_size),
                model_type=model_type,
                max_tokens=block_size
            ),
            metadata={
                'instance_idx': idx,
                'block_id': 3,
                'unique_id': unique_id
            }
        )

        all_block_tasks.extend([block1_task, block2_task, block3_task])

    logger.info(f"Prepared {len(all_block_tasks)} block tasks for {len(data)} instances")
    print(f"Prepared {len(all_block_tasks)} block tasks (3 blocks per instance)")

    return all_block_tasks, instance_metadata


# =============================================================================
# Block Merging and Answer Retrieval
# =============================================================================

def merge_blocks(block1: str, block2: str, block3: str) -> str:
    """
    Merge three blocks into a single reasoning text with newline separators

    Args:
        block1: First block content
        block2: Second block content
        block3: Third block content

    Returns:
        Merged reasoning text with '\n' separators between blocks
    """
    return block1 + '\n' + block2 + '\n' + block3


def generate_answer_with_retrieval(
    question: str,
    merged_reasoning: str,
    client: LLMClient,
    model_type: str = MODEL_TYPE,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> Tuple[str, bool, str]:
    """
    Generate answer using retrieval with prefilled answer

    Args:
        question: The math question
        merged_reasoning: Merged blocks reasoning
        client: LLM client
        model_type: Model type
        max_retries: Maximum number of retries on failure

    Returns:
        Tuple of (generated_answer, success, error_message)
    """
    def _generate():
        # Build retrieval prompt
        prompt = build_gpt_oss_prompt_with_reasoning_prefilled_answer(
            question=question,
            reasoning=merged_reasoning,
            prefill_text=PREFILL_TEXT,
            reasoning_effort="high"
        )

        # Generate answer
        request = CompletionRequest(
            prompt=prompt,
            model_type=model_type,
            max_tokens=100  # Answers are typically short
        )

        answer_text = client.complete(request)

        # Parse answer
        generated_answer = parse_answer_from_completion(answer_text)

        return generated_answer

    try:
        generated_answer = retry_on_failure(_generate, max_retries=max_retries)
        return generated_answer, True, None
    except Exception as e:
        logger.error(f"Error in answer retrieval after {max_retries + 1} attempts: {e}", exc_info=True)
        return "", False, str(e)


# =============================================================================
# Grading Functions
# =============================================================================

def create_grading_tasks(
    stage1_results: List[Dict],
    judge_model_type: str = MODEL_TYPE
) -> List[Task]:
    """
    Create grading tasks for generated answers

    Args:
        stage1_results: List of stage1 results
        judge_model_type: Model type for grading

    Returns:
        List of grading tasks
    """
    tasks = []

    for result in stage1_results:
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
            ),
            metadata={'result_id': result['unique_id']}
        ))

    return tasks


# =============================================================================
# JSON Rebuilding
# =============================================================================

def rebuild_json_from_jsonl(
    stage2_jsonl: Path,
    output_json: Path,
    experiment_metadata: Dict
):
    """
    Rebuild results.json from stage2.jsonl with metadata and summary

    Args:
        stage2_jsonl: Path to stage 2 JSONL file (with grading)
        output_json: Path to final JSON output
        experiment_metadata: Experiment metadata dictionary
    """
    # Load all results from stage 2
    all_results = load_from_jsonl(stage2_jsonl)

    # Deduplicate by unique_id, keeping the latest (last) entry for each
    results_map = {}
    for result in all_results:
        unique_id = result.get('unique_id')
        if unique_id:
            results_map[unique_id] = result

    # Convert back to list and sort by question_id for consistency
    results = list(results_map.values())
    results.sort(key=lambda x: x.get('question_id', 0))

    # Calculate summary statistics
    total = len(results)

    # Count instances with/without errors
    results_with_errors = sum(1 for r in results if r.get('has_errors', False))
    results_without_errors = total - results_with_errors

    # Generation stats (all instances)
    generation_successful = sum(1 for r in results if r.get('generation_success', False))
    generation_failed = total - generation_successful

    # Grading stats (only instances without errors)
    # An instance is "grading_successful" if it has no errors AND grading succeeded
    grading_successful = sum(
        1 for r in results
        if r.get('success', False) and not r.get('has_errors', False)
    )
    grading_failed = results_without_errors - grading_successful

    # Correct answers (only count instances without errors)
    correct = sum(
        1 for r in results
        if r.get('is_correct', False) and not r.get('has_errors', False)
    )

    # Accuracy is calculated only over instances without errors
    accuracy = correct / grading_successful if grading_successful > 0 else 0.0

    summary = {
        'total_questions': total,
        'results_with_errors': results_with_errors,
        'results_without_errors': results_without_errors,
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

    # Save to file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Rebuilt {output_json} from {stage2_jsonl}")


# =============================================================================
# Main Experiment Function
# =============================================================================

def run_block_generation_experiment(
    results_path: str,
    output_path: str,
    model_type: str = MODEL_TYPE,
    mode: str = 'openrouter',
    limit: int = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES
):
    """
    Run the block generation experiment

    Args:
        results_path: Path to results.json
        output_path: Path to save output
        model_type: Model type
        mode: 'openrouter' or 'local'
        limit: Limit number of questions (for testing)
        max_workers: Maximum concurrent workers
        max_retries: Maximum number of retries on failure
    """
    print(f"Loading results from {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"Limited to {limit} questions")

    # Prepare file paths
    output_path = Path(output_path)
    blocks_jsonl = output_path.parent / f"{output_path.stem}_blocks.jsonl"
    stage1_jsonl = output_path.parent / f"{output_path.stem}_stage1.jsonl"
    stage2_jsonl = output_path.parent / f"{output_path.stem}_stage2.jsonl"
    output_json = output_path

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Blocks JSONL:  {blocks_jsonl}")
    print(f"Stage 1 JSONL: {stage1_jsonl}")
    print(f"Stage 2 JSONL: {stage2_jsonl}")
    print(f"Final JSON:    {output_json}")

    # Initialize LLM client
    client = LLMClient(mode=mode, timeout=120)

    # =============================================================================
    # Phase 1: Generate all blocks in parallel (3*N tasks)
    # =============================================================================

    print(f"\n=== Phase 1: Generating all blocks in parallel ===")

    # Prepare all block tasks
    all_block_tasks, instance_metadata = prepare_all_block_tasks(data, model_type)

    # Dictionary to collect block results
    instance_blocks = {}  # {instance_idx: {1: block1, 2: block2, 3: block3}}
    instance_block_errors = {}  # {instance_idx: {block_id: error_message}}

    # Load existing blocks from blocks.jsonl if exists
    completed_block_keys = set()  # Track successfully completed (instance_idx, block_id) pairs
    if blocks_jsonl.exists():
        print(f"Loading existing blocks from {blocks_jsonl}...")
        existing_blocks = load_from_jsonl(blocks_jsonl)
        for block_result in existing_blocks:
            instance_idx = block_result['instance_idx']
            block_id = block_result['block_id']
            block_content = block_result.get('block_content', '')
            success = block_result.get('success', False)
            error = block_result.get('error')

            if instance_idx not in instance_blocks:
                instance_blocks[instance_idx] = {}
                instance_block_errors[instance_idx] = {}

            # Only mark as completed if successful
            if success:
                instance_blocks[instance_idx][block_id] = block_content
                completed_block_keys.add((instance_idx, block_id))
            elif error:
                # Failed blocks will be retried
                instance_block_errors[instance_idx][block_id] = error

        print(f"Loaded {len(existing_blocks)} existing block results ({len(completed_block_keys)} successful)")

    # Filter out successfully completed block tasks (failed ones will be retried)
    remaining_block_tasks = []
    for task in all_block_tasks:
        instance_idx = task.metadata['instance_idx']
        block_id = task.metadata['block_id']
        if (instance_idx, block_id) not in completed_block_keys:
            remaining_block_tasks.append(task)

    if len(remaining_block_tasks) == 0:
        print("All blocks already completed. Skipping block generation.")
    else:
        print(f"\nExecuting {len(remaining_block_tasks)} remaining block tasks (out of {len(all_block_tasks)} total)...")

        for completed_task in tqdm(
            client.complete_concurrent(remaining_block_tasks, max_workers=max_workers),
            total=len(remaining_block_tasks),
            desc="Generating blocks"
        ):
            instance_idx = completed_task.metadata['instance_idx']
            block_id = completed_task.metadata['block_id']
            unique_id = completed_task.metadata['unique_id']

            if instance_idx not in instance_blocks:
                instance_blocks[instance_idx] = {}
                instance_block_errors[instance_idx] = {}

            # Store block content (empty if failed)
            block_content = ""
            block_success = False
            error_message = None

            if completed_task.response.success:
                block_content = completed_task.response.content
                block_success = True
            else:
                error_message = completed_task.response.err_message
                logger.error(f"Block {block_id} for instance {instance_idx} failed: {error_message}")
                instance_block_errors[instance_idx][block_id] = error_message

            instance_blocks[instance_idx][block_id] = block_content

            # Save block result to blocks.jsonl
            block_result = {
                'unique_id': unique_id,
                'instance_idx': instance_idx,
                'block_id': block_id,
                'block_content': block_content,
                'success': block_success,
                'error': error_message
            }
            append_to_jsonl(blocks_jsonl, block_result)

    # =============================================================================
    # Phase 2: Merge blocks and generate answers with retrieval
    # =============================================================================

    print(f"\n=== Phase 2: Merging blocks and generating answers ===")

    stage1_results_map = {}
    completed_instance_ids = set()

    # Load existing stage1 results if exists
    if stage1_jsonl.exists():
        print(f"Loading existing stage1 results from {stage1_jsonl}...")
        existing_stage1 = load_from_jsonl(stage1_jsonl)
        successful_count = 0
        for result in existing_stage1:
            unique_id = result['unique_id']
            question_id = result['question_id']
            generation_success = result.get('generation_success', False)
            has_errors = result.get('has_errors', False)

            # Only mark as completed if successful and no errors
            if generation_success and not has_errors:
                stage1_results_map[unique_id] = result
                completed_instance_ids.add(question_id)
                successful_count += 1
            # Failed instances will be retried, so don't add to completed set

        print(f"Loaded {len(existing_stage1)} existing stage1 results ({successful_count} successful)")

    # Process only uncompleted instances (including previously failed ones)
    indices_to_process = [idx for idx in range(len(data)) if idx not in completed_instance_ids]

    if len(indices_to_process) == 0:
        print("All answers already generated successfully. Skipping answer generation.")
    else:
        print(f"Generating answers for {len(indices_to_process)} instances (out of {len(data)} total)...")

    for idx in tqdm(indices_to_process, desc="Generating answers"):
        metadata = instance_metadata[idx]
        blocks = instance_blocks.get(idx, {})
        block_errors = instance_block_errors.get(idx, {})

        # Get blocks (use empty string if failed)
        block1 = blocks.get(1, "")
        block2 = blocks.get(2, "")
        block3 = blocks.get(3, "")

        # Check if any block had errors
        has_block_errors = len(block_errors) > 0

        # Check if all blocks succeeded
        all_blocks_success = (block1 != "" and block2 != "" and block3 != "")

        if not all_blocks_success or has_block_errors:
            # Save failed result
            error_details = []
            if has_block_errors:
                for block_id, err_msg in block_errors.items():
                    error_details.append(f"Block {block_id}: {err_msg}")
            if not all_blocks_success:
                error_details.append("One or more blocks failed to generate")

            stage1_result = {
                'unique_id': metadata['unique_id'],
                'question_id': idx,
                'question': metadata['question'],
                'ground_truth': metadata['ground_truth'],
                'original_traj': metadata['original_traj'],
                'total_tokens': metadata['total_tokens'],
                'block_size': metadata['block_size'],
                'block1': block1,
                'block2': block2,
                'block3': block3,
                'merged_reasoning': "",
                'generated_answer': "",
                'generation_success': False,
                'has_errors': True,
                'error': "; ".join(error_details)
            }
        else:
            # Merge blocks
            merged_reasoning = merge_blocks(block1, block2, block3)

            # Generate answer with retrieval
            generated_answer, success, error = generate_answer_with_retrieval(
                metadata['question'],
                merged_reasoning,
                client,
                model_type,
                max_retries=max_retries
            )

            stage1_result = {
                'unique_id': metadata['unique_id'],
                'question_id': idx,
                'question': metadata['question'],
                'ground_truth': metadata['ground_truth'],
                'original_traj': metadata['original_traj'],
                'total_tokens': metadata['total_tokens'],
                'block_size': metadata['block_size'],
                'block1': block1,
                'block2': block2,
                'block3': block3,
                'merged_reasoning': merged_reasoning,
                'generated_answer': generated_answer,
                'generation_success': success,
                'has_errors': not success,  # Mark as having errors if answer generation failed
                'error': error
            }

        # Save to stage1 JSONL
        append_to_jsonl(stage1_jsonl, stage1_result)
        stage1_results_map[metadata['unique_id']] = stage1_result

    # =============================================================================
    # Phase 3: Grade answers
    # =============================================================================

    print(f"\n=== Phase 3: Grading answers ===")

    # Load existing stage2 results if exists
    stage2_results_map = {}
    completed_grading_ids = set()
    if stage2_jsonl.exists():
        print(f"Loading existing stage2 results from {stage2_jsonl}...")
        existing_stage2 = load_from_jsonl(stage2_jsonl)
        successful_count = 0
        for result in existing_stage2:
            unique_id = result['unique_id']
            success = result.get('success', False)
            has_errors = result.get('has_errors', False)

            # Only mark as completed if successful and no errors
            if success and not has_errors:
                stage2_results_map[unique_id] = result
                completed_grading_ids.add(unique_id)
                successful_count += 1
            # Failed grading will be retried, so don't add to completed set

        print(f"Loaded {len(existing_stage2)} existing stage2 results ({successful_count} successful)")

    # Get successful stage1 results (no errors and generation succeeded)
    stage1_successful = [
        r for r in stage1_results_map.values()
        if r.get('generation_success', False) and not r.get('has_errors', False)
    ]

    # Filter out already graded instances
    stage1_to_grade = [
        r for r in stage1_successful
        if r['unique_id'] not in completed_grading_ids
    ]

    print(f"Total successful generations: {len(stage1_successful)}")
    print(f"Already graded: {len(stage1_successful) - len(stage1_to_grade)}")
    print(f"Remaining to grade: {len(stage1_to_grade)}")

    # Create grading tasks only for ungraded instances
    grading_tasks = create_grading_tasks(stage1_to_grade, judge_model_type=model_type)

    if len(grading_tasks) > 0:
        print(f"Grading {len(grading_tasks)} answers...")

        for grading_task in tqdm(
            client.generate_concurrent(grading_tasks, max_workers=max_workers),
            total=len(grading_tasks),
            desc="Grading"
        ):
            result_id = grading_task.metadata['result_id']

            # Get result from Stage 1
            if result_id in stage1_results_map:
                graded_result = stage1_results_map[result_id].copy()

                if not grading_task.response.success:
                    # Grading failed - mark as error
                    logger.error(f"Grading failed for {result_id}: {grading_task.response.err_message}")
                    graded_result['has_errors'] = True
                    graded_result['grading_error'] = grading_task.response.err_message
                    graded_result['is_correct'] = False
                    graded_result['grading_reasoning'] = ""
                    graded_result['success'] = False
                else:
                    # Grading succeeded
                    graded_result['is_correct'] = parse_yes_no_response(grading_task.response.content)
                    graded_result['grading_reasoning'] = grading_task.response.content
                    graded_result['success'] = True

                # Save to Stage 2 JSONL
                append_to_jsonl(stage2_jsonl, graded_result)
                stage2_results_map[result_id] = graded_result

    # Save instances that had errors (from stage 1) to stage 2 with error flags (if not already saved)
    for result_id, result in stage1_results_map.items():
        if result.get('has_errors', False) and result_id not in stage2_results_map:
            error_result = result.copy()
            error_result['is_correct'] = False
            error_result['grading_reasoning'] = ""
            error_result['success'] = False
            append_to_jsonl(stage2_jsonl, error_result)
            stage2_results_map[result_id] = error_result

    if len(grading_tasks) == 0 and len(stage1_to_grade) == 0:
        print("No new answers to grade")

    # =============================================================================
    # Phase 4: Rebuild final JSON
    # =============================================================================

    print(f"\n=== Phase 4: Rebuilding final JSON ===")

    # Extract dataset name from results_path
    dataset_name = "unknown"
    try:
        path_parts = Path(results_path).parts
        if 'data' in path_parts:
            data_index = path_parts.index('data')
            if len(path_parts) > data_index + 1:
                dataset_name = path_parts[data_index + 1]
    except Exception as e:
        logger.debug(f"Failed to extract dataset name: {e}")

    # Build experiment metadata
    experiment_metadata = {
        'experiment_id': 'block_generation',
        'experiment_name': 'Block Generation with 3 Parallel Blocks',
        'experiment_date': datetime.now().isoformat(),
        'dataset': dataset_name,
        'model_type': model_type,
        'description': 'Generate reasoning in 3 parallel blocks with padding'
    }

    # Rebuild JSON from Stage 2 JSONL
    rebuild_json_from_jsonl(stage2_jsonl, output_json, experiment_metadata)

    # =============================================================================
    # Print final summary
    # =============================================================================

    with open(output_json, 'r', encoding='utf-8') as f:
        final_output = json.load(f)

    stats = final_output['summary']

    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Model:                        {model_type}")
    print(f"Total Questions:              {stats['total_questions']}")
    print(f"Results with Errors:          {stats['results_with_errors']}")
    print(f"Results without Errors:       {stats['results_without_errors']}")
    print(f"Generation Successful:        {stats['generation_successful']}")
    print(f"Generation Failed:            {stats['generation_failed']}")
    print(f"Grading Successful:           {stats['grading_successful']}")
    print(f"Grading Failed:               {stats['grading_failed']}")
    print(f"Correct Answers:              {stats['correct']}")
    print(f"Accuracy (no errors):         {stats['accuracy']:.2%}")
    print("="*60)
    print("Note: Accuracy is calculated only over instances without errors")
    print("="*60)

    print(f"\nBlocks JSONL:  {blocks_jsonl}")
    print(f"Stage 1 JSONL: {stage1_jsonl}")
    print(f"Stage 2 JSONL: {stage2_jsonl}")
    print(f"Final JSON:    {output_json}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Block Generation Experiment'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help='Path to results.json'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Path to save results'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default=MODEL_TYPE,
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
        '--max_retries',
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f'Maximum number of retries on failure (default: {DEFAULT_MAX_RETRIES})'
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    run_block_generation_experiment(
        results_path=args.results_path,
        output_path=args.output_path,
        model_type=args.model_type,
        mode=args.mode,
        limit=args.limit,
        max_workers=args.max_workers,
        max_retries=args.max_retries
    )


if __name__ == '__main__':
    main()
