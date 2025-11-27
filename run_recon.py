#!/usr/bin/env python3
"""
Reconstruction Experiment - run_recon.py

This script performs a three-stage reconstruction experiment:
1. Stage 0 (Reconstruction): Reconstruct complete reasoning from incomplete information
2. Stage 1 (Answer Generation): Generate answers using reconstructed reasoning
3. Stage 2 (Grading): Grade the generated answers

Usage:
    python run_recon.py \
        --result exp/mask_alphabet_mask_char/replace_s_replacement/results.json \
        --out exp/mask_alphabet_mask_char/replace_s_replacement/recon/results.json \
        --model_type gpt-oss \
        --mode openrouter \
        --limit 5

The output will be compatible with view_experiment.py for visualization.
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from datetime import datetime

from llm_client import LLMClient, Task, Request, CompletionRequest
from logger_config import setup_logger
from core import (
    parse_yes_no_response,
    build_gpt_oss_prompt_with_reasoning,
    build_gpt_oss_prompt_with_reasoning_prefilled_answer,
    GRADING_PROMPT
)
from run_experiment import (
    append_to_jsonl,
    load_from_jsonl,
    atomic_save_json,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MAX_RETRY
)

# Setup logger
logger = setup_logger(__name__, log_file='logs/run_recon.log')

# Reconstruction prompt template
RECONSTRUCTION_PROMPT = """My friend was solving a mathematical problem, as shown:
\"\"\"
{question}
\"\"\"

I picked up his manuscript. However, his handwriting was pretty bad. So what I can only recognize is some incomplete information, as shown:
\"\"\"
{broken_info}
\"\"\"

Your task is to RECONSTRUCT the complete reasoning process based on the incomplete information above.
- Focus on restoring the logical flow and intermediate steps
- Do not worry about whether the final answer is correct or not
- The goal is to reconstruct what your friend was thinking, not to solve the problem from scratch

Please write down the comprehensive reasoning process, starting with ```txt and ending with ```.
"""


# =============================================================================
# Core Utility Functions
# =============================================================================

def extract_reconstructed_reasoning(response_content: str) -> str:
    """
    Extract reconstructed reasoning from model response with fallback

    Args:
        response_content: Raw response from model

    Returns:
        Extracted reasoning text
    """
    # Try to extract from ```txt ... ``` block
    match = re.search(r'```txt\s*\n(.*?)\n```', response_content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try any code block
    match = re.search(r'```\s*\n(.*?)\n```', response_content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: use entire response
    return response_content.strip()


def build_reconstruction_prompt(question: str, broken_info: str) -> str:
    """
    Build reconstruction prompt

    Args:
        question: Original math problem
        broken_info: Incomplete/broken reasoning information

    Returns:
        Formatted prompt string
    """
    return RECONSTRUCTION_PROMPT.format(
        question=question,
        broken_info=broken_info
    )


def append_recon_to_flow(flow: str) -> str:
    """
    Append recon() to flow string

    Args:
        flow: Original flow string (e.g., "mask('number'),shuffle('line')")

    Returns:
        Flow string with recon() appended
    """
    if not flow or flow == 'N/A':
        return "recon()"
    return f"{flow},recon()"


def append_recon_to_flow_config(flow_config: List[Dict]) -> List[Dict]:
    """
    Append recon step to flow_config

    Args:
        flow_config: Original flow configuration list

    Returns:
        Flow config with recon step appended
    """
    new_config = flow_config.copy()
    next_step = len(new_config) + 1
    new_config.append({
        'step': next_step,
        'processor': 'recon',
        'params': {}
    })
    return new_config


def append_answer_retrieval_to_flow(flow: str) -> str:
    """
    Append answer('retrieval') to flow string

    Args:
        flow: Flow string (e.g., "mask('number'),recon()")

    Returns:
        Flow string with answer('retrieval') appended
    """
    return f"{flow},answer('retrieval')"


def append_answer_retrieval_to_flow_config(
    flow_config: List[Dict],
    prefill_text: str = "Thus, the answer is"
) -> List[Dict]:
    """
    Append answer step to flow_config

    Args:
        flow_config: Flow configuration list
        prefill_text: Prefill text for answer generation

    Returns:
        Flow config with answer step appended
    """
    new_config = flow_config.copy()
    next_step = len(new_config) + 1
    new_config.append({
        'step': next_step,
        'processor': 'answer',
        'params': {
            'mode': 'retrieval',
            'prefill_text': prefill_text
        }
    })
    return new_config


def load_input_results(results_path: str) -> Tuple[Dict, List[Dict]]:
    """
    Load input results.json

    Args:
        results_path: Path to input results.json

    Returns:
        Tuple of (experiment_metadata, results_list)

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('experiment_metadata', {})
    results = data.get('results', [])

    return metadata, results


# =============================================================================
# Task Preparation Functions
# =============================================================================

def prepare_reconstruction_task(item: Dict, model_type: str) -> Task:
    """
    Prepare reconstruction task (Stage 0)

    Args:
        item: Result item with processed_reasoning
        model_type: Model type to use

    Returns:
        Task for reconstruction
    """
    unique_id = item['unique_id']
    question = item['question']
    processed_reasoning = item['processed_reasoning']

    # Build reconstruction prompt
    prompt = build_reconstruction_prompt(question, processed_reasoning)

    # Create Request
    request = Request(
        queries=[prompt],
        model_type=model_type,
        system_prompt="You are a helpful assistant that reconstructs mathematical reasoning.",
        reasoning_on=False,
        temperature=0.5
    )

    # Create Task
    task = Task(
        index=item['question_id'],
        request=request,
        metadata={
            'unique_id': unique_id,
            'question': question,
            'ground_truth': item['ground_truth'],
            'original_reasoning': item['original_reasoning'],
            'processed_reasoning': processed_reasoning
        }
    )

    return task


def prepare_answer_generation_task(result: Dict, model_type: str) -> Task:
    """
    Prepare answer generation task (Stage 1)

    Args:
        result: Result with reconstructed_reasoning
        model_type: Model type to use

    Returns:
        Task for answer generation
    """
    unique_id = result['unique_id']
    question = result['question']
    reconstructed_reasoning = result['reconstructed_reasoning']

    # Check if answer_prefill is present (indicates answer retrieval mode)
    use_answer_prefill = 'answer_prefill' in result
    answer_prefill = result.get('answer_prefill', None)

    # Build prompt with reconstructed reasoning
    if use_answer_prefill:
        prompt = build_gpt_oss_prompt_with_reasoning_prefilled_answer(
            question,
            reconstructed_reasoning,
            prefill_text=answer_prefill
        )
    else:
        prompt = build_gpt_oss_prompt_with_reasoning(question, reconstructed_reasoning)

    # Create CompletionRequest
    request = CompletionRequest(
        prompt=prompt,
        model_type=model_type,
        temperature=0.5,
        max_tokens=5000
    )

    # Create Task with metadata
    metadata = {
        'unique_id': unique_id,
        'question': question,
        'ground_truth': result['ground_truth'],
        'reconstructed_reasoning': reconstructed_reasoning
    }

    # Add answer_prefill to metadata if present
    if use_answer_prefill:
        metadata['answer_prefill'] = answer_prefill

    task = Task(
        index=result['question_id'],
        request=request,
        metadata=metadata
    )

    return task


def prepare_grading_task(result: Dict, judge_model_type: str) -> Task:
    """
    Prepare grading task (Stage 2)

    Args:
        result: Result with generated_answer
        judge_model_type: Model type for grading

    Returns:
        Task for grading
    """
    task = Task(
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
    )
    return task


# =============================================================================
# Output Structure Building
# =============================================================================

def build_output_structure(
    input_metadata: Dict,
    results: List[Dict],
    use_answer_retrieval: bool = False,
    answer_prefill_text: str = "Thus, the answer is"
) -> Dict:
    """
    Build output results.json structure

    Args:
        input_metadata: Input experiment metadata
        results: List of result dictionaries
        use_answer_retrieval: Whether answer retrieval was used
        answer_prefill_text: Prefill text for answer generation

    Returns:
        Complete output structure with metadata, summary, and results
    """
    # Update flow and flow_config
    output_flow = append_recon_to_flow(input_metadata.get('flow', ''))
    output_flow_config = append_recon_to_flow_config(input_metadata.get('flow_config', []))

    # Add answer('retrieval') if enabled
    if use_answer_retrieval:
        output_flow = append_answer_retrieval_to_flow(output_flow)
        output_flow_config = append_answer_retrieval_to_flow_config(
            output_flow_config,
            prefill_text=answer_prefill_text
        )

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

    # Build experiment metadata
    experiment_metadata = {
        'experiment_name': input_metadata.get('experiment_name', 'unknown'),
        'experiment_date': datetime.now().isoformat(),
        'dataset': input_metadata.get('dataset', 'unknown'),
        'model_type': input_metadata.get('model_type', 'gpt-oss'),
        'flow': output_flow,
        'flow_config': output_flow_config
    }

    # Sort results by question_id
    results_sorted = sorted(results, key=lambda x: x.get('question_id', 0))

    return {
        'experiment_metadata': experiment_metadata,
        'summary': summary,
        'results': results_sorted
    }


# =============================================================================
# Main Experiment Function
# =============================================================================

def run_recon_experiment(
    result_path: str,
    output_path: str,
    model_type: str = 'gpt-oss',
    mode: str = 'openrouter',
    limit: int = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retry: int = DEFAULT_MAX_RETRY,
    use_answer_retrieval: bool = False,
    answer_prefill_text: str = "Thus, the answer is"
):
    """
    Run reconstruction experiment with three stages

    Args:
        result_path: Path to input results.json
        output_path: Path to save output results.json
        model_type: Model type to use
        mode: 'openrouter' or 'local'
        limit: Limit number of questions (for testing)
        max_workers: Maximum concurrent workers
        max_retry: Maximum retry attempts
        use_answer_retrieval: Whether to use answer retrieval mode
        answer_prefill_text: Prefill text for answer generation

    File Strategy:
        - Stage 0 (reconstruction): results_recon.jsonl
        - Stage 1 (generation): results_stage1.jsonl
        - Stage 2 (grading): results_stage2.jsonl
        - Final: results.json
    """
    print(f"Loading input results from {result_path}")
    input_metadata, input_results = load_input_results(result_path)

    if limit:
        input_results = input_results[:limit]
        print(f"Limited to {limit} questions")

    # Prepare file paths
    output_path = Path(output_path)
    recon_jsonl = output_path.parent / f"{output_path.stem}_recon.jsonl"
    stage1_jsonl = output_path.parent / f"{output_path.stem}_stage1.jsonl"
    stage2_jsonl = output_path.parent / f"{output_path.stem}_stage2.jsonl"
    output_json = output_path

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reconstruction JSONL: {recon_jsonl}")
    print(f"Stage 1 JSONL: {stage1_jsonl}")
    print(f"Stage 2 JSONL: {stage2_jsonl}")
    print(f"Final JSON: {output_json}")

    # Initialize LLM client
    client = LLMClient(mode=mode)

    # =========================================================================
    # Stage 0: Reconstruction
    # =========================================================================
    print(f"\n=== Stage 0: Reconstruction ===")

    # Load existing reconstruction results
    existing_recon = load_from_jsonl(recon_jsonl)
    recon_results_map = {r['unique_id']: r for r in existing_recon}

    # Filter out completed reconstructions
    need_recon = [
        item for item in input_results
        if item['unique_id'] not in recon_results_map or
           not recon_results_map[item['unique_id']].get('reconstruction_success', False)
    ]

    print(f"Already reconstructed: {len(existing_recon)}")
    print(f"Need to reconstruct: {len(need_recon)}")

    if len(need_recon) > 0:
        # Prepare reconstruction tasks
        recon_tasks = [prepare_reconstruction_task(item, model_type) for item in need_recon]

        # Execute reconstruction
        for completed_task in tqdm(client.generate_concurrent(recon_tasks, max_workers=max_workers),
                                   total=len(recon_tasks)):
            unique_id = completed_task.metadata['unique_id']
            response = completed_task.response

            if response.success and response.content:
                reconstructed_reasoning = extract_reconstructed_reasoning(response.content)
                recon_result = {
                    'unique_id': unique_id,
                    'question_id': completed_task.index,
                    'question': completed_task.metadata['question'],
                    'ground_truth': completed_task.metadata['ground_truth'],
                    'original_reasoning': completed_task.metadata['original_reasoning'],
                    'processed_reasoning': completed_task.metadata['processed_reasoning'],
                    'reconstructed_reasoning': reconstructed_reasoning,
                    'reconstruction_success': True,
                    'reconstruction_error': None
                }
            else:
                recon_result = {
                    'unique_id': unique_id,
                    'question_id': completed_task.index,
                    'question': completed_task.metadata['question'],
                    'ground_truth': completed_task.metadata['ground_truth'],
                    'original_reasoning': completed_task.metadata['original_reasoning'],
                    'processed_reasoning': completed_task.metadata['processed_reasoning'],
                    'reconstructed_reasoning': None,
                    'reconstruction_success': False,
                    'reconstruction_error': response.err_message or 'Unknown error'
                }

            # Update map and append to JSONL
            recon_results_map[unique_id] = recon_result
            append_to_jsonl(recon_jsonl, recon_result)

        # Retry failed reconstructions
        for retry_attempt in range(max_retry):
            failed_recon = [
                r for r in recon_results_map.values()
                if not r.get('reconstruction_success', False)
            ]

            if not failed_recon:
                break

            print(f"\n=== Retry attempt {retry_attempt + 1}/{max_retry}: {len(failed_recon)} failed ===")

            retry_tasks = [prepare_reconstruction_task(item, model_type) for item in failed_recon]

            for completed_task in tqdm(client.generate_concurrent(retry_tasks, max_workers=max_workers),
                                       total=len(retry_tasks)):
                unique_id = completed_task.metadata['unique_id']
                response = completed_task.response

                if response.success and response.content:
                    reconstructed_reasoning = extract_reconstructed_reasoning(response.content)
                    recon_results_map[unique_id]['reconstructed_reasoning'] = reconstructed_reasoning
                    recon_results_map[unique_id]['reconstruction_success'] = True
                    recon_results_map[unique_id]['reconstruction_error'] = None
                else:
                    recon_results_map[unique_id]['reconstruction_error'] = response.err_message

                append_to_jsonl(recon_jsonl, recon_results_map[unique_id])

    # Get successful reconstructions for next stage
    successful_recon = [
        r for r in recon_results_map.values()
        if r.get('reconstruction_success', False)
    ]

    # Add answer_prefill to successful_recon if answer retrieval is enabled
    if use_answer_retrieval:
        for r in successful_recon:
            r['answer_prefill'] = answer_prefill_text

    print(f"\nSuccessful reconstructions: {len(successful_recon)}")

    # =========================================================================
    # Stage 1: Answer Generation
    # =========================================================================
    print(f"\n=== Stage 1: Answer Generation ===")

    # Load existing stage 1 results
    existing_stage1 = load_from_jsonl(stage1_jsonl)
    stage1_results_map = {r['unique_id']: r for r in existing_stage1}

    # Filter out completed generations
    need_generation = [
        r for r in successful_recon
        if r['unique_id'] not in stage1_results_map or
           not stage1_results_map[r['unique_id']].get('generation_success', False)
    ]

    print(f"Already generated: {len(existing_stage1)}")
    print(f"Need to generate: {len(need_generation)}")

    if len(need_generation) > 0:
        # Prepare answer generation tasks
        gen_tasks = [prepare_answer_generation_task(r, model_type) for r in need_generation]

        # Execute generation
        for completed_task in tqdm(client.complete_concurrent(gen_tasks, max_workers=max_workers),
                                   total=len(gen_tasks)):
            unique_id = completed_task.metadata['unique_id']
            response = completed_task.response

            # Get base result from recon
            base_result = recon_results_map[unique_id].copy()

            if response.success and response.content:
                base_result['generated_answer'] = response.content
                base_result['generation_success'] = True
                base_result['error'] = None
            else:
                base_result['generated_answer'] = None
                base_result['generation_success'] = False
                base_result['error'] = response.err_message or 'Unknown error'

            # Initialize grading fields
            base_result['is_correct'] = None
            base_result['grading_reasoning'] = None
            base_result['success'] = False
            base_result['flow'] = append_recon_to_flow(input_metadata.get('flow', ''))

            # Update map and append to JSONL
            stage1_results_map[unique_id] = base_result
            append_to_jsonl(stage1_jsonl, base_result)

        # Retry failed generations
        for retry_attempt in range(max_retry):
            failed_gen = [
                r for r in stage1_results_map.values()
                if not r.get('generation_success', False)
            ]

            if not failed_gen:
                break

            print(f"\n=== Retry attempt {retry_attempt + 1}/{max_retry}: {len(failed_gen)} failed ===")

            retry_tasks = [prepare_answer_generation_task(r, model_type) for r in failed_gen]

            for completed_task in tqdm(client.complete_concurrent(retry_tasks, max_workers=max_workers),
                                       total=len(retry_tasks)):
                unique_id = completed_task.metadata['unique_id']
                response = completed_task.response

                if response.success and response.content:
                    stage1_results_map[unique_id]['generated_answer'] = response.content
                    stage1_results_map[unique_id]['generation_success'] = True
                    stage1_results_map[unique_id]['error'] = None
                else:
                    stage1_results_map[unique_id]['error'] = response.err_message

                append_to_jsonl(stage1_jsonl, stage1_results_map[unique_id])

    # Get successful generations for grading
    successful_gen = [
        r for r in stage1_results_map.values()
        if r.get('generation_success', False)
    ]

    print(f"\nSuccessful generations: {len(successful_gen)}")

    # =========================================================================
    # Stage 2: Grading
    # =========================================================================
    print(f"\n=== Stage 2: Grading ===")

    # Load existing stage 2 results
    existing_stage2 = load_from_jsonl(stage2_jsonl)
    stage2_results_map = {r['unique_id']: r for r in existing_stage2}

    # Filter out completed gradings
    need_grading = [
        r for r in successful_gen
        if r['unique_id'] not in stage2_results_map or
           not stage2_results_map[r['unique_id']].get('success', False)
    ]

    print(f"Already graded: {len(existing_stage2)}")
    print(f"Need to grade: {len(need_grading)}")

    if len(need_grading) > 0:
        # Prepare grading tasks
        grading_tasks = [prepare_grading_task(r, model_type) for r in need_grading]

        # Execute grading
        for completed_task in tqdm(client.generate_concurrent(grading_tasks, max_workers=max_workers),
                                   total=len(grading_tasks)):
            if not completed_task.response.success:
                logger.error(f"Grading failed for task {completed_task.index}")
                continue

            unique_id = completed_task.metadata['result_id']

            # Get base result from stage 1
            graded_result = stage1_results_map[unique_id].copy()
            graded_result['is_correct'] = parse_yes_no_response(completed_task.response.content)
            graded_result['grading_reasoning'] = completed_task.response.content
            graded_result['success'] = True

            # Update map and append to JSONL
            stage2_results_map[unique_id] = graded_result
            append_to_jsonl(stage2_jsonl, graded_result)

    # =========================================================================
    # Build Final Output
    # =========================================================================
    print(f"\n=== Building final output ===")

    # Collect all results (prioritize stage2 > stage1 > recon)
    all_results = []
    for unique_id in recon_results_map.keys():
        if unique_id in stage2_results_map:
            all_results.append(stage2_results_map[unique_id])
        elif unique_id in stage1_results_map:
            all_results.append(stage1_results_map[unique_id])
        else:
            # Add recon-only result with failure markers
            recon_only = recon_results_map[unique_id].copy()
            recon_only['generated_answer'] = None
            recon_only['is_correct'] = None
            recon_only['grading_reasoning'] = None
            recon_only['generation_success'] = False
            recon_only['success'] = False
            recon_only['error'] = 'Reconstruction failed'
            recon_only['flow'] = append_recon_to_flow(input_metadata.get('flow', ''))
            all_results.append(recon_only)

    # Build output structure
    output_data = build_output_structure(
        input_metadata,
        all_results,
        use_answer_retrieval=use_answer_retrieval,
        answer_prefill_text=answer_prefill_text
    )

    # Save final output
    atomic_save_json(output_json, output_data)

    # Print summary
    summary = output_data['summary']
    print("\n" + "="*60)
    print("RECONSTRUCTION EXPERIMENT RESULTS")
    print("="*60)
    print(f"Flow:                         {output_data['experiment_metadata']['flow']}")
    print(f"Model:                        {model_type}")
    print(f"Total Questions:              {summary['total_questions']}")
    print(f"Generation Successful:        {summary['generation_successful']}")
    print(f"Generation Failed:            {summary['generation_failed']}")
    print(f"Grading Successful:           {summary['grading_successful']}")
    print(f"Grading Failed:               {summary['grading_failed']}")
    print(f"Correct Answers:              {summary['correct']}")
    print(f"Accuracy:                     {summary['accuracy']:.2%}")
    print("="*60)
    print(f"\nResults saved to: {output_json}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reconstruction Experiment - Restore reasoning and evaluate'
    )
    parser.add_argument(
        '--result',
        type=str,
        required=True,
        help='Path to input results.json'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to save output results.json'
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
        help=f'Maximum number of retry attempts (default: {DEFAULT_MAX_RETRY})'
    )
    parser.add_argument(
        '--answer_retrieval',
        action='store_true',
        help='Use answer retrieval mode (prefill "Thus, the answer is" in answer generation)'
    )
    parser.add_argument(
        '--answer_prefill_text',
        type=str,
        default='Thus, the answer is',
        help='Prefill text for answer generation (default: "Thus, the answer is")'
    )

    args = parser.parse_args()

    # Run experiment
    run_recon_experiment(
        result_path=args.result,
        output_path=args.out,
        model_type=args.model_type,
        mode=args.mode,
        limit=args.limit,
        max_workers=args.max_workers,
        max_retry=args.max_retry,
        use_answer_retrieval=args.answer_retrieval,
        answer_prefill_text=args.answer_prefill_text
    )


if __name__ == '__main__':
    main()
