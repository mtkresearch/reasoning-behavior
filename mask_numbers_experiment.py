#!/usr/bin/env python3
"""
Mask Numbers Experiment on result.traj

This script:
1. Loads results.json with result.traj as reasoning
2. Masks all numbers (0-9) with '*' in reasoning
3. Generates new answers with masked reasoning
4. Grades answers and calculates accuracy
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from llm_client import LLMClient, Task, Request, CompletionRequest
from core import (
    parse_answer_from_completion,
    parse_yes_no_response,
    build_gpt_oss_prompt_with_reasoning,
    GRADING_PROMPT
)

CONCURRENCY = 10
MAX_TRY = 3


def mask_numbers_in_reasoning(reasoning: str, mask_char: str = '█') -> str:
    """
    Mask all numbers (digits 0-9) in reasoning with specified mask character

    Args:
        reasoning: Original reasoning content
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with all digits replaced by mask_char
    """
    # Replace all digits (0-9) with mask_char
    masked_reasoning = re.sub(r'\d', mask_char, reasoning)

    return masked_reasoning


def load_results_json(results_path: str) -> List[Dict]:
    """
    Load results.json file

    Returns:
        List of result items with unique_id, question, answer, result
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_mask_task(
    item: Dict,
    model_type: str,
    mask_char: str = '█'
) -> Task:
    """
    Prepare a Task for masked reasoning generation

    Args:
        item: Result item from results.json
        model_type: Model type
        mask_char: Character to use for masking numbers (default: '█')
    """
    unique_id = item['unique_id']
    question = item['question']
    ground_truth = item['answer']
    original_reasoning = item['result']['traj']

    # Extract index from unique_id (e.g., "aime2025-I-0-2" -> 2)
    try:
        index = int(unique_id.split('-')[-1])
    except:
        index = hash(unique_id) % 10000

    # Mask all numbers in reasoning
    masked_reasoning = mask_numbers_in_reasoning(original_reasoning, mask_char)

    # Build prompt with masked reasoning
    prompt = build_gpt_oss_prompt_with_reasoning(question, masked_reasoning)

    # Create CompletionRequest
    request = CompletionRequest(
        prompt=prompt,
        model_type=model_type,
        temperature=0.5,
        max_tokens=5000
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
            'masked_reasoning': masked_reasoning,
        }
    )

    return task


def task_to_result(task: Task) -> Dict:
    """
    Convert completed Task to result dict
    """
    metadata = task.metadata
    response = task.response

    if response.success:
        generated_answer = response.content

        result = {
            'unique_id': metadata['unique_id'],
            'question_id': task.index,
            'question': metadata['question'],
            'ground_truth': metadata['ground_truth'],
            'original_reasoning': metadata['original_reasoning'],
            'masked_reasoning': metadata['masked_reasoning'],
            'generated_answer': generated_answer,
            'is_correct': None,  # Will be filled by grading
            'grading_reasoning': None,
            'success': True,
            'error': None
        }
    else:
        result = {
            'unique_id': metadata['unique_id'],
            'question_id': task.index,
            'question': metadata['question'],
            'ground_truth': metadata['ground_truth'],
            'original_reasoning': metadata['original_reasoning'],
            'masked_reasoning': metadata['masked_reasoning'],
            'generated_answer': None,
            'is_correct': False,
            'grading_reasoning': None,
            'success': False,
            'error': response.err_message
        }

    return result


def create_grading_tasks(results: List[Dict], judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create grading tasks for generated answers"""
    tasks = []

    for result in results:
        if not result.get('success', False):
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


def run_experiment(
    results_path: str,
    output_path: str,
    model_type: str = 'gpt-oss',
    mode: str = 'openrouter',
    mask_char: str = '█',
    limit: int = None
):
    """
    Run the mask numbers experiment

    Args:
        results_path: Path to results.json
        output_path: Path to save output
        model_type: Model type
        mode: 'openrouter' or 'local'
        mask_char: Character to use for masking numbers (default: '█')
        limit: Limit number of questions (for testing)
    """
    print(f"Loading results from {results_path}")
    data = load_results_json(results_path)

    if limit:
        data = data[:limit]
        print(f"Limited to {limit} questions")

    print(f"Total questions: {len(data)}")
    print(f"Mask character: '{mask_char}'")

    # Initialize LLM client
    client = LLMClient(mode=mode)

    # Prepare tasks
    print("Preparing mask tasks...")
    tasks = []
    for item in data:
        task = prepare_mask_task(item, model_type, mask_char)
        tasks.append(task)

    # Phase 1: Generate answers with masked reasoning
    results = []
    print(f"\n=== Phase 1: Generating answers with masked reasoning ({len(tasks)} tasks) ===")

    for completed_task in tqdm(client.complete_concurrent(tasks, max_workers=CONCURRENCY), total=len(tasks)):
        result = task_to_result(completed_task)
        results.append(result)

        # Save incrementally
        results_sorted = sorted(results, key=lambda x: x['question_id'])
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_sorted, f, indent=2, ensure_ascii=False)

    # Phase 2: Grade answers
    print(f"\n=== Phase 2: Grading answers ===")
    grading_tasks = create_grading_tasks(results, judge_model_type=model_type)

    print(f"Grading {len(grading_tasks)} answers...")
    for grading_task in tqdm(client.generate_concurrent(grading_tasks, max_workers=CONCURRENCY),
                             total=len(grading_tasks)):
        if not grading_task.response.success:
            print(f"\nError in grading task {grading_task.index}: {grading_task.response.err_message}")
            continue

        result_id = grading_task.metadata['result_id']
        # Find corresponding result
        for result in results:
            if result['unique_id'] == result_id:
                result['is_correct'] = parse_yes_no_response(grading_task.response.content)
                result['grading_reasoning'] = grading_task.response.content
                break

        # Save incrementally after grading
        results_sorted = sorted(results, key=lambda x: x['question_id'])
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_sorted, f, indent=2, ensure_ascii=False)

    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    correct_count = sum(1 for r in successful_results if r.get('is_correct', False))

    stats = {
        'total_questions': len(data),
        'successful': len(successful_results),
        'failed': len(data) - len(successful_results),
        'correct': correct_count,
        'accuracy': correct_count / len(successful_results) if successful_results else 0
    }

    print("\n" + "="*60)
    print("EXPERIMENT RESULTS - Masked Numbers Reasoning")
    print("="*60)
    print(f"Total Questions:     {stats['total_questions']}")
    print(f"Successful:          {stats['successful']}")
    print(f"Failed:              {stats['failed']}")
    print(f"Correct Answers:     {stats['correct']}")
    print(f"Accuracy:            {stats['accuracy']:.2%}")
    print("="*60)

    print(f"\nResults saved to: {output_path}")


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
        default='data/baseline/mask_numbers_experiment.json',
        help='Path to save results'
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
        '--mask_char',
        type=str,
        default='█',
        help='Character to use for masking numbers (default: █)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions to process (for testing)'
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    run_experiment(
        results_path=args.results_path,
        output_path=args.output_path,
        model_type=args.model_type,
        mode=args.mode,
        mask_char=args.mask_char,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
