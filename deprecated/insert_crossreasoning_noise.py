"""
Insert Cross-Reasoning Noise Experiment

This script evaluates model robustness by inserting reasoning from OTHER AIME problems.

The experiment:
1. Control group: Clean reasoning results (from existing grades.json)
2. Experimental group: Insert reasoning lines from a different AIME problem
3. Compare accuracy drop to measure cross-reasoning noise influence
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict
import argparse

from llm_client import LLMClient, Task, CompletionRequest, Request
from core import (
    debug_print,
    load_existing_results,
    load_existing_grades,
    parse_answer_from_completion,
    parse_yes_no_response,
    build_gpt_oss_prompt_with_reasoning,
    extract_nonempty_lines,
    GRADING_PROMPT
)
from insert_noise_reasoning import shuffle_reasoning_lines


@dataclass
class CrossNoiseResult:
    """Result from a single question in cross-reasoning noise experiment"""
    question_id: int
    question: str
    ground_truth: str
    unique_id: str

    # Noise metadata
    noise_source_index: int
    noise_source_unique_id: str
    noise_source_problem: str
    noise_source_lines: List[str]
    noise_positions: List[int]

    # Clean results (from existing data) - only used in non-shuffle mode
    clean_answer: str = None
    is_clean_correct: bool = None

    # Shuffled results (only used in shuffle mode)
    shuffled_reasoning: str = None
    shuffled_answer: str = None
    shuffled_generation_time: float = None
    is_shuffled_correct: bool = None
    shuffled_grading_reasoning: str = None

    # Noisy results (newly generated)
    noisy_reasoning: str = None
    noisy_answer: str = None
    noisy_generation_time: float = None
    is_noisy_correct: bool = None
    noisy_grading_reasoning: str = None


def parse_problem_id(unique_id: str) -> str:
    """
    Extract problem ID from unique_id

    Args:
        unique_id: e.g., "aime2025-I-0-0"

    Returns:
        problem_id: e.g., "aime2025-I-0"
    """
    parts = unique_id.split('-')
    return f"{parts[0]}-{parts[1]}-{parts[2]}"


def select_noise_source(all_results: List[Dict], current_idx: int, seed: int = 42) -> int:
    """
    Select ONE other question as noise source from a DIFFERENT AIME problem

    Args:
        all_results: All items from results.json
        current_idx: Current item index in results.json
        seed: Random seed

    Returns:
        source_index: Index of selected noise source
    """
    current_uid = all_results[current_idx]['unique_id']
    current_problem = parse_problem_id(current_uid)

    # Find all indices with DIFFERENT problem
    candidates = []
    for i, item in enumerate(all_results):
        uid = item['unique_id']
        problem = parse_problem_id(uid)
        if problem != current_problem:
            candidates.append(i)

    # Randomly select one (deterministic per question)
    random.seed(seed + current_idx)
    source_index = random.choice(candidates)

    return source_index


def sample_reasoning_lines(reasoning: str, num_lines: int) -> List[str]:
    """
    Randomly sample N lines from reasoning

    Args:
        reasoning: Source reasoning text
        num_lines: Number of lines to sample

    Returns:
        List of sampled lines
    """
    lines = extract_nonempty_lines(reasoning)

    if len(lines) == 0:
        return []

    if len(lines) <= num_lines:
        # If not enough lines, allow repeated sampling
        needed = num_lines // len(lines) + 1
        extended_lines = lines * needed
        return random.sample(extended_lines, num_lines)
    else:
        # No repeated sampling
        return random.sample(lines, num_lines)


def insert_noise_lines(reasoning: str, noise_lines: List[str]) -> Tuple[str, List[int]]:
    """
    Insert noise lines at random positions in reasoning

    Args:
        reasoning: Original reasoning text
        noise_lines: Lines to insert as noise

    Returns:
        (noisy_reasoning, insertion_positions)
    """
    lines = extract_nonempty_lines(reasoning)
    positions = []

    for noise_line in noise_lines:
        # Random position (0 to len(lines))
        pos = random.randint(0, len(lines))
        lines.insert(pos, noise_line)
        positions.append(pos)

    return '\n'.join(lines), positions


def create_grading_tasks(client: LLMClient, results: List[CrossNoiseResult],
                        shuffle_enabled: bool = False,
                        judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create grading tasks for answers"""
    tasks = []

    if shuffle_enabled:
        # Grade both shuffled and shuffled_noisy answers
        for result in results:
            # Grade shuffled answer
            tasks.append(Task(
                index=f"{result.question_id}_shuffled",
                request=Request(
                    queries=[GRADING_PROMPT.format(
                        problem=result.question,
                        ground_truth=result.ground_truth,
                        model_answer=result.shuffled_answer
                    )],
                    model_type=judge_model_type,
                    system_prompt="You are a helpful mathematical grading assistant.",
                    reasoning_on=False,
                    temperature=0.01
                ),
                metadata={'result': result, 'type': 'shuffled'}
            ))

            # Grade shuffled_noisy answer
            tasks.append(Task(
                index=f"{result.question_id}_shuffled_noisy",
                request=Request(
                    queries=[GRADING_PROMPT.format(
                        problem=result.question,
                        ground_truth=result.ground_truth,
                        model_answer=result.noisy_answer
                    )],
                    model_type=judge_model_type,
                    system_prompt="You are a helpful mathematical grading assistant.",
                    reasoning_on=False,
                    temperature=0.01
                ),
                metadata={'result': result, 'type': 'shuffled_noisy'}
            ))
    else:
        # Grade only noisy answers
        for result in results:
            tasks.append(Task(
                index=f"{result.question_id}_noisy",
                request=Request(
                    queries=[GRADING_PROMPT.format(
                        problem=result.question,
                        ground_truth=result.ground_truth,
                        model_answer=result.noisy_answer
                    )],
                    model_type=judge_model_type,
                    system_prompt="You are a helpful mathematical grading assistant.",
                    reasoning_on=False,
                    temperature=0.01
                ),
                metadata={'result': result, 'type': 'noisy'}
            ))

    return tasks


def run_cross_reasoning_experiment(results_path: str, grades_path: str, output_path: str,
                                   num_samples: int = None, seed: int = 42,
                                   num_insertions: int = 5,
                                   shuffle_enabled: bool = False,
                                   judge_model_type: str = 'gpt-oss',
                                   max_workers: int = 50):
    """
    Run the cross-reasoning noise experiment

    Args:
        results_path: Path to existing results.json
        grades_path: Path to existing grades.json
        output_path: Path to save results
        num_samples: Number of samples to test (None = all)
        seed: Random seed for noise selection and shuffling
        num_insertions: Number of lines to insert from noise source
        shuffle_enabled: If True, compare shuffled vs shuffled+noisy; if False, compare clean vs noisy
        judge_model_type: Model to use for judging
        max_workers: Maximum number of concurrent workers
    """
    random.seed(seed)

    # Load data
    print("Loading data...")
    existing_results = load_existing_results(results_path)
    grade_map = load_existing_grades(grades_path)

    if num_samples:
        existing_results = existing_results[:num_samples]

    print(f"\nRunning cross-reasoning noise experiment on {len(existing_results)} items...")
    print(f"Mode: {'Shuffle' if shuffle_enabled else 'Normal'}")
    print(f"Number of noise lines to insert: {num_insertions}")
    print(f"Random seed: {seed}")

    # Create LLM client
    client = LLMClient()

    # Phase 1: Prepare generation tasks
    if shuffle_enabled:
        print("\n=== Phase 1: Preparing shuffled and shuffled+noisy generation tasks ===")
    else:
        print("\n=== Phase 1: Preparing noisy generation tasks ===")

    generation_tasks = []
    noise_metadata = []  # Store noise metadata for later use

    for idx, item in enumerate(existing_results):
        question = item['question']
        ground_truth = item['answer']
        unique_id = item['unique_id']

        # Get full reasoning
        full_reasoning = item['result']['traj']

        # Select noise source from different AIME problem
        noise_source_idx = select_noise_source(existing_results, idx, seed)
        noise_source_uid = existing_results[noise_source_idx]['unique_id']
        noise_source_problem = parse_problem_id(noise_source_uid)
        noise_source_reasoning = existing_results[noise_source_idx]['result']['traj']

        # Sample lines from noise source
        noise_lines = sample_reasoning_lines(noise_source_reasoning, num_insertions)

        # Store metadata
        metadata_entry = {
            'question_id': idx,
            'question': question,
            'ground_truth': ground_truth,
            'unique_id': unique_id,
            'noise_source_index': noise_source_idx,
            'noise_source_unique_id': noise_source_uid,
            'noise_source_problem': noise_source_problem,
            'noise_source_lines': noise_lines
        }

        if shuffle_enabled:
            # Shuffle mode: create shuffled and shuffled+noisy tasks

            # 1. Shuffled reasoning (without noise)
            shuffled_reasoning = shuffle_reasoning_lines(full_reasoning)
            shuffled_prompt = build_gpt_oss_prompt_with_reasoning(question, shuffled_reasoning)

            generation_tasks.append(Task(
                index=f"{idx}_shuffled",
                request=CompletionRequest(
                    prompt=shuffled_prompt,
                    model_type='gpt-oss',
                    temperature=0.6,
                    max_tokens=20480
                ),
                metadata={
                    **metadata_entry,
                    'type': 'shuffled',
                    'shuffled_reasoning': shuffled_reasoning
                }
            ))

            # 2. Shuffled + noisy reasoning
            shuffled_noisy_reasoning, noise_positions = insert_noise_lines(shuffled_reasoning, noise_lines)
            shuffled_noisy_prompt = build_gpt_oss_prompt_with_reasoning(question, shuffled_noisy_reasoning)

            generation_tasks.append(Task(
                index=f"{idx}_shuffled_noisy",
                request=CompletionRequest(
                    prompt=shuffled_noisy_prompt,
                    model_type='gpt-oss',
                    temperature=0.6,
                    max_tokens=20480
                ),
                metadata={
                    **metadata_entry,
                    'type': 'shuffled_noisy',
                    'noisy_reasoning': shuffled_noisy_reasoning,
                    'noise_positions': noise_positions
                }
            ))

        else:
            # Normal mode: only create noisy task (clean from grades.json)
            clean_answer = item['result']['answer']
            is_clean_correct = grade_map.get(idx, False)

            # Insert noise lines into reasoning
            noisy_reasoning, noise_positions = insert_noise_lines(full_reasoning, noise_lines)

            # Build prompt with noisy reasoning
            noisy_prompt = build_gpt_oss_prompt_with_reasoning(question, noisy_reasoning)

            # Create generation task
            generation_tasks.append(Task(
                index=f"{idx}_noisy",
                request=CompletionRequest(
                    prompt=noisy_prompt,
                    model_type='gpt-oss',
                    temperature=0.6,
                    max_tokens=20480
                ),
                metadata={
                    **metadata_entry,
                    'clean_answer': clean_answer,
                    'is_clean_correct': is_clean_correct,
                    'noisy_reasoning': noisy_reasoning,
                    'noise_positions': noise_positions
                }
            ))

        noise_metadata.append(metadata_entry)

    # Phase 2: Generate answers
    print(f"\n=== Phase 2: Generating answers ({len(generation_tasks)} tasks) ===")

    if shuffle_enabled:
        # Store intermediate results by question_id
        generation_results = {}

        for task in tqdm(client.complete_concurrent(generation_tasks, max_workers=max_workers),
                         total=len(generation_tasks), desc="Generating"):
            if not task.response.success:
                print(f"\nError in generation task {task.index}: {task.response.err_message}")
                continue

            question_id = task.metadata['question_id']
            task_type = task.metadata['type']
            answer = parse_answer_from_completion(task.response.content)

            if question_id not in generation_results:
                generation_results[question_id] = {
                    'question': task.metadata['question'],
                    'ground_truth': task.metadata['ground_truth'],
                    'unique_id': task.metadata['unique_id'],
                    'noise_source_index': task.metadata['noise_source_index'],
                    'noise_source_unique_id': task.metadata['noise_source_unique_id'],
                    'noise_source_problem': task.metadata['noise_source_problem'],
                    'noise_source_lines': task.metadata['noise_source_lines']
                }

            if task_type == 'shuffled':
                generation_results[question_id]['shuffled'] = {
                    'reasoning': task.metadata['shuffled_reasoning'],
                    'answer': answer,
                    'time': task.response.elapsed_seconds
                }
            else:  # shuffled_noisy
                generation_results[question_id]['shuffled_noisy'] = {
                    'reasoning': task.metadata['noisy_reasoning'],
                    'answer': answer,
                    'time': task.response.elapsed_seconds,
                    'noise_positions': task.metadata['noise_positions']
                }

        # Build result objects
        results = []
        for question_id in sorted(generation_results.keys()):
            data = generation_results[question_id]

            if 'shuffled' not in data or 'shuffled_noisy' not in data:
                print(f"Warning: Missing data for question {question_id}")
                continue

            result = CrossNoiseResult(
                question_id=question_id,
                question=data['question'],
                ground_truth=data['ground_truth'],
                unique_id=data['unique_id'],
                noise_source_index=data['noise_source_index'],
                noise_source_unique_id=data['noise_source_unique_id'],
                noise_source_problem=data['noise_source_problem'],
                noise_source_lines=data['noise_source_lines'],
                shuffled_reasoning=data['shuffled']['reasoning'],
                shuffled_answer=data['shuffled']['answer'],
                shuffled_generation_time=data['shuffled']['time'],
                noise_positions=data['shuffled_noisy']['noise_positions'],
                noisy_reasoning=data['shuffled_noisy']['reasoning'],
                noisy_answer=data['shuffled_noisy']['answer'],
                noisy_generation_time=data['shuffled_noisy']['time']
            )
            results.append(result)

    else:
        # Normal mode
        results = []

        for task in tqdm(client.complete_concurrent(generation_tasks, max_workers=max_workers),
                         total=len(generation_tasks), desc="Generating"):
            if not task.response.success:
                print(f"\nError in generation task {task.index}: {task.response.err_message}")
                continue

            # Parse answer
            noisy_answer = parse_answer_from_completion(task.response.content)

            # Create result object
            result = CrossNoiseResult(
                question_id=task.metadata['question_id'],
                question=task.metadata['question'],
                ground_truth=task.metadata['ground_truth'],
                unique_id=task.metadata['unique_id'],
                noise_source_index=task.metadata['noise_source_index'],
                noise_source_unique_id=task.metadata['noise_source_unique_id'],
                noise_source_problem=task.metadata['noise_source_problem'],
                noise_source_lines=task.metadata['noise_source_lines'],
                clean_answer=task.metadata['clean_answer'],
                is_clean_correct=task.metadata['is_clean_correct'],
                noise_positions=task.metadata['noise_positions'],
                noisy_reasoning=task.metadata['noisy_reasoning'],
                noisy_answer=noisy_answer,
                noisy_generation_time=task.response.elapsed_seconds
            )

            results.append(result)

    # Phase 3: Grade answers
    print(f"\n=== Phase 3: Grading answers ({len(results) * (2 if shuffle_enabled else 1)} tasks) ===")
    grading_tasks = create_grading_tasks(client, results, shuffle_enabled, judge_model_type)

    for task in tqdm(client.generate_concurrent(grading_tasks, max_workers=max_workers),
                     total=len(grading_tasks), desc="Grading"):
        if not task.response.success:
            print(f"\nError in grading task {task.index}: {task.response.err_message}")
            continue

        result = task.metadata['result']
        task_type = task.metadata['type']
        is_correct = parse_yes_no_response(task.response.content)

        if task_type == 'shuffled':
            result.is_shuffled_correct = is_correct
            result.shuffled_grading_reasoning = task.response.content
        elif task_type == 'shuffled_noisy' or task_type == 'noisy':
            result.is_noisy_correct = is_correct
            result.noisy_grading_reasoning = task.response.content

    # Phase 4: Calculate summary and save results
    print("\n=== Phase 4: Saving results ===")

    total = len(results)

    if shuffle_enabled:
        # Shuffle mode: compare shuffled vs shuffled_noisy
        shuffled_correct_count = sum(1 for r in results if r.is_shuffled_correct)
        shuffled_noisy_correct_count = sum(1 for r in results if r.is_noisy_correct)

        summary = {
            'total_problems': total,
            'shuffled_correct': {
                'count': shuffled_correct_count,
                'percentage': round(shuffled_correct_count / total * 100, 2) if total > 0 else 0
            },
            'shuffled_noisy_correct': {
                'count': shuffled_noisy_correct_count,
                'percentage': round(shuffled_noisy_correct_count / total * 100, 2) if total > 0 else 0
            },
            'difference': {
                'absolute': shuffled_correct_count - shuffled_noisy_correct_count,
                'percentage': round((shuffled_correct_count - shuffled_noisy_correct_count) / total * 100, 2) if total > 0 else 0
            }
        }
    else:
        # Normal mode: compare clean vs noisy
        clean_correct_count = sum(1 for r in results if r.is_clean_correct)
        noisy_correct_count = sum(1 for r in results if r.is_noisy_correct)

        summary = {
            'total_problems': total,
            'clean_correct': {
                'count': clean_correct_count,
                'percentage': round(clean_correct_count / total * 100, 2) if total > 0 else 0
            },
            'noisy_correct': {
                'count': noisy_correct_count,
                'percentage': round(noisy_correct_count / total * 100, 2) if total > 0 else 0
            },
            'difference': {
                'absolute': clean_correct_count - noisy_correct_count,
                'percentage': round((clean_correct_count - noisy_correct_count) / total * 100, 2) if total > 0 else 0
            }
        }

    config = {
        'noise_strategy': 'cross-reasoning',
        'num_insertions': num_insertions,
        'shuffle_enabled': shuffle_enabled,
        'temperature': 0.6,
        'seed': seed
    }

    output_data = {
        'config': config,
        'summary': summary,
        'results': [asdict(r) for r in results]
    }

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n✓ Experiment complete. Results saved to {output_path}")
    print(f"\n{'='*80}")
    print("CROSS-REASONING NOISE EXPERIMENT - SUMMARY")
    print(f"{'='*80}")
    print(f"Mode: {'Shuffle' if shuffle_enabled else 'Normal'}")
    print(f"Total items: {total}")
    print(f"Noise strategy: Cross-reasoning from different AIME problems")
    print(f"Number of noise lines inserted: {num_insertions}")
    print(f"Random seed: {seed}")
    print(f"\nAccuracy:")

    if shuffle_enabled:
        print(f"  Shuffled:        {shuffled_correct_count}/{total} ({summary['shuffled_correct']['percentage']:.2f}%)")
        print(f"  Shuffled+Noisy:  {shuffled_noisy_correct_count}/{total} ({summary['shuffled_noisy_correct']['percentage']:.2f}%)")
    else:
        print(f"  Clean (original):  {clean_correct_count}/{total} ({summary['clean_correct']['percentage']:.2f}%)")
        print(f"  Noisy (cross-reasoning):  {noisy_correct_count}/{total} ({summary['noisy_correct']['percentage']:.2f}%)")

    print(f"\nDifference:")
    print(f"  Absolute: {summary['difference']['absolute']}")
    print(f"  Percentage: {summary['difference']['percentage']:+.2f}%")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Reasoning Noise Experiment")
    parser.add_argument("--results_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/results.json",
                       help="Path to existing results.json")
    parser.add_argument("--grades_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/grades.json",
                       help="Path to existing grades.json")
    parser.add_argument("--output_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/insert-crossreasoning-noise.json",
                       help="Output path for results")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to test (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for noise selection (default: 42)")
    parser.add_argument("--num_insertions", type=int, default=5,
                       help="Number of lines to insert from noise source (default: 5)")
    parser.add_argument("--shuffle", action='store_true',
                       help="Enable shuffle mode: compare shuffled vs shuffled+noisy (default: False)")
    parser.add_argument("--judge_model_type", type=str, default='gpt-oss',
                       help="Model type for judging (default: gpt-oss)")
    parser.add_argument("--max_workers", type=int, default=50,
                       help="Maximum number of concurrent workers (default: 50)")

    args = parser.parse_args()

    # Run experiment
    results = run_cross_reasoning_experiment(
        results_path=args.results_path,
        grades_path=args.grades_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed,
        num_insertions=args.num_insertions,
        shuffle_enabled=args.shuffle,
        judge_model_type=args.judge_model_type,
        max_workers=args.max_workers
    )

    print(f"\n✓ Cross-reasoning noise experiment complete!")
    print(f"✓ Results saved to {args.output_path}")
    print(f"✓ Total items processed: {len(results)}")


if __name__ == "__main__":
    main()
