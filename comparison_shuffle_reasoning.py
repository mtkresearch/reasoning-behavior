"""
Comparison experiment: Normal reasoning vs Shuffled reasoning (VLLM version)

This script compares model performance when:
1. Control group: Normal reasoning (truncated from full reasoning)
2. Experimental group: Shuffled reasoning (line-by-line shuffle)

Uses VLLM via llm_client for parallel execution and efficiency.
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict
import argparse

from llm_client import LLMClient, Task, Request, CompletionRequest
from core import (
    debug_print,
    load_existing_results,
    load_existing_grades,
    parse_answer_from_completion,
    parse_yes_no_response,
    clean_multiple_newlines,
    build_gpt_oss_prompt_with_reasoning,
    GRADING_PROMPT,
    SAME_ANSWER_PROMPT
)

# Global tokenizer cache
_TOKENIZER_CACHE = {}


@dataclass
class ExperimentResult:
    """Result from a single question"""
    question_id: int
    question: str
    ground_truth: str

    # Full reasoning results (from existing results.json)
    full_reasoning: str
    full_answer: str
    full_generation_time: float

    # Normal reasoning results (after truncation)
    normal_reasoning: str
    normal_answer: str
    normal_generation_time: float

    # Shuffled reasoning results
    shuffled_reasoning: str
    shuffled_answer: str
    shuffled_generation_time: float

    # Evaluation metrics
    is_same_answer: bool = None
    is_full_correct: bool = None
    is_normal_correct: bool = None
    is_shuffle_correct: bool = None

    # Evaluation reasoning
    same_answer_reasoning: str = None
    full_correct_reasoning: str = None
    normal_correct_reasoning: str = None
    shuffle_correct_reasoning: str = None


def truncate_reasoning_lines(reasoning: str, del_last_line: float) -> str:
    """
    Remove last n lines or ratio of lines from reasoning content

    Args:
        reasoning: Original reasoning content
        del_last_line: If >= 1, number of lines to remove from the end (integer)
                      If 0 < del_last_line < 1, ratio of lines to remove (float)

    Returns:
        Truncated reasoning content
    """
    if del_last_line <= 0:
        return reasoning

    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Calculate number of lines to remove
    if del_last_line < 1:
        # Ratio mode: remove a percentage of lines
        lines_to_remove = int(len(lines) * del_last_line)
    else:
        # Count mode: remove specific number of lines
        lines_to_remove = int(del_last_line)

    # Remove last n lines
    if lines_to_remove >= len(lines):
        # If trying to remove more lines than available, return first line
        return lines[0] if lines else ""

    if lines_to_remove == 0:
        return reasoning

    truncated_lines = lines[:-lines_to_remove]
    return '\n'.join(truncated_lines)


def shuffle_reasoning_lines(reasoning: str) -> str:
    """
    Shuffle reasoning content line-by-line

    Args:
        reasoning: Original reasoning content

    Returns:
        Shuffled reasoning content
    """
    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Shuffle
    random.shuffle(lines)

    return '\n'.join(lines)


def shuffle_reasoning_words(reasoning: str) -> str:
    """
    Shuffle reasoning content word-by-word

    Args:
        reasoning: Original reasoning content

    Returns:
        Shuffled reasoning content with words shuffled
    """
    # Split by whitespace to get words
    words = reasoning.split()

    # Shuffle words
    random.shuffle(words)

    # Rejoin with single space
    return ' '.join(words)


def shuffle_reasoning_tokens(reasoning: str, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") -> str:
    """
    Shuffle reasoning content token-by-token using the model's tokenizer

    Args:
        reasoning: Original reasoning content
        model_name: Model name for tokenizer (default: gpt-oss model)

    Returns:
        Shuffled reasoning content with tokens shuffled
    """
    from transformers import AutoTokenizer

    # Load tokenizer (with caching to avoid reloading)
    if model_name not in _TOKENIZER_CACHE:
        debug_print(f"Loading tokenizer for {model_name}...")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    tokenizer = _TOKENIZER_CACHE[model_name]

    # Tokenize
    tokens = tokenizer.encode(reasoning, add_special_tokens=False)

    # Shuffle token IDs
    random.shuffle(tokens)

    # Decode back to text
    shuffled_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return shuffled_text


def shuffle_reasoning(reasoning: str, shuffle_type: str = 'line',
                     tokenizer_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") -> str:
    """
    Shuffle reasoning content using specified method

    Args:
        reasoning: Original reasoning content
        shuffle_type: Type of shuffle - 'line', 'word', or 'token'
        tokenizer_model: Model name for tokenizer (only used for token shuffle)

    Returns:
        Shuffled reasoning content
    """
    if shuffle_type == 'line':
        return shuffle_reasoning_lines(reasoning)
    elif shuffle_type == 'word':
        return shuffle_reasoning_words(reasoning)
    elif shuffle_type == 'token':
        return shuffle_reasoning_tokens(reasoning, tokenizer_model)
    else:
        raise ValueError(f"Invalid shuffle_type: {shuffle_type}. Must be 'line', 'word', or 'token'")


def create_grading_tasks(client: LLMClient, results: List[ExperimentResult],
                        judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create grading tasks for both normal and shuffled answers"""
    tasks = []

    for result in results:
        # Normal answer grading
        tasks.append(Task(
            index=result.question_id * 2,  # Even index for normal
            request=Request(
                queries=[GRADING_PROMPT.format(
                    problem=result.question,
                    ground_truth=result.ground_truth,
                    model_answer=result.normal_answer
                )],
                model_type=judge_model_type,
                system_prompt="You are a helpful mathematical grading assistant.",
                reasoning_on=False,
                temperature=0.01
            ),
            metadata={'type': 'normal', 'result': result}
        ))

        # Shuffled answer grading
        tasks.append(Task(
            index=result.question_id * 2 + 1,  # Odd index for shuffled
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
            metadata={'type': 'shuffled', 'result': result}
        ))

    return tasks


def create_same_answer_tasks(client: LLMClient, results: List[ExperimentResult],
                             judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create tasks for checking if normal and shuffled answers are the same"""
    tasks = []

    for result in results:
        # Skip if both are correct (they must be the same)
        if result.is_normal_correct and result.is_shuffle_correct:
            continue

        tasks.append(Task(
            index=result.question_id,
            request=Request(
                queries=[SAME_ANSWER_PROMPT.format(
                    answer1=result.normal_answer,
                    answer2=result.shuffled_answer
                )],
                model_type=judge_model_type,
                system_prompt="You are a helpful mathematical assistant.",
                reasoning_on=False,
                temperature=0.01
            ),
            metadata={'result': result}
        ))

    return tasks


def run_experiment(results_path: str, grades_path: str, output_dir: str,
                  num_samples: int = None, seed: int = 42, del_last_line: float = 0,
                  judge_model_type: str = 'gpt-oss', max_workers: int = 50,
                  shuffle_type: str = 'line', tokenizer_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                  empty_question: bool = False):
    """
    Run the comparison experiment using VLLM

    Args:
        results_path: Path to existing results.json
        grades_path: Path to existing grades.json
        output_dir: Directory to save results
        num_samples: Number of samples to test (None = all)
        seed: Random seed for shuffling
        del_last_line: Number of lines to remove from end (>=1) or ratio of lines (0-1)
        judge_model_type: Model to use for judging
        max_workers: Maximum number of concurrent workers
        shuffle_type: Type of shuffle - 'line', 'word', or 'token'
        tokenizer_model: Model name for tokenizer (only used for token shuffle)
        empty_question: If True, replace question with empty string in prompts
    """
    random.seed(seed)

    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    existing_results = load_existing_results(results_path)
    grade_map = load_existing_grades(grades_path)

    # Match dataset with results
    if num_samples:
        existing_results = existing_results[:num_samples]

    print(f"\nRunning experiment on {len(existing_results)} problems...")

    # Create LLM client
    client = LLMClient()

    # Step 1: Create all generation tasks for normal and shuffled reasoning
    print("\n=== Phase 1: Preparing generation tasks ===")
    generation_tasks = []
    temp_results = []

    for idx, item in enumerate(existing_results):
        question = item['question']
        ground_truth = item['answer']

        # Get full reasoning and answer from existing results
        full_reasoning = item['result']['traj']
        full_answer = item['result']['answer']
        full_time = item['result']['elapsed_seconds']
        is_full_correct = grade_map.get(idx, False)

        # Clean reasoning
        full_reasoning = clean_multiple_newlines(full_reasoning)

        # Truncate reasoning
        normal_reasoning = truncate_reasoning_lines(full_reasoning, del_last_line)

        # Shuffle reasoning using specified method
        shuffled_reasoning = shuffle_reasoning(normal_reasoning, shuffle_type, tokenizer_model)

        # Build prompts for text completion
        normal_prompt = build_gpt_oss_prompt_with_reasoning(question, normal_reasoning, empty_question=empty_question)
        shuffled_prompt = build_gpt_oss_prompt_with_reasoning(question, shuffled_reasoning, empty_question=empty_question)

        # Create tasks
        generation_tasks.append(Task(
            index=idx * 2,  # Even for normal
            request=CompletionRequest(
                prompt=normal_prompt,
                model_type='gpt-oss',
                temperature=0.6,
                max_tokens=20480
            ),
            metadata={
                'type': 'normal',
                'question_id': idx,
                'question': question,
                'ground_truth': ground_truth,
                'full_reasoning': full_reasoning,
                'full_answer': full_answer,
                'full_time': full_time,
                'is_full_correct': is_full_correct,
                'normal_reasoning': normal_reasoning,
                'shuffled_reasoning': shuffled_reasoning
            }
        ))

        generation_tasks.append(Task(
            index=idx * 2 + 1,  # Odd for shuffled
            request=CompletionRequest(
                prompt=shuffled_prompt,
                model_type='gpt-oss',
                temperature=0.6,
                max_tokens=20480
            ),
            metadata={
                'type': 'shuffled',
                'question_id': idx
            }
        ))

    # Step 2: Run generation tasks in parallel
    print(f"\n=== Phase 2: Generating answers ({len(generation_tasks)} tasks) ===")
    generation_results = {}

    for task in tqdm(client.complete_concurrent(generation_tasks, max_workers=max_workers),
                     total=len(generation_tasks), desc="Generating"):
        if not task.response.success:
            print(f"\nError in generation task {task.index}: {task.response.err_message}")
            continue

        question_id = task.metadata['question_id']
        task_type = task.metadata['type']

        # Parse answer
        answer = parse_answer_from_completion(task.response.content)

        # Store results
        if question_id not in generation_results:
            generation_results[question_id] = {}

        generation_results[question_id][task_type] = {
            'answer': answer,
            'time': task.response.elapsed_seconds
        }

        # If this is the normal task, also store metadata
        if task_type == 'normal':
            generation_results[question_id]['metadata'] = task.metadata

    # Step 3: Build ExperimentResult objects
    print("\n=== Phase 3: Building result objects ===")
    results = []

    for question_id in sorted(generation_results.keys()):
        data = generation_results[question_id]

        if 'normal' not in data or 'shuffled' not in data:
            print(f"Warning: Missing data for question {question_id}")
            continue

        metadata = data['metadata']

        result = ExperimentResult(
            question_id=question_id,
            question=metadata['question'],
            ground_truth=metadata['ground_truth'],
            full_reasoning=metadata['full_reasoning'],
            full_answer=metadata['full_answer'],
            full_generation_time=metadata['full_time'],
            normal_reasoning=metadata['normal_reasoning'],
            normal_answer=data['normal']['answer'],
            normal_generation_time=data['normal']['time'],
            shuffled_reasoning=metadata['shuffled_reasoning'],
            shuffled_answer=data['shuffled']['answer'],
            shuffled_generation_time=data['shuffled']['time'],
            is_full_correct=metadata['is_full_correct']
        )

        results.append(result)

    # Step 4: Grade normal and shuffled answers
    print(f"\n=== Phase 4: Grading answers ({len(results) * 2} tasks) ===")
    grading_tasks = create_grading_tasks(client, results, judge_model_type)

    for task in tqdm(client.generate_concurrent(grading_tasks, max_workers=max_workers),
                     total=len(grading_tasks), desc="Grading"):
        if not task.response.success:
            print(f"\nError in grading task {task.index}: {task.response.err_message}")
            continue

        result = task.metadata['result']
        task_type = task.metadata['type']
        is_correct = parse_yes_no_response(task.response.content)

        if task_type == 'normal':
            result.is_normal_correct = is_correct
            result.normal_correct_reasoning = task.response.content
        else:  # shuffled
            result.is_shuffle_correct = is_correct
            result.shuffle_correct_reasoning = task.response.content

    # Step 5: Check if answers are the same
    print(f"\n=== Phase 5: Checking answer equivalence ===")

    # First, set is_same for cases where both are correct
    for result in results:
        if result.is_normal_correct and result.is_shuffle_correct:
            result.is_same_answer = True
            result.same_answer_reasoning = "Both answers are correct, therefore they must be the same."

    # Create tasks for remaining cases
    same_answer_tasks = create_same_answer_tasks(client, results, judge_model_type)

    if same_answer_tasks:
        for task in tqdm(client.generate_concurrent(same_answer_tasks, max_workers=max_workers),
                        total=len(same_answer_tasks), desc="Checking equivalence"):
            if not task.response.success:
                print(f"\nError in same_answer task {task.index}: {task.response.err_message}")
                continue

            result = task.metadata['result']
            is_same = parse_yes_no_response(task.response.content)

            result.is_same_answer = is_same
            result.same_answer_reasoning = task.response.content

    # Step 6: Save results
    print("\n=== Phase 6: Saving results ===")

    # Calculate summary statistics
    total = len(results)
    same_answer_count = sum(1 for r in results if r.is_same_answer)
    full_correct_count = sum(1 for r in results if r.is_full_correct)
    normal_correct_count = sum(1 for r in results if r.is_normal_correct)
    shuffle_correct_count = sum(1 for r in results if r.is_shuffle_correct)

    summary = {
        'total_problems': total,
        'del_last_line': del_last_line,
        'shuffle_type': shuffle_type,
        'empty_question': empty_question,
        'same_answer': {
            'count': same_answer_count,
            'percentage': round(same_answer_count / total * 100, 2) if total > 0 else 0
        },
        'full_correct': {
            'count': full_correct_count,
            'percentage': round(full_correct_count / total * 100, 2) if total > 0 else 0
        },
        'normal_correct': {
            'count': normal_correct_count,
            'percentage': round(normal_correct_count / total * 100, 2) if total > 0 else 0
        },
        'shuffle_correct': {
            'count': shuffle_correct_count,
            'percentage': round(shuffle_correct_count / total * 100, 2) if total > 0 else 0
        },
        'difference': {
            'normal_vs_shuffle': {
                'absolute': normal_correct_count - shuffle_correct_count,
                'percentage': round((normal_correct_count - shuffle_correct_count) / total * 100, 2) if total > 0 else 0
            },
            'full_vs_normal': {
                'absolute': full_correct_count - normal_correct_count,
                'percentage': round((full_correct_count - normal_correct_count) / total * 100, 2) if total > 0 else 0
            },
            'full_vs_shuffle': {
                'absolute': full_correct_count - shuffle_correct_count,
                'percentage': round((full_correct_count - shuffle_correct_count) / total * 100, 2) if total > 0 else 0
            }
        }
    }

    # Save experiment results
    results_path = output_dir / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n✓ Experiment complete. Results saved to {output_dir}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total problems: {total}")
    print(f"Shuffle type: {shuffle_type}")
    print(f"Empty question: {empty_question}")
    if del_last_line < 1:
        print(f"Deleted last {del_last_line*100:.1f}% of lines from reasoning")
    else:
        print(f"Deleted last {int(del_last_line)} lines from reasoning")
    print(f"\nSame answer: {same_answer_count}/{total} ({summary['same_answer']['percentage']:.2f}%)")
    print(f"\nCorrectness:")
    print(f"  Full correct:    {full_correct_count}/{total} ({summary['full_correct']['percentage']:.2f}%)")
    print(f"  Normal correct:  {normal_correct_count}/{total} ({summary['normal_correct']['percentage']:.2f}%)")
    print(f"  Shuffle correct: {shuffle_correct_count}/{total} ({summary['shuffle_correct']['percentage']:.2f}%)")
    print(f"\nDifferences:")
    print(f"  Normal vs Shuffle: {summary['difference']['normal_vs_shuffle']['absolute']} ({summary['difference']['normal_vs_shuffle']['percentage']:+.2f}%)")
    print(f"  Full vs Normal:    {summary['difference']['full_vs_normal']['absolute']} ({summary['difference']['full_vs_normal']['percentage']:+.2f}%)")
    print(f"  Full vs Shuffle:   {summary['difference']['full_vs_shuffle']['absolute']} ({summary['difference']['full_vs_shuffle']['percentage']:+.2f}%)")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare normal vs shuffled reasoning (VLLM version)")
    parser.add_argument("--results_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/results.json",
                       help="Path to existing results.json")
    parser.add_argument("--grades_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/grades.json",
                       help="Path to existing grades.json")
    parser.add_argument("--output_dir", type=str,
                       default="./data/shuffle_comparison",
                       help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to test (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--del_last_line", type=float, default=0,
                       help="Lines to remove: integer >=1 for line count, float 0-1 for ratio (default: 0)")
    parser.add_argument("--judge_model_type", type=str, default='gpt-oss',
                       help="Model type for judging (default: gpt-oss)")
    parser.add_argument("--max_workers", type=int, default=50,
                       help="Maximum number of concurrent workers (default: 50)")
    parser.add_argument("--shuffle_type", type=str, default='line',
                       choices=['line', 'word', 'token'],
                       help="Shuffle method: 'line' (by-line), 'word' (by-word), or 'token' (by-token) (default: line)")
    parser.add_argument("--tokenizer_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                       help="Model name for tokenizer (only used for token shuffle, default: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)")
    parser.add_argument("--empty_question", action="store_true",
                       help="Replace question with empty string in prompts (default: False)")

    args = parser.parse_args()

    # Run experiment
    results = run_experiment(
        results_path=args.results_path,
        grades_path=args.grades_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        del_last_line=args.del_last_line,
        judge_model_type=args.judge_model_type,
        max_workers=args.max_workers,
        shuffle_type=args.shuffle_type,
        tokenizer_model=args.tokenizer_model,
        empty_question=args.empty_question
    )

    print(f"\n✓ Experiment complete!")
    print(f"✓ Results saved to {args.output_dir}/experiment_results.json")
    print(f"✓ Total problems processed: {len(results)}")


if __name__ == "__main__":
    main()
