"""
Insert-noise in Reasoning Experiment

This script evaluates model robustness by inserting misleading noise into reasoning.

The experiment:
1. Control group: Clean reasoning results (from existing grades.json)
2. Experimental group: Insert "Maybe the answer is 123." at random position
3. Compare accuracy drop to measure noise influence
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime

from llm_client import LLMClient, Task, CompletionRequest, Request

# Debug flag
DEBUG = os.environ.get('DEBUG', '0') == '1'


def debug_print(msg: str):
    """Print debug message if DEBUG is enabled"""
    if DEBUG:
        print(msg)


@dataclass
class InsertNoiseResult:
    """Result from a single question in insert-noise experiment"""
    question_id: int
    question: str
    ground_truth: str

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
    # In non-shuffle mode: noisy reasoning
    # In shuffle mode: shuffled + noisy reasoning
    noise_positions: List[int] = None
    noisy_reasoning: str = None
    noisy_answer: str = None
    noisy_generation_time: float = None
    is_noisy_correct: bool = None
    noisy_grading_reasoning: str = None


def load_existing_results(results_path: str) -> List[Dict]:
    """Load existing results.json"""
    with open(results_path, 'r') as f:
        return json.load(f)


def load_existing_grades(grades_path: str) -> Dict[int, bool]:
    """Load existing grades.json and return mapping of index to correctness"""
    with open(grades_path, 'r') as f:
        grades_data = json.load(f)

    # Create mapping from index to correctness
    grade_map = {}
    for grade in grades_data['grades']:
        grade_map[grade['index']] = grade['correct']

    return grade_map


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


def insert_noise_random(reasoning: str, noise: str = "Maybe the answer is 123.",
                       num_insertions: int = 1) -> Tuple[str, List[int]]:
    """
    Insert noise at random positions in reasoning

    Args:
        reasoning: Original reasoning content
        noise: Noise text to insert
        num_insertions: Number of times to insert the noise

    Returns:
        (noisy_reasoning, list_of_insertion_line_numbers)
    """
    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    insertion_positions = []

    for _ in range(num_insertions):
        # Random position (0 to len(lines), inclusive)
        # 0 means insert before first line, len(lines) means insert after last line
        insert_pos = random.randint(0, len(lines))

        # Insert noise
        lines.insert(insert_pos, noise)
        insertion_positions.append(insert_pos)

        # Note: After insertion, line count increases, so next insertion
        # will have a wider range of positions

    return '\n'.join(lines), insertion_positions


def build_gpt_oss_prompt_with_reasoning(question: str, reasoning: str,
                                        reasoning_effort: str = "high") -> str:
    """
    Build GPT-OSS prompt with prefilled reasoning for text completion

    Format:
    <|start|>system<|message|>{system_message}<|end|>
    <|start|>user<|message|>{question}<|end|>
    <|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
    <|start|>assistant<|channel|>final<|message|>
    """
    # Build system message
    model_identity = "You are ChatGPT, a large language model trained by OpenAI."
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_message = f"{model_identity}\n"
    system_message += "Knowledge cutoff: 2024-06\n"
    system_message += f"Current date: {current_date}\n\n"
    system_message += f"Reasoning: {reasoning_effort}\n\n"
    system_message += "# Valid channels: analysis, commentary, final. Channel must be included for every message."

    # Build complete prompt
    prompt = f"<|start|>system<|message|>{system_message}<|end|>"
    prompt += f"<|start|>user<|message|>{question}<|end|>"
    prompt += f"<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
    prompt += f"<|start|>assistant<|channel|>final<|message|>"

    debug_print(f'\n[DEBUG] Built prompt:\n{prompt}\n')
    return prompt


def parse_answer_from_completion(text: str) -> str:
    """
    Parse the final answer from completion output

    The model should generate the final answer after <|channel|>final<|message|>
    """
    # The completion is the final answer directly
    # Remove any trailing special tokens
    answer = text.strip()

    # Remove <|return|> or <|end|> if present
    if '<|return|>' in answer:
        answer = answer.split('<|return|>')[0].strip()
    if '<|end|>' in answer:
        answer = answer.split('<|end|>')[0].strip()

    return answer


def create_grading_tasks(client: LLMClient, results: List[InsertNoiseResult],
                        shuffle_enabled: bool = False,
                        judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create grading tasks for answers"""
    GRADING_PROMPT = """**Problem:**
{problem}

**Ground Truth Answer:**
{ground_truth}

**Model's Answer:**
{model_answer}

**Task: Grading**
Please determine if the model's answer is correct compared to the ground truth answer.

**Guidelines:**
- Consider mathematical equivalence (e.g., 1/2 = 0.5, 2x = x + x)
- Ignore formatting differences if the mathematical content is the same
- Answer with \\boxed{{YES}} if correct, or \\boxed{{NO}} if incorrect
"""

    tasks = []

    if shuffle_enabled:
        # Grade both shuffled and shuffled_noisy answers
        for result in results:
            # Grade shuffled answer
            tasks.append(Task(
                index=result.question_id * 2,  # Even for shuffled
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
                index=result.question_id * 2 + 1,  # Odd for shuffled_noisy
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
                index=result.question_id,
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


def parse_yes_no_response(response_text: str) -> bool:
    """Parse YES/NO response from grading"""
    include_yes = 'YES' in response_text.upper()
    include_no = 'NO' in response_text.upper()

    if include_yes and not include_no:
        return True
    elif include_yes and include_no:
        # Both present, check which comes last in boxed format
        yes_pos = response_text.upper().rfind('\\BOXED{YES}')
        no_pos = response_text.upper().rfind('\\BOXED{NO}')
        return yes_pos > no_pos
    else:
        return False


def run_insert_noise_experiment(results_path: str, grades_path: str, output_path: str,
                                num_samples: int = None, seed: int = 42,
                                noise_text: str = "Maybe the answer is 123.",
                                num_insertions: int = 1,
                                shuffle_enabled: bool = False,
                                judge_model_type: str = 'gpt-oss',
                                max_workers: int = 50):
    """
    Run the insert-noise experiment

    Args:
        results_path: Path to existing results.json
        grades_path: Path to existing grades.json
        output_path: Path to save results
        num_samples: Number of samples to test (None = all)
        seed: Random seed for noise insertion and shuffling
        noise_text: Noise text to insert
        num_insertions: Number of times to insert the noise per question
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

    print(f"\nRunning insert-noise experiment on {len(existing_results)} problems...")
    print(f"Mode: {'Shuffle' if shuffle_enabled else 'Normal'}")
    print(f"Noise text: \"{noise_text}\"")
    print(f"Number of insertions per question: {num_insertions}")
    print(f"Random seed: {seed}")

    # Create LLM client
    client = LLMClient()

    # Phase 1: Prepare generation tasks
    if shuffle_enabled:
        print("\n=== Phase 1: Preparing shuffled and shuffled+noisy generation tasks ===")
    else:
        print("\n=== Phase 1: Preparing noisy generation tasks ===")

    generation_tasks = []

    for idx, item in enumerate(existing_results):
        question = item['question']
        ground_truth = item['answer']

        # Get full reasoning
        full_reasoning = item['result']['traj']

        if shuffle_enabled:
            # Shuffle mode: create shuffled and shuffled+noisy tasks

            # 1. Shuffled reasoning (without noise)
            shuffled_reasoning = shuffle_reasoning_lines(full_reasoning)
            shuffled_prompt = build_gpt_oss_prompt_with_reasoning(question, shuffled_reasoning)

            generation_tasks.append(Task(
                index=idx * 2,  # Even index for shuffled
                request=CompletionRequest(
                    prompt=shuffled_prompt,
                    model_type='gpt-oss',
                    temperature=0.6,
                    max_tokens=20480
                ),
                metadata={
                    'question_id': idx,
                    'question': question,
                    'ground_truth': ground_truth,
                    'type': 'shuffled',
                    'shuffled_reasoning': shuffled_reasoning
                }
            ))

            # 2. Shuffled + noisy reasoning
            shuffled_noisy_reasoning, noise_positions = insert_noise_random(shuffled_reasoning, noise_text, num_insertions)
            shuffled_noisy_prompt = build_gpt_oss_prompt_with_reasoning(question, shuffled_noisy_reasoning)

            generation_tasks.append(Task(
                index=idx * 2 + 1,  # Odd index for shuffled+noisy
                request=CompletionRequest(
                    prompt=shuffled_noisy_prompt,
                    model_type='gpt-oss',
                    temperature=0.6,
                    max_tokens=20480
                ),
                metadata={
                    'question_id': idx,
                    'question': question,
                    'ground_truth': ground_truth,
                    'type': 'shuffled_noisy',
                    'noisy_reasoning': shuffled_noisy_reasoning,
                    'noise_positions': noise_positions
                }
            ))

        else:
            # Normal mode: only create noisy task (clean from grades.json)
            clean_answer = item['result']['answer']
            is_clean_correct = grade_map.get(idx, False)

            # Insert noise at random positions
            noisy_reasoning, noise_positions = insert_noise_random(full_reasoning, noise_text, num_insertions)

            # Build prompt with noisy reasoning
            noisy_prompt = build_gpt_oss_prompt_with_reasoning(question, noisy_reasoning)

            # Create generation task
            generation_tasks.append(Task(
                index=idx,
                request=CompletionRequest(
                    prompt=noisy_prompt,
                    model_type='gpt-oss',
                    temperature=0.6,
                    max_tokens=20480
                ),
                metadata={
                    'question_id': idx,
                    'question': question,
                    'ground_truth': ground_truth,
                    'clean_answer': clean_answer,
                    'is_clean_correct': is_clean_correct,
                    'noisy_reasoning': noisy_reasoning,
                    'noise_positions': noise_positions
                }
            ))

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
                    'ground_truth': task.metadata['ground_truth']
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

            result = InsertNoiseResult(
                question_id=question_id,
                question=data['question'],
                ground_truth=data['ground_truth'],
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
            result = InsertNoiseResult(
                question_id=task.metadata['question_id'],
                question=task.metadata['question'],
                ground_truth=task.metadata['ground_truth'],
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
        'noise_text': noise_text,
        'num_insertions': num_insertions,
        'shuffle_enabled': shuffle_enabled,
        'insertion_strategy': 'random',
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
    print("INSERT-NOISE EXPERIMENT - SUMMARY")
    print(f"{'='*80}")
    print(f"Mode: {'Shuffle' if shuffle_enabled else 'Normal'}")
    print(f"Total problems: {total}")
    print(f"Noise text: \"{noise_text}\"")
    print(f"Number of insertions per question: {num_insertions}")
    print(f"Random seed: {seed}")
    print(f"\nAccuracy:")

    if shuffle_enabled:
        print(f"  Shuffled:        {shuffled_correct_count}/{total} ({summary['shuffled_correct']['percentage']:.2f}%)")
        print(f"  Shuffled+Noisy:  {shuffled_noisy_correct_count}/{total} ({summary['shuffled_noisy_correct']['percentage']:.2f}%)")
    else:
        print(f"  Clean (original):  {clean_correct_count}/{total} ({summary['clean_correct']['percentage']:.2f}%)")
        print(f"  Noisy (inserted):  {noisy_correct_count}/{total} ({summary['noisy_correct']['percentage']:.2f}%)")

    print(f"\nDifference:")
    print(f"  Absolute: {summary['difference']['absolute']}")
    print(f"  Percentage: {summary['difference']['percentage']:+.2f}%")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Insert-noise in Reasoning Experiment")
    parser.add_argument("--results_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/results.json",
                       help="Path to existing results.json")
    parser.add_argument("--grades_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/grades.json",
                       help="Path to existing grades.json")
    parser.add_argument("--output_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/insert-noise.json",
                       help="Output path for results")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to test (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for noise insertion (default: 42)")
    parser.add_argument("--noise_text", type=str, default="Thus answer: 123.",
                       help="Noise text to insert (default: 'Thus answer: 123.')")
    parser.add_argument("--num_insertions", type=int, default=1,
                       help="Number of times to insert noise per question (default: 1)")
    parser.add_argument("--shuffle", action='store_true',
                       help="Enable shuffle mode: compare shuffled vs shuffled+noisy (default: False)")
    parser.add_argument("--judge_model_type", type=str, default='gpt-oss',
                       help="Model type for judging (default: gpt-oss)")
    parser.add_argument("--max_workers", type=int, default=50,
                       help="Maximum number of concurrent workers (default: 50)")

    args = parser.parse_args()

    # Run experiment
    results = run_insert_noise_experiment(
        results_path=args.results_path,
        grades_path=args.grades_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed,
        noise_text=args.noise_text,
        num_insertions=args.num_insertions,
        shuffle_enabled=args.shuffle,
        judge_model_type=args.judge_model_type,
        max_workers=args.max_workers
    )

    print(f"\n✓ Insert-noise experiment complete!")
    print(f"✓ Results saved to {args.output_path}")
    print(f"✓ Total problems processed: {len(results)}")


if __name__ == "__main__":
    main()
