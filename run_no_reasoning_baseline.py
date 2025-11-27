"""
No Reasoning Baseline Experiment

This script evaluates model performance when no reasoning is provided,
establishing a baseline for comparison with reasoning-based approaches.

The model is prompted to directly generate answers without any prefilled reasoning.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
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
class NoReasoningResult:
    """Result from a single question in no-reasoning baseline"""
    question_id: int
    question: str
    ground_truth: str
    generated_answer: str
    generation_time: float
    is_correct: bool = None
    grading_reasoning: str = None


def load_existing_results(results_path: str) -> List[Dict]:
    """Load existing results.json"""
    with open(results_path, 'r') as f:
        return json.load(f)


def build_gpt_oss_prompt_no_reasoning(question: str, reasoning_effort: str = "high") -> str:
    """
    Build GPT-OSS prompt with empty reasoning for text completion

    Format:
    <|start|>system<|message|>{system_message}<|end|>
    <|start|>user<|message|>{question}<|end|>
    <|start|>assistant<|channel|>analysis<|message|><|end|>
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

    # Build complete prompt with empty analysis channel
    prompt = f"<|start|>system<|message|>{system_message}<|end|>"
    prompt += f"<|start|>user<|message|>{question}<|end|>"
    prompt += f"<|start|>assistant<|channel|>analysis<|message|><|end|>"
    prompt += f"<|start|>assistant<|channel|>final<|message|>"

    debug_print(f'\n[DEBUG] Built no-reasoning prompt:\n{prompt}\n')
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


def create_grading_tasks(client: LLMClient, results: List[NoReasoningResult],
                        judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create grading tasks for generated answers"""
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

    for result in results:
        tasks.append(Task(
            index=result.question_id,
            request=Request(
                queries=[GRADING_PROMPT.format(
                    problem=result.question,
                    ground_truth=result.ground_truth,
                    model_answer=result.generated_answer
                )],
                model_type=judge_model_type,
                system_prompt="You are a helpful mathematical grading assistant.",
                reasoning_on=False,
                temperature=0.01
            ),
            metadata={'result': result}
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


def run_no_reasoning_experiment(results_path: str, output_path: str,
                                num_samples: int = None,
                                judge_model_type: str = 'gpt-oss',
                                max_workers: int = 50):
    """
    Run the no-reasoning baseline experiment

    Args:
        results_path: Path to existing results.json
        output_path: Path to save results
        num_samples: Number of samples to test (None = all)
        judge_model_type: Model to use for judging
        max_workers: Maximum number of concurrent workers
    """
    # Load data
    print("Loading data...")
    existing_results = load_existing_results(results_path)

    if num_samples:
        existing_results = existing_results[:num_samples]

    print(f"\nRunning no-reasoning baseline on {len(existing_results)} problems...")

    # Create LLM client
    client = LLMClient()

    # Phase 1: Create generation tasks
    print("\n=== Phase 1: Preparing generation tasks ===")
    generation_tasks = []

    for idx, item in enumerate(existing_results):
        question = item['question']
        ground_truth = item['answer']

        # Build prompt with no reasoning
        prompt = build_gpt_oss_prompt_no_reasoning(question)

        generation_tasks.append(Task(
            index=idx,
            request=CompletionRequest(
                prompt=prompt,
                model_type='gpt-oss',
                temperature=0.6,
                max_tokens=20480
            ),
            metadata={
                'question_id': idx,
                'question': question,
                'ground_truth': ground_truth
            }
        ))

    # Phase 2: Generate answers
    print(f"\n=== Phase 2: Generating answers ({len(generation_tasks)} tasks) ===")
    results = []

    for task in tqdm(client.complete_concurrent(generation_tasks, max_workers=max_workers),
                     total=len(generation_tasks), desc="Generating"):
        if not task.response.success:
            print(f"\nError in generation task {task.index}: {task.response.err_message}")
            continue

        # Parse answer
        answer = parse_answer_from_completion(task.response.content)

        # Create result object
        result = NoReasoningResult(
            question_id=task.metadata['question_id'],
            question=task.metadata['question'],
            ground_truth=task.metadata['ground_truth'],
            generated_answer=answer,
            generation_time=task.response.elapsed_seconds
        )

        results.append(result)

    # Phase 3: Grade answers
    print(f"\n=== Phase 3: Grading answers ({len(results)} tasks) ===")
    grading_tasks = create_grading_tasks(client, results, judge_model_type)

    for task in tqdm(client.generate_concurrent(grading_tasks, max_workers=max_workers),
                     total=len(grading_tasks), desc="Grading"):
        if not task.response.success:
            print(f"\nError in grading task {task.index}: {task.response.err_message}")
            continue

        result = task.metadata['result']
        is_correct = parse_yes_no_response(task.response.content)

        result.is_correct = is_correct
        result.grading_reasoning = task.response.content

    # Phase 4: Calculate summary and save results
    print("\n=== Phase 4: Saving results ===")

    total = len(results)
    correct_count = sum(1 for r in results if r.is_correct)
    avg_time = sum(r.generation_time for r in results) / total if total > 0 else 0

    summary = {
        'total_problems': total,
        'correct_count': correct_count,
        'accuracy': round(correct_count / total * 100, 2) if total > 0 else 0,
        'avg_generation_time': round(avg_time, 2)
    }

    output_data = {
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
    print("NO REASONING BASELINE - SUMMARY")
    print(f"{'='*80}")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct_count}/{total} ({summary['accuracy']:.2f}%)")
    print(f"Average generation time: {summary['avg_generation_time']:.2f}s")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="No Reasoning Baseline Experiment")
    parser.add_argument("--results_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/results.json",
                       help="Path to existing results.json")
    parser.add_argument("--output_path", type=str,
                       default="/mnt/shared/p01/yc/reasoning-behavior/data/AIME2025__R10/gpt-oss/p1/no_reasoning.json",
                       help="Output path for results")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to test (default: all)")
    parser.add_argument("--judge_model_type", type=str, default='gpt-oss',
                       help="Model type for judging (default: gpt-oss)")
    parser.add_argument("--max_workers", type=int, default=50,
                       help="Maximum number of concurrent workers (default: 50)")

    args = parser.parse_args()

    # Run experiment
    results = run_no_reasoning_experiment(
        results_path=args.results_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        judge_model_type=args.judge_model_type,
        max_workers=args.max_workers
    )

    print(f"\n✓ No reasoning baseline experiment complete!")
    print(f"✓ Results saved to {args.output_path}")
    print(f"✓ Total problems processed: {len(results)}")


if __name__ == "__main__":
    main()
