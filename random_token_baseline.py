#!/usr/bin/env python3
"""
Random Token Baseline Experiment

This script tests the effect of replacing reasoning with random tokens:
1. Load normal_reasoning from experiment_results.json
2. Tokenize each reasoning and count tokens
3. Generate same number of random tokens
4. Use random tokens as "reasoning" and get model's answer
5. Evaluate accuracy with random token reasoning

實驗目的：測試如果 reasoning 內容是隨機的，模型回答問題的正確率
"""

import json
import os
import random
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
from tqdm import tqdm

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from llm_client import LLMClient, CompletionRequest, Task, Request
from comparison_shuffle_reasoning import (
    parse_answer_from_completion,
    parse_yes_no_response,
    build_gpt_oss_prompt_with_reasoning
)

# Global tokenizer cache
_TOKENIZER_CACHE = {}

CONCURRENCY = 10
MAX_TRY = 3


def load_tokenizer(model_name: str = "openai/gpt-oss-120b"):
    """Load and cache tokenizer"""
    if model_name not in _TOKENIZER_CACHE:
        print(f"Loading tokenizer for {model_name}...")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return _TOKENIZER_CACHE[model_name]


def is_english_token(token_text: str) -> bool:
    """
    Check if a token contains only English characters, numbers, spaces, and common punctuation

    Args:
        token_text: The decoded token text

    Returns:
        True if token is English-only, False otherwise
    """
    import re
    # Allow English letters, digits, spaces, and common punctuation
    # Pattern: only ASCII printable characters (space to ~)
    return bool(re.match(r'^[ -~]+$', token_text))


# Cache for English token IDs per tokenizer
_ENGLISH_TOKEN_CACHE = {}


def get_english_token_ids(tokenizer) -> List[int]:
    """
    Get list of token IDs that decode to English-only text

    Args:
        tokenizer: Tokenizer instance

    Returns:
        List of valid English token IDs
    """
    # Use cache to avoid recomputing
    tokenizer_name = str(tokenizer.name_or_path)
    if tokenizer_name in _ENGLISH_TOKEN_CACHE:
        return _ENGLISH_TOKEN_CACHE[tokenizer_name]

    vocab_size = len(tokenizer)
    special_token_ids = set(tokenizer.all_special_ids)

    english_token_ids = []

    print(f"Filtering English tokens from vocabulary (size: {vocab_size})...")
    for token_id in range(vocab_size):
        if token_id in special_token_ids:
            continue

        # Decode single token
        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            if token_text and is_english_token(token_text):
                english_token_ids.append(token_id)
        except:
            continue

    print(f"Found {len(english_token_ids)} English tokens out of {vocab_size}")

    # Cache the result
    _ENGLISH_TOKEN_CACHE[tokenizer_name] = english_token_ids

    return english_token_ids


def generate_random_tokens(num_tokens: int, tokenizer, seed: int = None) -> str:
    """
    Generate a string of random English-only tokens with the same length as the original reasoning

    Args:
        num_tokens: Number of tokens to generate
        tokenizer: Tokenizer instance
        seed: Random seed for reproducibility

    Returns:
        Decoded text from random English tokens
    """
    if seed is not None:
        random.seed(seed)

    # Get English-only token IDs
    english_token_ids = get_english_token_ids(tokenizer)

    if not english_token_ids:
        raise ValueError("No English tokens found in vocabulary")

    # Generate random token IDs from English-only tokens
    random_token_ids = random.choices(english_token_ids, k=num_tokens)

    # Decode back to text
    random_text = tokenizer.decode(random_token_ids, skip_special_tokens=True)

    return random_text


def tokenize_and_count(text: str, tokenizer) -> Tuple[List[int], int]:
    """
    Tokenize text and return tokens and count

    Returns:
        (token_ids, token_count)
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens, len(tokens)


# Reuse build_gpt_oss_prompt_with_reasoning from comparison_shuffle_reasoning
# Just alias it for clarity in this context
build_prompt_with_random_reasoning = build_gpt_oss_prompt_with_reasoning


def load_experiment_data(results_path: str) -> List[Dict]:
    """Load experiment results"""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_grading_tasks(results: List[Dict], judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create grading tasks for random token baseline answers"""
    GRADING_PROMPT = """## Problem:
{problem}

## Ground Truth Answer:
{ground_truth}

## Model's Answer:
{model_answer}

## Task: Grading
Please determine if the model's answer is correct compared to the ground truth answer.

**Guidelines:**
- Consider mathematical equivalence (e.g., 1/2 = 0.5, 2x = x + x)
- Ignore formatting differences if the mathematical content is the same
- Answer with \\boxed{{YES}} if correct, or \\boxed{{NO}} if incorrect
"""

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
            metadata={'result_index': result['question_id']}
        ))

    return tasks


def prepare_task(
    item: Dict,
    tokenizer,
    model_type: str,
    seed_base: int = 42
) -> Task:
    """
    Prepare a Task for a single item: tokenize reasoning, generate random tokens
    """
    question_id = item['question_id']
    question = item['question']
    ground_truth = item['ground_truth']
    normal_reasoning = item['normal_reasoning']

    # Tokenize normal reasoning
    tokens, token_count = tokenize_and_count(normal_reasoning, tokenizer)

    # Generate random tokens with same length
    random_reasoning = generate_random_tokens(
        token_count,
        tokenizer,
        seed=seed_base + question_id
    )

    # Build prompt
    prompt = build_prompt_with_random_reasoning(question, random_reasoning)

    # Create CompletionRequest
    request = CompletionRequest(
        prompt=prompt,
        model_type=model_type,
        temperature=0.0,
        max_tokens=2048
    )

    # Create Task with metadata
    task = Task(
        index=question_id,
        request=request,
        metadata={
            'question': question,
            'ground_truth': ground_truth,
            'normal_reasoning_token_count': token_count,
            'random_reasoning': random_reasoning,
        }
    )

    return task


def task_to_result(task: Task) -> Dict:
    """
    Convert a completed Task to result dict
    """
    metadata = task.metadata
    response = task.response

    if response.success:
        generated_answer = response.content

        result = {
            'question_id': task.index,
            'question': metadata['question'],
            'ground_truth': metadata['ground_truth'],
            'normal_reasoning_token_count': metadata['normal_reasoning_token_count'],
            'random_reasoning': metadata['random_reasoning'],
            'generated_answer': generated_answer,
            'is_correct': None,  # Will be filled by grading
            'grading_reasoning': None,  # Will be filled by grading
            'success': True,
            'error': None
        }
    else:
        result = {
            'question_id': task.index,
            'question': metadata['question'],
            'ground_truth': metadata['ground_truth'],
            'normal_reasoning_token_count': metadata['normal_reasoning_token_count'],
            'random_reasoning': metadata['random_reasoning'],
            'generated_answer': None,
            'is_correct': False,
            'grading_reasoning': None,
            'success': False,
            'error': response.err_message
        }

    return result


def run_experiment(
    results_path: str,
    output_path: str,
    model_type: str = 'gpt-oss',
    mode: str = 'openrouter',
    tokenizer_model: str = "openai/gpt-oss-120b",
    limit: int = None
):
    """
    Run the random token baseline experiment

    Args:
        results_path: Path to experiment_results.json
        output_path: Path to save results
        model_type: Model type ('gpt-oss', 'deepseek', etc.)
        mode: 'openrouter' or 'local'
        tokenizer_model: Model name for tokenizer
        limit: Limit number of questions to process (for testing)
    """
    print(f"Loading experiment data from {results_path}")
    data = load_experiment_data(results_path)

    if limit:
        data = data[:limit]
        print(f"Limited to {limit} questions")

    print(f"Total questions: {len(data)}")

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_model)

    # Initialize LLM client
    client = LLMClient(mode=mode)

    # Prepare all tasks
    print("Preparing tasks...")
    tasks = []
    for item in data:
        task = prepare_task(item, tokenizer, model_type)
        tasks.append(task)

    # Phase 1: Generate answers with random reasoning
    results = []
    print(f"\n=== Phase 1: Generating answers ({len(tasks)} tasks) ===")

    for completed_task in tqdm(client.complete_concurrent(tasks, max_workers=CONCURRENCY), total=len(tasks)):
        result = task_to_result(completed_task)
        results.append(result)

        # Save incrementally
        results_sorted = sorted(results, key=lambda x: x['question_id'])
        with open(output_path, 'w') as f:
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

        result_index = grading_task.metadata['result_index']
        # Find the corresponding result
        for result in results:
            if result['question_id'] == result_index:
                result['is_correct'] = parse_yes_no_response(grading_task.response.content)
                result['grading_reasoning'] = grading_task.response.content
                break

        # Save incrementally after grading
        results_sorted = sorted(results, key=lambda x: x['question_id'])
        with open(output_path, 'w') as f:
            json.dump(results_sorted, f, indent=2, ensure_ascii=False)

    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    correct_count = sum(1 for r in successful_results if r.get('is_correct', False))

    total_tokens = sum(r['normal_reasoning_token_count'] for r in successful_results)
    avg_tokens = total_tokens / len(successful_results) if successful_results else 0

    stats = {
        'total_questions': len(data),
        'successful': len(successful_results),
        'failed': len(data) - len(successful_results),
        'correct': correct_count,
        'accuracy': correct_count / len(successful_results) if successful_results else 0,
        'avg_token_count': avg_tokens,
        'total_tokens': total_tokens
    }

    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Total Questions:     {stats['total_questions']}")
    print(f"Successful:          {stats['successful']}")
    print(f"Failed:              {stats['failed']}")
    print(f"Correct Answers:     {stats['correct']}")
    print(f"Accuracy:            {stats['accuracy']:.2%}")
    print(f"Avg Token Count:     {stats['avg_token_count']:.1f}")
    print(f"Total Tokens:        {stats['total_tokens']:,}")
    print("="*60)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Random Token Baseline Experiment'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default='data/shuffle_comparison_exp/word_turncate_f00/experiment_results.json',
        help='Path to experiment_results.json'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/shuffle_comparison_exp/word_turncate_f00/random_token_baseline.json',
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
        '--tokenizer_model',
        type=str,
        default='openai/gpt-oss-120b',
        help='Model name for tokenizer'
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
        tokenizer_model=args.tokenizer_model,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
