#!/usr/bin/env python3
"""
Mask Numbers Experiment on result.traj

This script:
1. Loads results.json with result.traj as reasoning
2. Masks numbers in reasoning (five modes available):
   - all: Mask all numbers (0-9) with '█' (default)
   - answer: Mask only the answer number
   - line: Mask all numbers in lines containing the answer
   - n-lines: Mask all numbers in answer line and N previous non-empty lines
              (N is configurable via --num-prev-lines, default: 1)
   - all-advance: Mask computational numbers while preserving algebraic notation
                  (keeps numbers adjacent to letters/underscores like A12, x_1, 3x)
3. Optionally shuffles lines after masking (--shuffle)
4. Generates new answers with masked (and optionally shuffled) reasoning
5. Grades answers and calculates accuracy

Purpose: Test whether the model relies on specific numbers and/or
         reasoning order in the reasoning process
"""

import json
import re
import random
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


def mask_answer_only_in_reasoning(reasoning: str, answer: str, mask_char: str = '█') -> str:
    """
    Mask only the answer number in reasoning with specified mask character

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer to mask
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with only answer occurrences replaced by mask_char
    """
    # Clean answer string (remove potential whitespace)
    answer_clean = answer.strip()

    # Escape special regex characters in answer
    answer_escaped = re.escape(answer_clean)

    # Replace all occurrences of the answer with masked version
    # Use word boundaries to avoid partial matches
    masked_answer = mask_char * len(answer_clean)
    masked_reasoning = re.sub(r'\b' + answer_escaped + r'\b', masked_answer, reasoning)

    return masked_reasoning


def mask_numbers_in_lines_with_answer(reasoning: str, answer: str, mask_char: str = '█') -> str:
    """
    Mask all numbers in lines that contain the answer

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer to identify relevant lines
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with all numbers masked in lines containing the answer
    """
    # Clean answer string (remove potential whitespace)
    answer_clean = answer.strip()

    # Escape special regex characters in answer for matching
    answer_escaped = re.escape(answer_clean)

    # Split reasoning into lines
    lines = reasoning.split('\n')
    masked_lines = []

    for line in lines:
        # Check if this line contains the answer (with word boundaries)
        if re.search(r'\b' + answer_escaped + r'\b', line):
            # Mask all digits in this line
            masked_line = re.sub(r'\d', mask_char, line)
            masked_lines.append(masked_line)
        else:
            # Keep line as is
            masked_lines.append(line)

    return '\n'.join(masked_lines)


def mask_numbers_in_nlines_with_answer(reasoning: str, answer: str, n: int = 1, mask_char: str = '█') -> str:
    """
    Mask all numbers in the line containing answer and the N non-empty lines before it

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer to identify relevant lines
        n: Number of previous non-empty lines to mask (default: 1)
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with all numbers masked in the answer line and previous N non-empty lines
    """
    # Clean answer string (remove potential whitespace)
    answer_clean = answer.strip()

    # Escape special regex characters in answer for matching
    answer_escaped = re.escape(answer_clean)

    # Split reasoning into lines
    lines = reasoning.split('\n')

    # Build mapping of non-empty lines: valid_index -> original_index
    non_empty_indices = []
    for i, line in enumerate(lines):
        if line.strip():  # Only count non-empty lines
            non_empty_indices.append(i)

    # Find lines that contain the answer (in non-empty lines)
    answer_line_positions = []  # positions in non_empty_indices
    for pos, orig_idx in enumerate(non_empty_indices):
        line = lines[orig_idx]
        if re.search(r'\b' + answer_escaped + r'\b', line):
            answer_line_positions.append(pos)

    # Build set of original line indices to mask
    lines_to_mask = set()
    for pos in answer_line_positions:
        # Mask the answer line
        orig_idx = non_empty_indices[pos]
        lines_to_mask.add(orig_idx)

        # Mask the previous N non-empty lines (if exist)
        for i in range(1, n + 1):
            if pos >= i:  # If there's an i-th previous non-empty line
                prev_orig_idx = non_empty_indices[pos - i]
                lines_to_mask.add(prev_orig_idx)

    # Apply masking
    masked_lines = []
    for i, line in enumerate(lines):
        if i in lines_to_mask:
            # Mask all digits in this line
            masked_line = re.sub(r'\d', mask_char, line)
            masked_lines.append(masked_line)
        else:
            # Keep line as is
            masked_lines.append(line)

    return '\n'.join(masked_lines)


def mask_numbers_all_advance(reasoning: str, answer: str = None, mask_char: str = '█') -> str:
    """
    Mask numbers with advanced rules: keep numbers adjacent to letters/underscores

    This mode masks computational numbers while preserving algebraic notation.

    Rules (in priority order):
    1. HARD RULE: If number equals answer → ALWAYS mask (highest priority)
    2. Number with [A-Za-z_] immediately before or after → Don't mask (algebraic)
    3. Number with inequality symbols (< > ≤ ≥ etc.) nearby (with optional spaces) → Don't mask
    4. Exception: "digit + x + digit" pattern → Force mask (multiplication like 3x3)
    5. Other numbers → Mask (computational values)

    Examples:
        A12 → A12 (not masked, variable index)
        x_1 → x_1 (not masked, subscript)
        3x → 3x (not masked, coefficient)
        1st → 1st (not masked, ordinal)
        n < 5 → n < 5 (not masked, inequality)
        1 ≤ x ≤ 10 → 1 ≤ x ≤ 10 (not masked, inequality)
        x^2 → x^█ (masked, exponent)
        3x3 → █x█ (masked, multiplication)
        1+2 → █+█ (masked, calculation)
        f(3) → f(█) (masked, function argument)
        answer=42, text="x42" → x██ (ALWAYS mask answer, even if adjacent to letter)

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer (if provided, will always be masked)
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with computational numbers masked
    """
    # Exception: Handle "digit+x+digit" multiplication pattern first
    # This must be done before the main rule to catch patterns like 3x3, 10x5
    reasoning = re.sub(
        r'\b(\d+)x(\d+)\b',
        lambda m: mask_char * len(m.group(1)) + 'x' + mask_char * len(m.group(2)),
        reasoning
    )

    # Main rule: Check each number sequence
    def should_mask_number(match):
        pos = match.start()
        text = match.string
        number = match.group()

        # HARD RULE: If number equals answer, ALWAYS mask (highest priority)
        if answer is not None and number == answer.strip():
            return mask_char * len(number)

        # Check character immediately before the number
        char_before = text[pos - 1] if pos > 0 else ''
        is_letter_before = char_before.isalpha() or char_before == '_'

        # Check character immediately after the number
        char_after = text[pos + len(number)] if pos + len(number) < len(text) else ''
        is_letter_after = char_after.isalpha() or char_after == '_'

        # Don't mask if adjacent to letter or underscore
        if is_letter_before or is_letter_after:
            return number

        # Check for inequality symbols near the number (with optional spaces)
        # Look for: <, >, ≤, ≥, \leq, \geq, \le, \ge, <=, >=
        # Search in a window around the number
        window_start = max(0, pos - 10)
        window_end = min(len(text), pos + len(number) + 10)
        window = text[window_start:window_end]

        # Inequality patterns (including LaTeX commands)
        inequality_patterns = [
            r'<', r'>', r'≤', r'≥', r'≦', r'≧',
            r'<=', r'>=',
            r'\\leq', r'\\geq', r'\\le', r'\\ge',
            r'\\lt', r'\\gt'
        ]

        has_inequality = any(re.search(pattern, window) for pattern in inequality_patterns)

        if has_inequality:
            return number
        else:
            return mask_char * len(number)

    # Apply main masking rule
    masked_reasoning = re.sub(r'\d+', should_mask_number, reasoning)

    return masked_reasoning


def shuffle_lines(reasoning: str, seed: int = None) -> str:
    """
    Shuffle reasoning content line-by-line

    Args:
        reasoning: Original reasoning content
        seed: Random seed for reproducibility

    Returns:
        Shuffled reasoning content
    """
    if seed is not None:
        random.seed(seed)

    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Shuffle
    random.shuffle(lines)

    return '\n'.join(lines)


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
    mask_char: str = '█',
    mask_mode: str = 'all',
    num_prev_lines: int = 1,
    shuffle: bool = False,
    seed_base: int = 42
) -> Task:
    """
    Prepare a Task for masked reasoning generation

    Args:
        item: Result item from results.json
        model_type: Model type
        mask_char: Character to use for masking numbers (default: '█')
        mask_mode: Masking mode - 'all', 'answer', 'line', 'n-lines', or 'all-advance'
                  'all': mask all numbers
                  'answer': mask only answer occurrences
                  'line': mask all numbers in lines containing answer
                  'n-lines': mask all numbers in answer line and N previous non-empty lines
                  'all-advance': mask computational numbers, keep algebraic notation
        num_prev_lines: Number of previous non-empty lines to mask (used with 'n-lines' mode)
        shuffle: If True, shuffle lines after masking
        seed_base: Base seed for shuffling
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

    # Mask numbers in reasoning based on mode
    if mask_mode == 'answer':
        masked_reasoning = mask_answer_only_in_reasoning(original_reasoning, ground_truth, mask_char)
    elif mask_mode == 'line':
        masked_reasoning = mask_numbers_in_lines_with_answer(original_reasoning, ground_truth, mask_char)
    elif mask_mode == 'n-lines':
        masked_reasoning = mask_numbers_in_nlines_with_answer(original_reasoning, ground_truth, num_prev_lines, mask_char)
    elif mask_mode == 'all-advance':
        masked_reasoning = mask_numbers_all_advance(original_reasoning, answer=ground_truth, mask_char=mask_char)
    else:  # 'all'
        masked_reasoning = mask_numbers_in_reasoning(original_reasoning, mask_char)

    # Apply line shuffle if requested
    if shuffle:
        masked_reasoning = shuffle_lines(masked_reasoning, seed=seed_base + index)

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
    mask_mode: str = 'all',
    num_prev_lines: int = 1,
    shuffle: bool = False,
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
        mask_mode: Masking mode - 'all', 'answer', 'line', or 'n-lines'
        num_prev_lines: Number of previous non-empty lines to mask (used with 'n-lines' mode)
        shuffle: If True, shuffle lines after masking
        limit: Limit number of questions (for testing)
    """
    print(f"Loading results from {results_path}")
    data = load_results_json(results_path)

    if limit:
        data = data[:limit]
        print(f"Limited to {limit} questions")

    mode_descriptions = {
        'all': 'Mask all numbers',
        'answer': 'Mask only answer',
        'line': 'Mask all numbers in lines containing answer',
        'n-lines': f'Mask all numbers in answer line and {num_prev_lines} previous non-empty line(s)',
        'all-advance': 'Mask computational numbers, keep algebraic notation (numbers adjacent to letters/underscores)'
    }

    print(f"Total questions: {len(data)}")
    print(f"Mask character: '{mask_char}'")
    print(f"Mask mode: {mode_descriptions.get(mask_mode, mask_mode)}")
    if mask_mode == 'n-lines':
        print(f"Previous lines: {num_prev_lines}")
    print(f"Shuffle lines: {'Yes' if shuffle else 'No'}")

    # Initialize LLM client
    client = LLMClient(mode=mode)

    # Prepare tasks
    print("Preparing mask tasks...")
    tasks = []
    for item in data:
        task = prepare_mask_task(item, model_type, mask_char, mask_mode, num_prev_lines, shuffle)
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

    mode_titles = {
        'all': 'All Numbers Masked',
        'answer': 'Answer Only Masked',
        'line': 'Lines with Answer Masked',
        'n-lines': f'N-Lines Masked (Answer + Previous {num_prev_lines} Line(s))',
        'all-advance': 'Advanced Masking (Computational Numbers Only)'
    }

    title = mode_titles.get(mask_mode, 'Masked')
    if shuffle:
        title += " + Shuffled"

    print("\n" + "="*60)
    print(f"EXPERIMENT RESULTS - {title}")
    print("="*60)
    print(f"Mask Mode:           {mode_descriptions.get(mask_mode, mask_mode)}")
    if mask_mode == 'n-lines':
        print(f"Previous Lines:      {num_prev_lines}")
    print(f"Shuffle Lines:       {'Yes' if shuffle else 'No'}")
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
        '--mask-mode',
        type=str,
        default='all',
        choices=['all', 'answer', 'line', 'n-lines', 'all-advance'],
        help='Masking mode: "all" (mask all numbers), "answer" (mask only answer), '
             '"line" (mask all numbers in lines containing answer), '
             '"n-lines" (mask all numbers in answer line and N previous non-empty lines), '
             '"all-advance" (mask computational numbers, keep algebraic notation)'
    )
    parser.add_argument(
        '--num-prev-lines',
        type=int,
        default=1,
        help='Number of previous non-empty lines to mask (used with --mask-mode n-lines, default: 1)'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='If set, shuffle lines after masking'
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
        mask_mode=args.mask_mode,
        num_prev_lines=args.num_prev_lines,
        shuffle=args.shuffle,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
