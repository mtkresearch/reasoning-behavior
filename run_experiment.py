#!/usr/bin/env python3
"""
Reasoning Processing Experiment with Pipeline Architecture

This script processes reasoning text through a configurable pipeline of transformations
(masking, truncating, shuffling, inserting, removing) and evaluates model performance on the processed reasoning.

=============================================================================
USAGE
=============================================================================

Basic syntax:
    python run_experiment.py --flow "<pipeline_steps>"

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
   - 'all-nonblank': Mask all non-whitespace characters (letters, numbers, symbols)
                     Preserves spaces, tabs, newlines, and other whitespace
   - 'non-number': Mask all non-digit characters, preserving only digits (0-9)
                   Preserves whitespace characters (spaces, tabs, newlines)

   Optional parameters:
   - mask_char: Character to use for masking (default: '█')
   - num_prev_lines: For 'n-lines' mode (default: 1)

2. truncate(mode, ...)
   Remove content from reasoning.

   Modes:
   - 'answer': Remove exact answer string from reasoning (all occurrences, word boundaries)
   - 'answer_and_after': Remove answer line and all lines after it
   - 'before_answer': Remove all lines before answer line (answer line kept)
   - 'last_n_lines': Remove last N lines (specify n=N)
   - 'last_ratio': Remove last X% of lines (specify ratio=X, e.g., 0.3 for 30%)
   - 'after_line': Keep only first N lines or first X% of lines (specify n=N or r=X)
                   Use n for fixed number (e.g., n=10), r for ratio (e.g., r=0.1 for 10%)
                   If both provided, n takes priority
   - 'before_line': Remove first N lines or first X% of lines (specify n=N or r=X)
                    Use n for fixed number (e.g., n=10), r for ratio (e.g., r=0.1 for 10%)
                    If both provided, n takes priority

3. shuffle(mode, seed=None, ...)
   Shuffle reasoning content.

   Modes:
   - 'line': Shuffle lines (randomly reorder lines while preserving content)
   - 'word': Shuffle words across entire text (all words randomized globally)
   - 'in-line-word': Shuffle words within each line independently (preserves line boundaries, shuffles words within each line)
   - 'token': Shuffle tokens (tokenizer-based shuffling, specify tokenizer_model='gpt2' or other)

   Optional parameters:
   - seed: Random seed for reproducibility

4. insert(mode, sentence='...', count=1, seed=None)
   Insert text into reasoning chain.

   Modes:
   - 'fix': Insert fixed text at random positions
   - 'head': Insert text at the beginning (first line) of reasoning
             Supports {ANSWER} placeholder which will be replaced with ground truth answer
   - 'middle': Insert text at the middle of reasoning
               Supports {ANSWER} placeholder which will be replaced with ground truth answer
   - 'tail': Insert text at the end (last line) of reasoning
             Supports {ANSWER} placeholder which will be replaced with ground truth answer

   Parameters (for 'fix' mode):
   - sentence: Text to insert (default: 'Maybe the answer is 123.')
   - count: Number of times to insert the text
           Can be an integer (e.g., count=5) or a percentage string (e.g., count='100% # of answer')

           Integer format:
           - Fixed number of insertions (e.g., count=5 inserts 5 times)

           Percentage format: '<percentage>% # of answer'
           - Dynamic count based on answer occurrences in reasoning
           - '100% # of answer': Insert once for each answer occurrence
           - '200% # of answer': Insert twice for each answer occurrence
           - '50% # of answer': Insert 0.5 times for each answer occurrence (rounded)

   - seed: Random seed for reproducibility

   Parameters (for 'head' mode):
   - sentence: Text to insert at the beginning (supports {ANSWER} placeholder)

   Parameters (for 'middle' mode):
   - sentence: Text to insert at the middle (supports {ANSWER} placeholder)

   Parameters (for 'tail' mode):
   - sentence: Text to insert at the end (supports {ANSWER} placeholder)

5. remove(mode)
   Remove or consolidate whitespace characters in reasoning.

   Modes:
   - 'blank': Consolidate all continuous blank characters (spaces, tabs, newlines)
              into a single space. Strips leading/trailing whitespace.

   Examples:
   - Input:  "hello\n\n\n   world   \t\ntest"
   - Output: "hello world test"

6. replace(pattern, replacement='')
   Replace text using regular expressions.

   Parameters:
   - pattern: Regular expression pattern to match (e.g., '\\s', '\\d', '\\w+')
             OR special placeholder like '{ANSWER}' to match ground truth answer
   - replacement: String to replace matches with (default: empty string)

   Special Placeholders:
   - '{ANSWER}' or '{answer}': Automatically replaced with the ground truth answer
                                This allows you to replace the correct answer with a different value

   Common use cases:
   - Replace all whitespace with single space: pattern='\\s', replacement=' '
   - Replace all digits with 'X': pattern='\\d', replacement='X'
   - Remove specific pattern: pattern='\\d+', replacement=''
   - Replace ground truth answer with fixed value: pattern='{ANSWER}', replacement='123'

   Examples:
   - Input:  "hello\nworld" with pattern='\\s', replacement=' '
   - Output: "hello world"
   - Input:  "Calculate 2 + 2 = 4" with pattern='\\d', replacement='X'
   - Output: "Calculate X + X = X"
   - Input:  "The answer is 42" (ground truth: 42) with pattern='{ANSWER}', replacement='999'
   - Output: "The answer is 999"

7. answer(mode)
   Add prefill text to guide model answer generation.

   This processor does not modify the reasoning text itself. Instead, it adds
   prefill text at the beginning of the answer section in the prompt to guide
   the model's response format. The prefill text is auto-determined based on the
   dataset type (math vs code).

   Modes:
   - 'retrieval': Add prefill text to guide answer generation

   Auto-determined prefill text:
   - Math datasets: "Thus, the answer is"
   - Code datasets: "Thus, the code is\n```cpp\n"

   Examples:
   - answer('retrieval') - Auto-selects prefill based on dataset type

   Use cases:
   - Guide the model to start answers in a specific format
   - Ensure consistent answer structure across responses
   - Improve model performance by providing format hints

   Note:
   - For math datasets: max_tokens is automatically reduced from 5000 to 50 when using
     answer('retrieval') to optimize generation speed and cost, since only a short answer is expected
   - For code datasets: max_tokens remains at 5000 to allow full code generation

8. reason_is(text)
   Replace reasoning content with a custom text pattern.

   This processor replaces the entire reasoning text with a custom pattern
   that can include the ground truth answer via the {ANSWER} placeholder.
   The {ANSWER} placeholder supports mathematical expressions.
   Useful for testing if models can retrieve the answer when given minimal
   or reformatted reasoning chains.

   Pattern Placeholders:
   - '{ANSWER}' or '{answer}': Replaced with the ground truth answer from context
   - '{ANSWER * expr}': Mathematical expressions with ANSWER (e.g., {ANSWER * 0.9})
   - Type preservation: If ANSWER is integer, result is integer; if float, result is float

   Parameters:
   - text: Text pattern to replace reasoning with (supports {ANSWER} placeholder and math expressions)

   Examples:
   - reason_is('{ANSWER}') - Replace reasoning with pure answer (e.g., "42")
   - reason_is('Thus, the answer is {ANSWER}.') - Replace with illustrated format
   - reason_is('The final result is {ANSWER}') - Custom format with answer
   - reason_is('The answer is {ANSWER} or {ANSWER * 0.9}') - Math expressions with type preservation
   - reason_is('{ANSWER + 10}') - Addition operation
   - reason_is('{ANSWER / 2}') - Division operation

   Use cases:
   - Test if model can retrieve answer when reasoning is just the answer
   - Baseline comparison: answer-only vs. full reasoning
   - Measure impact of reasoning steps on model performance
   - Test different answer presentation formats
   - Test model performance with alternative answers derived from ground truth

9. random(mode, seed=42)
   Randomize digits in reasoning text.

   This processor randomly replaces all digits (0-9) with other random digits.
   Useful for testing if models rely on specific numeric values or just
   the structural pattern of reasoning.

   Modes:
   - 'number': Randomize all digits (0-9) to random digits

   Optional parameters:
   - seed: Random seed for reproducibility (default: 42)

   Examples:
   - random('number') - Randomize all digits with default seed
   - random('number', seed=123) - Randomize all digits with custom seed

   Use cases:
   - Test if model relies on specific numeric values vs. reasoning structure
   - Evaluate robustness to numeric variations
   - Test if reasoning patterns are more important than exact numbers

10. question(mode)
   Modify or remove question text from the prompt.

   This processor modifies the question field in the context rather than the reasoning text.
   Useful for testing if models rely on the question text or can answer from reasoning alone.

   Modes:
   - 'remove': Remove entire question (set to empty string)

   Use cases:
   - Test if model can answer based solely on reasoning without question context
   - Evaluate model robustness to missing problem statements

11. padding(mode, tokenizer_model='gpt2', words_tsv_path=None, seed=42)
   Replace reasoning text with random tokens or words.

   This processor replaces the original reasoning with random tokens or words sampled from a distribution.
   The count of replacement tokens/words equals the original reasoning's token/word count.

   Useful for testing if models rely on actual reasoning content or just the presence of
   reasoning-like text with the same length.

   Modes:
   - 'token': Replace reasoning with random tokens using a tokenizer (count = original token count)
   - 'word': Replace reasoning with random words sampled from word frequency distribution (count = original word count)

   Parameters:
   - tokenizer_model: Model name for tokenizer (for token mode, default: 'gpt2')
   - words_tsv_path: Path to words.tsv file (required for word mode)
   - seed: Random seed for reproducibility (default: 42)

   Note: The 'position' parameter was deprecated and is no longer supported

   Examples:
   - padding('token') - Replace reasoning with random tokens (count = original token count) using gpt2 tokenizer
   - padding('token', tokenizer_model='gpt2', seed=42) - Custom tokenizer and seed
   - padding('word', words_tsv_path='data/AIME2025__R10/gpt-oss/p1/words.tsv') - Replace reasoning with random words (count = original word count)
   - padding('word', words_tsv_path='data/AIME2025__R10/gpt-oss/p1/words.tsv', seed=42) - Custom seed for word selection

   Use cases:
   - Test if model relies on actual reasoning content vs. just having text of the same length
   - Evaluate baseline performance with random content replacing reasoning
   - Compare performance between real reasoning and same-length random noise
   - Test if reasoning structure matters more than content

-----------------------------------------------------------------------------
Examples
-----------------------------------------------------------------------------

# Example 1: Basic masking
python run_experiment.py --flow "mask('number')"

# Example 2: Mask and shuffle
python run_experiment.py --flow "mask('number'),shuffle('line')"

# Example 3: Full pipeline - truncate, mask, shuffle
python run_experiment.py --flow "truncate('answer_and_after'),mask('number'),shuffle('line')"

# Example 4: Custom mask character
python run_experiment.py --flow "mask('number',mask_char='*')"

# Example 5: Ratio-based truncation
python run_experiment.py --flow "truncate('last_ratio',ratio=0.3),mask('alphabet')"

# Example 6: Word-level shuffle with specific seed
python run_experiment.py --flow "mask('number'),shuffle('word',seed=42)"

# Example 6a: In-line word shuffle (shuffle words within each line independently)
python run_experiment.py --flow "shuffle('in-line-word')"

# Example 6b: Combine in-line word shuffle with masking
python run_experiment.py --flow "mask('number'),shuffle('in-line-word',seed=42)"

# Example 6c: Token-level shuffle with custom tokenizer
python run_experiment.py --flow "shuffle('token',tokenizer_model='gpt2',seed=42)"

# Example 7: N-lines masking
python run_experiment.py --flow "mask('n-lines',num_prev_lines=2)"

# Example 8: Complex combination
python run_experiment.py --flow "truncate('last_ratio',ratio=0.3),mask('number-advance'),shuffle('line',seed=123)"

# Example 9: Insert noise at random positions
python run_experiment.py --flow "insert('fix',sentence='Maybe the answer is 123.',count=5)"

# Example 10: Combine insert with other processors
python run_experiment.py --flow "insert('fix',sentence='Thus answer: 123.',count=3),shuffle('line')"

# Example 11: Insert with specific seed for reproducibility
python run_experiment.py --flow "insert('fix',sentence='Answer: 456.',count=5,seed=42)"

# Example 11a: Insert noise based on answer occurrence count (100%)
python run_experiment.py --flow "insert('fix',sentence='Thus answer: 123.',count='100% # of answer')"

# Example 11b: Insert noise 200% of answer occurrences (twice per answer)
python run_experiment.py --flow "insert('fix',sentence='Maybe answer: 999.',count='200% # of answer')"

# Example 11c: Insert noise 50% of answer occurrences (half, rounded)
python run_experiment.py --flow "insert('fix',sentence='Check: 000.',count='50% # of answer')"

# Example 11d: Insert text at the beginning (first line) with {ANSWER} placeholder
python run_experiment.py --flow "insert('head',sentence='The answer is {ANSWER}.')"

# Example 11e: Insert static text at the beginning without placeholder
python run_experiment.py --flow "insert('head',sentence='Important note: Read carefully.')"

# Example 11f: Insert text at the middle with {ANSWER} placeholder
python run_experiment.py --flow "insert('middle',sentence='The answer is {ANSWER}.')"

# Example 11g: Insert static text at the middle without placeholder
python run_experiment.py --flow "insert('middle',sentence='Let me verify this step.')"

# Example 11h: Insert text at the end (last line) with {ANSWER} placeholder
python run_experiment.py --flow "insert('tail',sentence='The answer is {ANSWER}.')"

# Example 11i: Insert static text at the end without placeholder
python run_experiment.py --flow "insert('tail',sentence='Remember to verify your answer.')"

# Example 12: Mask all non-blank characters (preserve only whitespace structure)
python run_experiment.py --flow "mask('all-nonblank')"

# Example 13: Combine all-nonblank masking with shuffle
python run_experiment.py --flow "mask('all-nonblank'),shuffle('line')"

# Example 13a: Mask all non-digit characters (preserve only numbers)
python run_experiment.py --flow "mask('non-number')"

# Example 13b: Mask all non-digit characters with custom mask character
python run_experiment.py --flow "mask('non-number',mask_char=' ')"

# Example 14: Remove all continuous blank characters (consolidate to single space)
python run_experiment.py --flow "remove('blank')"

# Example 15: Combine remove with other processors
python run_experiment.py --flow "mask('number'),remove('blank')"

# Example 16: Multi-step processing with blank removal
python run_experiment.py --flow "truncate('last_ratio',ratio=0.2),mask('alphabet'),remove('blank')"

# Example 17: Replace all whitespace with single space
python run_experiment.py --flow "replace('\\s',replacement=' ')"

# Example 18: Replace all digits with 'X'
python run_experiment.py --flow "replace('\\d',replacement='X')"

# Example 19: Combine replace with other processors
python run_experiment.py --flow "mask('alphabet',mask_char=' '),replace('\\s+',replacement=' ')"

# Example 20: Replace ground truth answer with a fixed value (e.g., 42 -> 123)
python run_experiment.py --flow "replace('{ANSWER}',replacement='123')"

# Example 21: Replace ground truth answer and shuffle
python run_experiment.py --flow "replace('{ANSWER}',replacement='999'),shuffle('line')"

# Example 22: Remove exact answer from reasoning (all occurrences)
python run_experiment.py --flow "truncate('answer')"

# Example 23: Combine answer removal with shuffle
python run_experiment.py --flow "truncate('answer'),shuffle('line')"

# Example 23a: Keep only first 10 lines, remove all lines after line 10
python run_experiment.py --flow "truncate('after_line',n=10)"

# Example 23b: Keep only first 10% of lines (ratio-based truncation)
python run_experiment.py --flow "truncate('after_line',r=0.1)"

# Example 23c: Remove first 10% of lines (inverse of after_line)
python run_experiment.py --flow "truncate('before_line',r=0.1)"

# Example 23d: Remove first 5 lines, keep the rest
python run_experiment.py --flow "truncate('before_line',n=5)"

# Example 24: Add answer prefill to guide model response format (auto-determined by dataset type)
python run_experiment.py --flow "answer('retrieval')"

# Example 25: Combine answer prefill with masking
python run_experiment.py --flow "mask('number'),answer('retrieval')"

# Example 26: Full pipeline with answer prefill
python run_experiment.py --flow "truncate('last_ratio',ratio=0.3),mask('number'),shuffle('line'),answer('retrieval')"

# Example 26a: Remove question and provide only reasoning
python run_experiment.py --flow "question('remove')"

# Example 26b: Remove question and mask numbers in reasoning
python run_experiment.py --flow "question('remove'),mask('number')"

# Example 27: Replace reasoning with answer only (pure answer)
python run_experiment.py --flow "reason_is('{ANSWER}')"

# Example 28: Replace reasoning with illustrated answer format
python run_experiment.py --flow "reason_is('Thus, the answer is {ANSWER}.')"

# Example 29: Combine reason_is with other processors (though reason_is replaces all, so order matters)
python run_experiment.py --flow "mask('number'),reason_is('{ANSWER}')"

# Example 30: Use reason_is as baseline for comparison with custom format
python run_experiment.py --flow "reason_is('The final result is {ANSWER}')"

# Example 31: Replace reasoning with math expression using ANSWER
python run_experiment.py --flow "reason_is('The answer is {ANSWER} or {ANSWER * 0.9}')"

# Example 32: Use math expression with addition
python run_experiment.py --flow "reason_is('The result is {ANSWER + 10}')"

# Example 33: Multiple math expressions in one pattern
python run_experiment.py --flow "reason_is('Options: {ANSWER}, {ANSWER * 0.9}, {ANSWER * 1.1}')"

# Example 34: Randomize all digits with default seed
python run_experiment.py --flow "random('number')"

# Example 35: Randomize all digits with custom seed
python run_experiment.py --flow "random('number',seed=123)"

# Example 36: Combine randomization with other processors
python run_experiment.py --flow "random('number'),shuffle('line')"

# Example 37: Randomize digits and mask letters
python run_experiment.py --flow "random('number'),mask('alphabet')"

# Example 38: Replace reasoning with random tokens (token mode, count = original token count)
python run_experiment.py --flow "padding('token')"

# Example 39: Replace reasoning with random tokens using custom tokenizer (count = original token count)
python run_experiment.py --flow "padding('token',tokenizer_model='gpt2',seed=42)"

# Example 40: Replace reasoning with random words (word mode, count = original word count)
python run_experiment.py --flow "padding('word',words_tsv_path='data/AIME2025__R10/gpt-oss/p1/words.tsv')"

# Example 41: Replace reasoning with random words using custom seed
python run_experiment.py --flow "padding('word',words_tsv_path='data/AIME2025__R10/gpt-oss/p1/words.tsv',seed=123)"

# Example 42: Replace reasoning with random words, then apply masking to the random words
python run_experiment.py --flow "padding('word',words_tsv_path='data/AIME2025__R10/gpt-oss/p1/words.tsv'),mask('number')"

# Example 43: Replace reasoning with random words, apply masking, then shuffle
python run_experiment.py --flow "padding('word',words_tsv_path='data/AIME2025__R10/gpt-oss/p1/words.tsv'),mask('number'),shuffle('line')"

-----------------------------------------------------------------------------
Other Parameters
-----------------------------------------------------------------------------

--results_path: Path to input results.json (default: data/AIME2025__R10/gpt-oss/p1/results.json)
                Model type is auto-detected from path (e.g., data/{DATASET}/{MODEL}/p{n}/results.json)
--output_path: Path to save output (auto-generated based on --flow if not specified)
--mode: LLM client mode (openrouter, local)
--limit: Limit number of questions for testing
--max_workers: Maximum concurrent workers (default: 16)
--max_retry: Maximum retry attempts for failed tasks (default: 1)

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
    GRADING_PROMPT,
    mask_numbers_in_reasoning,
    mask_answer_only_in_reasoning,
    mask_numbers_in_lines_with_answer,
    mask_numbers_in_nlines_with_answer,
    mask_alphabet_in_reasoning,
    mask_alphabet_and_answer_in_reasoning,
    mask_numbers_advance,
    mask_all_nonblank_in_reasoning,
    remove_answer_and_after,
    shuffle_lines,
    append_to_jsonl,
    load_from_jsonl
)
from pipeline import parse_flow, Pipeline

# Default concurrency (can be overridden by --max_workers)
DEFAULT_MAX_WORKERS = 16
GRADED_MAX_WORKERS = 64
# Default max retry (can be overridden by --max_retry)
DEFAULT_MAX_RETRY = 1

# Setup logger
logger = setup_logger(__name__, log_file='logs/run_experiment.log')

# Also setup logging for llm_client module
import logging
llm_client_logger = setup_logger('llm_client', log_file='logs/run_experiment.log', level=logging.INFO)


# =============================================================================
# Safe File Operations
# =============================================================================

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
    all_results = load_from_jsonl(stage2_jsonl)

    # Filter out results with null generated_answer
    # Any result with null generated_answer is considered a generation failure
    results = [r for r in all_results if r.get('generated_answer') is not None]

    # Log if we filtered out any results
    filtered_count = len(all_results) - len(results)
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} results with null generated_answer")
        print(f"Warning: Filtered out {filtered_count} results with null generated_answer")

    # Sort by question_id for consistency
    results.sort(key=lambda x: x.get('question_id', 0))

    # Calculate summary statistics
    total = len(all_results)  # Total including filtered results for transparency

    # generation_successful: must have generation_success=True AND non-null generated_answer
    generation_successful = sum(1 for r in all_results
                               if r.get('generation_success', False) and r.get('generated_answer') is not None)
    # generation_failed: includes both API failures AND null generated_answer results
    generation_failed = total - generation_successful

    # grading stats based on filtered results (only those that passed generation)
    grading_successful = sum(1 for r in results if r.get('success', False))
    # grading_failed: among successfully generated results, how many failed grading
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
        'accuracy': accuracy,
        'note': f'Filtered out {filtered_count} results with null generated_answer' if filtered_count > 0 else ''
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
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} results with null generated_answer")


# =============================================================================
# Path Generation
# =============================================================================

def extract_model_from_path(results_path: str) -> str:
    """
    Extract model type from results_path

    Format: data/{DATASET}/{MODEL}/p{n}/results.json
    Example: data/AIME2025__R10/gpt-oss/p1/results.json → 'gpt-oss'

    Args:
        results_path: Path to results.json

    Returns:
        Model type string

    Raises:
        ValueError: If model cannot be extracted from path
    """
    from pathlib import Path

    path_parts = Path(results_path).parts
    if 'data' in path_parts:
        data_index = path_parts.index('data')
        if len(path_parts) > data_index + 2:
            return path_parts[data_index + 2]  # MODEL 在第三位置
    raise ValueError(f"Cannot extract model from path: {results_path}")


def generate_output_path_from_flow(results_path: str, flow: str) -> str:
    """
    Generate output path based on dataset, model, and flow string

    Rules:
    - gpt-oss + AIME2025__R10: exp/<hash>/results.json
    - Other combinations: exp_{DATA}_{MODEL}/<hash>/results.json

    Args:
        results_path: Original results.json path (format: data/{DATASET}/{MODEL}/p{n}/results.json)
        flow: Flow string

    Returns:
        Generated output path

    Examples:
        data/AIME2025__R10/gpt-oss/p1/results.json + flow
        -> experiments/exp/<hash>/results.json

        data/AIME2025__R10/deepseek/p1/results.json + flow
        -> experiments/exp_AIME2025__R10_deepseek/<hash>/results.json

        data/MATH500/gpt-oss/p1/results.json + flow
        -> experiments/exp_MATH500_gpt-oss/<hash>/results.json
    """
    import hashlib
    from pathlib import Path

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

    # Extract model type from results_path
    model_type = extract_model_from_path(results_path)

    # Generate hash from flow string
    hash_obj = hashlib.md5(flow.encode('utf-8'))
    flow_hash = hash_obj.hexdigest()[:8]

    # Determine base directory based on dataset and model
    if model_type == 'gpt-oss' and dataset_name == 'AIME2025__R10':
        # Default: experiments/exp/<hash>/results.json (保持向後兼容)
        base_dir = Path("experiments/exp")
    else:
        # New format: experiments/exp_{DATA}_{MODEL}/<hash>/results.json
        base_dir = Path(f"experiments/exp_{dataset_name}_{model_type}")

    # Construct full path
    output_path = base_dir / flow_hash / "results.json"

    # Update flow_to_hash.json mapping (in base directory)
    flow_to_hash_file = base_dir / "flow_to_hash.json"
    flow_to_hash = {}

    # Load existing mapping if it exists
    if flow_to_hash_file.exists():
        try:
            with open(flow_to_hash_file, 'r', encoding='utf-8') as f:
                flow_to_hash = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {flow_to_hash_file}: {e}")

    # Add current flow if not already present
    if flow not in flow_to_hash:
        flow_to_hash[flow] = flow_hash

        # Ensure base directory exists
        flow_to_hash_file.parent.mkdir(parents=True, exist_ok=True)

        # Save updated mapping
        try:
            with open(flow_to_hash_file, 'w', encoding='utf-8') as f:
                json.dump(flow_to_hash, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save {flow_to_hash_file}: {e}")

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
    'all-nonblank': mask_all_nonblank_in_reasoning,
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
        mask_mode: Masking mode (number, answer, line, n-lines, number-advance, alphabet, alphabet-and-answer, all-nonblank)
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
    elif mask_mode in ['number', 'alphabet', 'all-nonblank']:
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
    dataset_type: str = 'math'
) -> Task:
    """
    Prepare a Task for reasoning processing (masking, truncating, shuffling, etc.)

    Note: This function supports both the new --flow syntax and both math/code formats.

    Args:
        item: Result item from results.json
        model_type: Model type
        flow: Flow string (e.g., "mask('number'),shuffle('line')").
        dataset_type: Dataset type ('math' or 'code'), used to determine answer prefix format
    """
    unique_id = item['unique_id']
    question = item['question']

    # Support both formats:
    # - Math: item['answer'] (string/number)
    # - Code: item['test_cases'] (list of test cases)
    ground_truth = item.get('test_cases') or item.get('answer', '')
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
    context = {
        'question': question,
        'answer': ground_truth,
        'ground_truth': ground_truth,
        'dataset_type': dataset_type  # Pass dataset_type to processors for auto-prefill selection
    }

    if flow_str:
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        processed_reasoning, processing_metadata = pipeline.execute(original_reasoning, context)
        # Use question from context (may be modified by processors like truncate_question)
        final_question = context['question']
    else:
        # No processing
        processed_reasoning = original_reasoning
        processing_metadata = []
        final_question = question

    # Build CompletionRequest with processed reasoning
    # Get answer_prefill and max_tokens from context (set by AnswerProcessor if using answer('retrieval'))
    answer_prefix = context.get('answer_prefill', '')
    # Default to 5000 if max_tokens not set by AnswerProcessor
    max_tokens = context.get('max_tokens', 5000)

    # Get system_prompt from baseline result (if available)
    system_prompt = item.get('result', {}).get('sys_prompt', 'You are a helpful assistant')

    # Create CompletionRequest (template will be applied by LLMClient)
    request = CompletionRequest(
        question=final_question,
        reasoning=processed_reasoning,
        answer_prefix=answer_prefix,
        model_type=model_type,
        temperature=0.01,
        max_tokens=max_tokens,
        system_prompt=system_prompt
        # Note: min_tokens not supported by OpenRouter API
    )

    # Create Task with metadata
    metadata = {
        'unique_id': unique_id,
        'question': question,
        'ground_truth': ground_truth,
        'original_reasoning': original_reasoning,
        'processed_reasoning': processed_reasoning,
        'flow': flow_str,
        'processing_metadata': processing_metadata,
    }

    # Preserve test_cases for code datasets
    # ground_truth can be either answer (string) or test_cases (list)
    if isinstance(ground_truth, list):
        metadata['test_cases'] = ground_truth

    # Add answer_prefill to metadata if present
    if 'answer_prefill' in context:
        metadata['answer_prefill'] = context['answer_prefill']

    task = Task(
        index=index,
        request=request,
        metadata=metadata
    )

    return task


def _build_generated_answer(response_content: str, metadata: Dict) -> str:
    """
    Unified helper to construct generated_answer with answer_prefix.

    When using answer('retrieval'), the API response doesn't include the prefix.
    This function prepends the answer_prefix from metadata if present.

    Args:
        response_content: LLM API response content
        metadata: Task metadata containing optional 'answer_prefill' key

    Returns:
        Complete generated_answer with prefix prepended (if available)
    """
    answer_prefix = metadata.get('answer_prefill', '')
    if answer_prefix:
        return answer_prefix + " " + response_content.lstrip()
    return response_content


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

    # Prepare generated_answer with answer_prefix if present
    generated_answer = None
    if success:
        generated_answer = _build_generated_answer(response.content, metadata)

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
        'generated_answer': generated_answer,
        'is_correct': None,
        'grading_reasoning': None,
        'generation_success': success,
        'success': False,
        'error': error,
        'retry_count': 0  # Track retry attempts
    }

    # Preserve test_cases for code datasets
    if 'test_cases' in metadata:
        result['test_cases'] = metadata['test_cases']

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
    # Generate human-readable experiment name from flow
    experiment_name = flow_str if flow_str else "no_processing"
    if len(experiment_name) > 100:
        experiment_name = experiment_name[:97] + "..."

    experiment_metadata = {
        'experiment_id': Path(output_path).parent.name,  # Hash value for file system
        'experiment_name': experiment_name,  # Human-readable flow string
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


def detect_dataset_type(results: List[Dict]) -> str:
    """
    自動檢測數據集類型

    採樣檢查多個結果以確定數據集類型，基於 unique_id 和 result 結構。

    Args:
        results: 結果列表

    Returns:
        'code'、'science' 或 'math'（預設）
    """
    if not results:
        return 'math'  # 預設為 math

    # 定義數據集類型的檢測模式
    dataset_patterns = {
        'code': ['codeforces', 'codeeml', 'arc', 'codeelo'],  # Code contest platforms
        'science': ['gpqa', 'gsm8k'],  # Science/knowledge datasets
        'math': ['aime', 'math500', 'mathdialogue']  # Math datasets
    }

    # 採樣多個結果以驗證一致性（取前3個或全部）
    sample_size = min(3, len(results))
    detected_types = []

    for i in range(sample_size):
        result = results[i]
        unique_id = result.get('unique_id', '').lower()

        # 檢查 unique_id 中的模式
        detected_type = None
        for dataset_type, patterns in dataset_patterns.items():
            if any(pattern.lower() in unique_id for pattern in patterns):
                detected_type = dataset_type
                break

        # 如果無法從 unique_id 確定，檢查 result 結構
        if detected_type is None:
            # Code datasets 通常有 test_cases 而不是 answer
            if 'test_cases' in result and 'answer' not in result:
                detected_type = 'code'
            # Science datasets 可能有多選答案
            elif result.get('answer', '').upper() in ['A', 'B', 'C', 'D']:
                detected_type = 'science'
            else:
                detected_type = 'math'  # 預設

        detected_types.append(detected_type)

    # 如果多個樣本類型一致，使用該類型；否則使用第一個檢測到的類型
    if detected_types:
        first_type = detected_types[0]
        if all(t == first_type for t in detected_types):
            return first_type
        else:
            # 不同的類型被檢測到，記錄警告但使用第一個
            logger.debug(
                f"Mixed dataset types detected in sample: {detected_types}. "
                f"Using first detected type: {first_type}"
            )
            return first_type

    return 'math'  # 備用預設


def create_code_grading_tasks(results: List[Dict]) -> List[Dict]:
    """
    創建代碼本地執行任務（非 LLM Task）

    支援兩種格式：
    1. Baseline 格式：代碼在 result.answer，測試用例在 test_cases
    2. Experiment 格式：代碼在 generated_answer，測試用例在 test_cases

    Args:
        results: 結果列表

    Returns:
        本地執行任務列表（字典格式）
    """
    from core import extract_code_blocks

    tasks = []
    for result in results:
        if not result.get('generation_success', False):
            continue

        # 支援兩種格式提取代碼：
        # 1. Baseline: result.answer
        # 2. Experiment: generated_answer
        if 'generated_answer' in result and result['generated_answer']:
            answer_text = result['generated_answer']
        else:
            answer_text = result.get('result', {}).get('answer', '')

        code_blocks = extract_code_blocks(answer_text)

        # 優先選擇 C++ 代碼塊（標記為 cpp, c++, C++）
        cpp_blocks = [block for block in code_blocks if block[0].lower() in ['cpp', 'c++', 'c']]
        if cpp_blocks:
            code = cpp_blocks[0][1]
        elif code_blocks:
            # Fallback: 使用第一個代碼塊
            code = code_blocks[0][1]
        else:
            code = ""

        # 從 test_cases 欄位獲取測試用例
        test_cases = result.get('test_cases', [])

        tasks.append({
            'type': 'code_execution',
            'unique_id': result['unique_id'],
            'question_id': result.get('question_id', 0),
            'code': code,
            'test_cases': test_cases,
            'result': result
        })

    return tasks


def create_math_grading_tasks(results: List[Dict], judge_model_type: str = 'gpt-oss') -> List[Task]:
    """Create grading tasks for mathematical answers (original logic)"""
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


def create_grading_tasks(
    results: List[Dict],
    judge_model_type: str = 'gpt-oss',
    dataset_type: str = None
) -> List:
    """
    Create grading tasks (supports both math and code datasets)

    Args:
        results: 結果列表
        judge_model_type: LLM 判官模型（僅用於 math）
        dataset_type: 數據集類型 ('math' 或 'code')，None 為自動檢測

    Returns:
        任務列表（math 為 List[Task]，code 為 List[Dict]）
    """
    # 自動檢測數據集類型
    if dataset_type is None:
        dataset_type = detect_dataset_type(results)

    if dataset_type == 'code':
        return create_code_grading_tasks(results)
    else:
        return create_math_grading_tasks(results, judge_model_type)


def run_experiment(
    results_path: str,
    output_path: str,
    mode: str = 'openrouter',
    flow: str = None,
    limit: int = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retry: int = DEFAULT_MAX_RETRY
):
    """
    Run the mask numbers experiment using JSONL strategy

    Args:
        results_path: Path to results.json (format: data/{DATASET}/{MODEL}/p{n}/results.json)
        output_path: Path to save output (e.g., exp/mask_number/results.json)
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
    # Extract model_type from results_path
    model_type = extract_model_from_path(results_path)
    print(f"Detected model: {model_type}")

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
    # A result is considered truly successful only if:
    # 1. generation_success = True AND
    # 2. generated_answer is not None (to avoid including null answers)
    completed_stage1_ids = {
        r['unique_id'] for r in existing_stage1
        if r.get('generation_success', False) and r.get('generated_answer') is not None
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
    client = LLMClient(mode=mode, timeout=60)

    # Detect dataset type early (before preparing tasks)
    dataset_type = detect_dataset_type(data)
    print(f"Dataset type: {dataset_type}")

    # Prepare tasks
    print("Preparing processing tasks...")
    tasks = []
    for item in data:
        task = prepare_task(item, model_type, flow=flow_str, dataset_type=dataset_type)
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
                        generated_answer = _build_generated_answer(response.content, completed_task.metadata)
                        stage1_results_map[unique_id]['generated_answer'] = generated_answer
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

    # Get all successful Stage 1 results (must have non-null generated_answer)
    stage1_successful = [
        r for r in stage1_results_map.values()
        if r.get('generation_success', False) and r.get('generated_answer') is not None
    ]

    # Identify which need grading
    need_grading = [
        r for r in stage1_successful
        if r['unique_id'] not in stage2_results_map or not stage2_results_map[r['unique_id']].get('success', False)
    ]

    print(f"Total successful generations: {len(stage1_successful)}")
    print(f"Already graded: {len(stage1_successful) - len(need_grading)}")
    print(f"Need to grade: {len(need_grading)}")

    # Note: dataset_type was already detected in Phase 1

    grading_tasks = create_grading_tasks(need_grading, judge_model_type='gpt-oss')

    if len(grading_tasks) > 0:
        if dataset_type == 'code':
            # Code grading: local C++ execution
            print(f"Executing C++ code for {len(grading_tasks)} tasks...")
            from core import compile_and_execute_cpp, normalize_output

            for task in tqdm(grading_tasks, desc="Executing C++"):
                code = task['code']
                test_cases = task['test_cases']

                # Execute all test cases
                all_passed = True
                execution_details = []

                for test_input, expected_output in test_cases:
                    exec_result = compile_and_execute_cpp(
                        code=code,
                        test_input=test_input,
                        timeout=2
                    )

                    # Check status and output
                    if exec_result['status'] != 'AC':
                        all_passed = False
                    elif normalize_output(exec_result['output']) != normalize_output(expected_output):
                        all_passed = False
                        exec_result['status'] = 'WA'

                    execution_details.append({
                        'input': test_input[:50],
                        'expected': expected_output[:50],
                        'actual': exec_result['output'][:50],
                        'status': exec_result['status']
                    })

                # Update result
                graded_result = task['result'].copy()
                graded_result['is_correct'] = all_passed
                graded_result['grading_reasoning'] = json.dumps(execution_details, indent=2)
                graded_result['grading_status'] = 'AC' if all_passed else execution_details[0]['status']
                graded_result['success'] = True

                # Update Stage 2 map
                stage2_results_map[graded_result['unique_id']] = graded_result

                # Append to Stage 2 JSONL
                append_to_jsonl(stage2_jsonl, graded_result)

        else:
            # Math grading: use LLM
            print(f"Grading {len(grading_tasks)} answers with LLM...")
            for grading_task in tqdm(client.generate_concurrent(grading_tasks, max_workers=GRADED_MAX_WORKERS),
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

    # Generate human-readable experiment name from flow
    # Use flow string (truncated if too long) as primary identifier
    # Hash is stored as experiment_id for file system organization
    experiment_name = flow_str if flow_str else "no_processing"
    if len(experiment_name) > 100:
        experiment_name = experiment_name[:97] + "..."

    experiment_metadata = {
        'experiment_id': output_json.parent.name,  # Hash value for file system
        'experiment_name': experiment_name,  # Human-readable flow string
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
        mode=args.mode,
        flow=args.flow,
        limit=args.limit,
        max_workers=args.max_workers,
        max_retry=args.max_retry
    )


if __name__ == '__main__':
    main()
