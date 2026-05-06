"""
Core utilities for reasoning behavior experiments

This module contains common functions and constants used across multiple
experiment scripts including:
- Data loading functions
- Parsing functions
- Debug utilities
- Text processing functions
- Prompt construction functions
- Prompt templates
"""

import json
import os
import re
import random
from typing import List, Dict, Tuple
from datetime import datetime


# =============================================================================
# Debug Utilities
# =============================================================================

DEBUG = os.environ.get('DEBUG', '0') == '1'


def debug_print(msg: str):
    """Print debug message if DEBUG is enabled"""
    if DEBUG:
        print(msg)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_existing_results(results_path: str) -> List[Dict]:
    """Load existing results.json"""
    with open(results_path, 'r') as f:
        return json.load(f)


def load_existing_grades(grades_path: str) -> Dict[int, bool]:
    """
    Load existing grades.json and return mapping of index to correctness

    Args:
        grades_path: Path to grades.json file

    Returns:
        Dictionary mapping question index to correctness (bool)
    """
    with open(grades_path, 'r') as f:
        grades_data = json.load(f)

    # Create mapping from index to correctness
    grade_map = {}
    for grade in grades_data['grades']:
        grade_map[grade['index']] = grade['correct']

    return grade_map


# =============================================================================
# Parsing Functions
# =============================================================================

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


# =============================================================================
# Text Processing Functions
# =============================================================================

def clean_multiple_newlines(text: str) -> str:
    """Replace multiple consecutive newlines with a single newline"""
    cleaned = re.sub(r'\n\n+', '\n', text)
    return cleaned


def remove_continuous_blanks(text: str) -> str:
    """
    Replace multiple consecutive blank characters (spaces, tabs, newlines) with a single space

    This function consolidates all continuous whitespace (including newlines, spaces, tabs)
    into a single space character.

    Args:
        text: Input text with potential multiple consecutive blank characters

    Returns:
        Text with all continuous blanks replaced by a single space

    Example:
        >>> remove_continuous_blanks("hello\\n\\n\\n  \\nworld   \\t\\n\\n   test")
        'hello world test'
    """
    # Replace all consecutive whitespace characters (spaces, tabs, newlines) with a single space
    result = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return result.strip()


def extract_nonempty_lines(text: str) -> List[str]:
    """
    Extract non-empty lines from text

    Args:
        text: Input text with multiple lines

    Returns:
        List of non-empty lines (whitespace-only lines are excluded)
    """
    lines = text.strip().split('\n')
    return [line for line in lines if line.strip()]


# =============================================================================
# Text Masking Utilities
# =============================================================================

def mask_numbers_in_reasoning(reasoning: str, mask_char: str = '█') -> str:
    """
    Mask all numbers (digits 0-9) in reasoning with specified mask character

    Args:
        reasoning: Original reasoning content
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with all digits replaced by mask_char

    Examples:
        >>> mask_numbers_in_reasoning("The answer is 42")
        'The answer is ██'

        >>> mask_numbers_in_reasoning("1 + 2 = 3", mask_char='*')
        '* + * = *'
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

    Examples:
        >>> mask_answer_only_in_reasoning("The answer is 42", "42")
        'The answer is ██'

        >>> mask_answer_only_in_reasoning("We have 123 and 23", "23")
        'We have 123 and ██'
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

    Examples:
        >>> mask_numbers_in_lines_with_answer("Line 1: 10\\nAnswer: 42", "42")
        'Line 1: 10\\nAnswer: ██'
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

    Examples:
        >>> mask_numbers_in_nlines_with_answer("L1: 10\\nL2: 20\\nAnswer: 42", "42", n=1)
        'L1: 10\\nL█: ██\\nAnswer: ██'
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


def mask_alphabet_in_reasoning(reasoning: str, mask_char: str = '█') -> str:
    """
    Mask all alphabetic characters (A-Z and a-z) in reasoning with specified mask character

    Args:
        reasoning: Original reasoning content
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with all letters replaced by mask_char

    Examples:
        >>> mask_alphabet_in_reasoning("Hello 123")
        '█████ 123'
    """
    # Replace all alphabetic characters (A-Z and a-z) with mask_char
    masked_reasoning = re.sub(r'[A-Za-z]', mask_char, reasoning)

    return masked_reasoning


def mask_alphabet_and_answer_in_reasoning(reasoning: str, answer: str, mask_char: str = '█') -> str:
    """
    Mask all alphabetic characters (A-Z and a-z) AND the answer number in reasoning

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer to mask
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with all letters and answer occurrences replaced by mask_char

    Examples:
        >>> mask_alphabet_and_answer_in_reasoning("The answer is 42", "42")
        '███ ██████ ██ ██'
    """
    # First mask all alphabetic characters
    masked_reasoning = re.sub(r'[A-Za-z]', mask_char, reasoning)

    # Then mask the answer (same logic as mask_answer_only_in_reasoning)
    answer_clean = answer.strip()
    answer_escaped = re.escape(answer_clean)
    masked_answer = mask_char * len(answer_clean)
    masked_reasoning = re.sub(r'\b' + answer_escaped + r'\b', masked_answer, masked_reasoning)

    return masked_reasoning


def mask_all_nonblank_in_reasoning(reasoning: str, mask_char: str = '█') -> str:
    """
    Mask all non-blank characters (letters, numbers, symbols) in reasoning text,
    preserving only whitespace characters (spaces, tabs, newlines).

    Args:
        reasoning: Original reasoning content
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with all non-whitespace characters replaced by mask_char

    Examples:
        >>> mask_all_nonblank_in_reasoning("Hello World\\n123 + 456")
        '█████ █████\\n███ █ ███'
    """
    # Replace all non-whitespace characters with mask_char
    masked_reasoning = re.sub(r'\S', mask_char, reasoning)

    return masked_reasoning


def mask_non_numbers_in_reasoning(reasoning: str, mask_char: str = '█') -> str:
    """
    Mask all non-digit characters in reasoning text, preserving only digits (0-9).

    This function masks all characters that are not digits, including:
    - Alphabetic characters (A-Z, a-z)
    - Symbols and punctuation (+, -, *, =, etc.)
    - Whitespace characters (spaces, tabs, newlines) - preserved as-is

    Only digits 0-9 are preserved in their original form.

    Args:
        reasoning: Original reasoning content
        mask_char: Character to use for masking (default: '█')

    Returns:
        Reasoning content with all non-digit characters (except whitespace) replaced by mask_char

    Examples:
        >>> mask_non_numbers_in_reasoning("Calculate 2 + 2 = 4")
        '█████████ 2 █ 2 █ 4'

        >>> mask_non_numbers_in_reasoning("Answer is 42", mask_char=' ')
        '       is 42'

        >>> mask_non_numbers_in_reasoning("Step 1: 5 + 3 = 8")
        '████ 1█ 5 █ 3 █ 8'
    """
    # Replace all non-digit characters except whitespace with mask_char
    # Pattern: \D matches any non-digit, but we need to preserve whitespace
    # So we use negative lookahead to exclude whitespace from matching
    masked_reasoning = re.sub(r'[^\d\s]', mask_char, reasoning)

    return masked_reasoning


def randomize_numbers_in_reasoning(reasoning: str, seed: int = None) -> str:
    """
    Randomize all digits (0-9) in reasoning text to random digits.

    Each digit is independently replaced with a random digit from 0-9.
    Useful for testing if models rely on specific numeric values or just
    the structural pattern of reasoning.

    Args:
        reasoning: Original reasoning content
        seed: Random seed for reproducibility (optional)

    Returns:
        Reasoning content with all digits randomly replaced

    Examples:
        >>> randomize_numbers_in_reasoning("The answer is 42", seed=42)
        'The answer is 17'  # Example output with seed

        >>> randomize_numbers_in_reasoning("Calculate 2 + 2 = 4")
        'Calculate 7 + 3 = 9'  # Example output (varies without seed)

        >>> randomize_numbers_in_reasoning("Step 1: 5 + 3 = 8", seed=123)
        'Step 4: 2 + 7 = 1'  # Example output with seed
    """
    if seed is not None:
        random.seed(seed)

    # Replace each digit with a random digit (0-9)
    def replace_digit(match):
        return str(random.randint(0, 9))

    randomized_reasoning = re.sub(r'\d', replace_digit, reasoning)

    return randomized_reasoning


def mask_numbers_advance(reasoning: str, answer: str = None, mask_char: str = '█') -> str:
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
        >>> mask_numbers_advance("A12 and 1+2")
        'A12 and █+█'

        >>> mask_numbers_advance("x42", answer="42")
        'x██'
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


# =============================================================================
# Text Preprocessing Utilities
# =============================================================================

def remove_answer_and_after(reasoning: str, answer: str) -> str:
    """
    Remove the line containing the answer and all subsequent lines

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer to identify the line to remove from

    Returns:
        Reasoning content with answer line and all lines after it removed

    Examples:
        >>> remove_answer_and_after("L1\\nAnswer: 42\\nL3", "42")
        'L1'
    """
    # Clean answer string (remove potential whitespace)
    answer_clean = answer.strip()

    # Escape special regex characters in answer for matching
    answer_escaped = re.escape(answer_clean)

    # Split reasoning into lines
    lines = reasoning.split('\n')

    # Find the first line that contains the answer (with word boundaries)
    answer_line_index = None
    for i, line in enumerate(lines):
        if re.search(r'\b' + answer_escaped + r'\b', line):
            answer_line_index = i
            break

    # If answer is found, keep only lines before it
    if answer_line_index is not None:
        lines = lines[:answer_line_index]

    return '\n'.join(lines)


def remove_before_answer(reasoning: str, answer: str) -> str:
    """
    Remove all lines before the line containing the answer (excluding the answer line itself)

    This is the opposite of remove_answer_and_after().

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer to identify the line to keep from

    Returns:
        Reasoning content with all lines before the answer line removed (answer line and after are kept)

    Examples:
        >>> remove_before_answer("L1\\nL2\\nAnswer: 42\\nL4", "42")
        'Answer: 42\\nL4'
    """
    # Clean answer string (remove potential whitespace)
    answer_clean = answer.strip()

    # Escape special regex characters in answer for matching
    answer_escaped = re.escape(answer_clean)

    # Split reasoning into lines
    lines = reasoning.split('\n')

    # Find the first line that contains the answer (with word boundaries)
    answer_line_index = None
    for i, line in enumerate(lines):
        if re.search(r'\b' + answer_escaped + r'\b', line):
            answer_line_index = i
            break

    # If answer is found, keep only lines from the answer line onwards
    if answer_line_index is not None:
        lines = lines[answer_line_index:]

    return '\n'.join(lines)


def remove_exact_answer(reasoning: str, answer: str) -> str:
    """
    Remove the exact answer string from reasoning text

    This function finds and removes all occurrences of the exact answer string
    from the reasoning text, using word boundaries to avoid partial matches.

    Args:
        reasoning: Original reasoning content
        answer: The ground truth answer to remove

    Returns:
        Reasoning content with the exact answer removed

    Examples:
        >>> remove_exact_answer("The answer is 42.", "42")
        'The answer is .'
        >>> remove_exact_answer("Result: 123, which equals 123.", "123")
        'Result: , which equals .'
    """
    # Clean answer string (remove potential whitespace)
    answer_clean = answer.strip()

    if not answer_clean:
        return reasoning

    # Escape special regex characters in answer for matching
    answer_escaped = re.escape(answer_clean)

    # Replace all occurrences of the answer with empty string (using word boundaries)
    result = re.sub(r'\b' + answer_escaped + r'\b', '', reasoning)

    return result


def shuffle_lines(reasoning: str, seed: int = None) -> str:
    """
    Shuffle reasoning content line-by-line

    Args:
        reasoning: Original reasoning content
        seed: Random seed for reproducibility

    Returns:
        Shuffled reasoning content

    Examples:
        >>> shuffle_lines("L1\\nL2\\nL3", seed=42)
        # Returns shuffled lines with deterministic order due to seed
    """
    if seed is not None:
        random.seed(seed)

    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Shuffle
    random.shuffle(lines)

    return '\n'.join(lines)


# =============================================================================
# Prompt Construction Functions
# =============================================================================

def build_gpt_oss_prompt_with_reasoning(
    question: str,
    reasoning: str,
    reasoning_effort: str = "high",
    empty_question: bool = False
) -> str:
    """
    Build GPT-OSS prompt with prefilled reasoning for text completion

    Based on chat_template.jinja, the format should be:
    <|start|>system<|message|>{system_message}<|end|>
    <|start|>user<|message|>{question}<|end|>
    <|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
    <|start|>assistant<|channel|>final<|message|>

    The system message includes model identity, date, and reasoning effort.

    Args:
        question: The question to ask
        reasoning: The reasoning content to prefill
        reasoning_effort: Reasoning effort level (default: "high")
        empty_question: If True, replace question with empty string (default: False)

    Returns:
        Complete prompt string ready for completion API
    """
    # Build system message (based on build_system_message macro in template)
    model_identity = "You are ChatGPT, a large language model trained by OpenAI."
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_message = f"{model_identity}\n"
    system_message += "Knowledge cutoff: 2024-06\n"
    system_message += f"Current date: {current_date}\n\n"
    system_message += f"Reasoning: {reasoning_effort}\n\n"
    system_message += "# Valid channels: analysis, commentary, final. Channel must be included for every message."

    # Replace question with empty string if requested
    question_text = "" if empty_question else question

    # Build complete prompt
    prompt = f"<|start|>system<|message|>{system_message}<|end|>"
    prompt += f"<|start|>user<|message|>{question_text}<|end|>"
    prompt += f"<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
    prompt += f"<|start|>assistant<|channel|>final<|message|>"

    debug_print(f'\n[DEBUG] Built prompt:\n{prompt}\n')
    return prompt


def build_gpt_oss_prompt_with_reasoning_prefilled_answer(
    question: str,
    reasoning: str,
    prefill_text: str = "Thus, the answer is",
    reasoning_effort: str = "high",
    empty_question: bool = False
) -> str:
    """
    Build GPT-OSS prompt with prefilled reasoning and answer prefix for text completion

    This function is similar to build_gpt_oss_prompt_with_reasoning, but adds
    prefill text at the beginning of the answer section to guide the model's
    response format.

    Based on chat_template.jinja, the format should be:
    <|start|>system<|message|>{system_message}<|end|>
    <|start|>user<|message|>{question}<|end|>
    <|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
    <|start|>assistant<|channel|>final<|message|>{prefill_text}

    Args:
        question: The question to ask
        reasoning: The reasoning content to prefill
        prefill_text: Text to prefill at the start of answer (default: "Thus, the answer is")
        reasoning_effort: Reasoning effort level (default: "high")
        empty_question: If True, replace question with empty string (default: False)

    Returns:
        Complete prompt string ready for completion API with prefilled answer text
    """
    # Build system message (based on build_system_message macro in template)
    model_identity = "You are ChatGPT, a large language model trained by OpenAI."
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_message = f"{model_identity}\n"
    system_message += "Knowledge cutoff: 2024-06\n"
    system_message += f"Current date: {current_date}\n\n"
    system_message += f"Reasoning: {reasoning_effort}\n\n"
    system_message += "# Valid channels: analysis, commentary, final. Channel must be included for every message."

    # Replace question with empty string if requested
    question_text = "" if empty_question else question

    # Build complete prompt with prefilled answer text
    prompt = f"<|start|>system<|message|>{system_message}<|end|>"
    prompt += f"<|start|>user<|message|>{question_text}<|end|>"
    prompt += f"<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
    prompt += f"<|start|>assistant<|channel|>final<|message|>{prefill_text}"

    debug_print(f'\n[DEBUG] Built prompt with prefilled answer:\n{prompt}\n')
    return prompt


# =============================================================================
# Prompt Templates
# =============================================================================

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


SAME_ANSWER_PROMPT = """Compare these two answers and determine if they are mathematically equivalent.

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

**Task:**
Determine if these two answers are the same or equivalent.

**Guidelines:**
- Consider mathematical equivalence (e.g., 1/2 = 0.5, 2x = x + x)
- Ignore formatting differences if the mathematical content is the same
- Answer with \\boxed{{YES}} if they are the same, or \\boxed{{NO}} if they are different

Provide your reasoning first, then give your final answer in \\boxed{{}}."""


# =============================================================================
# Truncate Functions
# =============================================================================

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


def truncate_after_line(reasoning: str, line_num: int = None, ratio: float = None) -> str:
    """
    Keep only the first N lines or first X% of lines and remove the rest

    Args:
        reasoning: Original reasoning content
        line_num: Number of lines to keep (lines 1 to N) - optional if ratio is provided
        ratio: Ratio of lines to keep (0.0 to 1.0) - optional if line_num is provided

    Returns:
        Truncated reasoning content (first N or X% non-empty lines)

    Examples:
        >>> reasoning = "Line 1\\nLine 2\\nLine 3\\nLine 4\\nLine 5"
        >>> truncate_after_line(reasoning, line_num=2)
        'Line 1\\nLine 2'
        >>> truncate_after_line(reasoning, ratio=0.4)  # 40% of 5 lines = 2 lines
        'Line 1\\nLine 2'
        >>> truncate_after_line(reasoning, line_num=10)  # More than available lines
        'Line 1\\nLine 2\\nLine 3\\nLine 4\\nLine 5'
    """
    if line_num is None and ratio is None:
        raise ValueError("Either line_num or ratio must be provided")

    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Calculate number of lines to keep
    if line_num is not None:
        # Use explicit line number (priority over ratio)
        if line_num <= 0:
            return ""
        num_to_keep = line_num
    else:
        # Use ratio to calculate
        if ratio <= 0:
            return ""
        num_to_keep = int(len(lines) * ratio)
        if num_to_keep == 0 and ratio > 0 and len(lines) > 0:
            num_to_keep = 1  # Keep at least 1 line if ratio > 0

    # Keep only first N lines
    kept_lines = lines[:num_to_keep]
    return '\n'.join(kept_lines)


def truncate_before_line(reasoning: str, line_num: int = None, ratio: float = None) -> str:
    """
    Remove the first N lines or first X% of lines and keep the rest

    Args:
        reasoning: Original reasoning content
        line_num: Number of lines to remove from the beginning - optional if ratio is provided
        ratio: Ratio of lines to remove from the beginning (0.0 to 1.0) - optional if line_num is provided

    Returns:
        Truncated reasoning content (remaining lines after removing first N or X%)

    Examples:
        >>> reasoning = "Line 1\\nLine 2\\nLine 3\\nLine 4\\nLine 5"
        >>> truncate_before_line(reasoning, line_num=2)
        'Line 3\\nLine 4\\nLine 5'
        >>> truncate_before_line(reasoning, ratio=0.4)  # Remove 40% of 5 lines = 2 lines
        'Line 3\\nLine 4\\nLine 5'
        >>> truncate_before_line(reasoning, line_num=10)  # More than available lines
        ''
    """
    if line_num is None and ratio is None:
        raise ValueError("Either line_num or ratio must be provided")

    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Calculate number of lines to remove
    if line_num is not None:
        # Use explicit line number (priority over ratio)
        if line_num <= 0:
            return reasoning  # Don't remove anything if line_num <= 0
        num_to_remove = line_num
    else:
        # Use ratio to calculate
        if ratio <= 0:
            return '\n'.join(lines)  # Don't remove anything if ratio <= 0
        num_to_remove = int(len(lines) * ratio)
        # If ratio > 0 but rounds to 0, remove at least 1 line
        if num_to_remove == 0 and ratio > 0 and len(lines) > 0:
            num_to_remove = 1

    # Remove first N lines and keep the rest
    if num_to_remove >= len(lines):
        return ""  # All lines removed

    remaining_lines = lines[num_to_remove:]
    return '\n'.join(remaining_lines)


# =============================================================================
# Shuffle Functions
# =============================================================================

# Global tokenizer cache for token-level shuffle
_TOKENIZER_CACHE = {}

# Model type to tokenizer name mapping
MODEL_TYPE_TO_TOKENIZER = {
    'deepseek': 'deepseek-ai/DeepSeek-V3',
    'gpt-oss': 'openai/gpt-oss-120b',
    'qwen3': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    'olmo': 'allenai/Olmo-3-7B-Instruct',
}


def shuffle_reasoning_words(reasoning: str, seed: int = None) -> str:
    """
    Shuffle reasoning content word-by-word

    Args:
        reasoning: Original reasoning content
        seed: Random seed for reproducibility (optional)

    Returns:
        Shuffled reasoning content with words shuffled
    """
    if seed is not None:
        random.seed(seed)

    # Split by whitespace to get words
    words = reasoning.split()

    # Shuffle words
    random.shuffle(words)

    # Rejoin with single space
    return ' '.join(words)


def shuffle_words_in_line(reasoning: str, seed: int = None) -> str:
    """
    Shuffle words within each line independently

    This function preserves line boundaries while shuffling words within each line.
    Each line's words are shuffled independently with the same seed for reproducibility.

    Args:
        reasoning: Original reasoning content
        seed: Random seed for reproducibility (optional)

    Returns:
        Shuffled reasoning content with words shuffled within each line

    Examples:
        >>> shuffle_words_in_line("The quick brown\\nJumps over lazy", seed=42)
        "brown quick The\\nlazy Jumps over"  # Each line shuffled independently
    """
    if not reasoning:
        return ""

    # Split by newline to get lines
    lines = reasoning.split('\n')

    # Shuffle words in each line independently
    shuffled_lines = []
    for i, line in enumerate(lines):
        if not line.strip():  # Empty or whitespace-only line
            shuffled_lines.append(line)
            continue

        # Set seed for this line (combining global seed with line index for independence)
        if seed is not None:
            random.seed(seed + i)

        # Split line into words and shuffle
        words = line.split()
        random.shuffle(words)

        # Rejoin words with single space
        shuffled_lines.append(' '.join(words))

    # Rejoin lines with newline
    return '\n'.join(shuffled_lines)


def shuffle_reasoning_tokens(reasoning: str, tokenizer_model: str = None, model_type: str = None, seed: int = None) -> str:
    """
    Shuffle reasoning content token-by-token using the model's tokenizer

    Args:
        reasoning: Original reasoning content
        tokenizer_model: Model name for tokenizer (optional, will use model_type if not provided)
        model_type: Model type ('deepseek', 'gpt-oss', 'qwen3') to automatically select tokenizer
        seed: Random seed for reproducibility (optional)

    Returns:
        Shuffled reasoning content with tokens shuffled

    Raises:
        ValueError: If neither tokenizer_model nor model_type is provided, or if model_type is invalid
    """
    from transformers import AutoTokenizer

    if seed is not None:
        random.seed(seed)

    # Handle empty string
    if not reasoning:
        return ""

    # Determine tokenizer model to use
    if tokenizer_model is None:
        if model_type is None:
            raise ValueError("Either tokenizer_model or model_type must be provided")

        if model_type not in MODEL_TYPE_TO_TOKENIZER:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(MODEL_TYPE_TO_TOKENIZER.keys())}")

        tokenizer_model = MODEL_TYPE_TO_TOKENIZER[model_type]
        debug_print(f"Using tokenizer for {model_type}: {tokenizer_model}")

    # Load tokenizer (with caching to avoid reloading)
    if tokenizer_model not in _TOKENIZER_CACHE:
        debug_print(f"Loading tokenizer for {tokenizer_model}...")
        _TOKENIZER_CACHE[tokenizer_model] = AutoTokenizer.from_pretrained(
            tokenizer_model, trust_remote_code=True
        )

    tokenizer = _TOKENIZER_CACHE[tokenizer_model]

    # Tokenize
    tokens = tokenizer.encode(reasoning, add_special_tokens=False)

    # Shuffle token IDs
    random.shuffle(tokens)

    # Decode back to text
    shuffled_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return shuffled_text


def shuffle_reasoning(reasoning: str, mode: str = 'line', seed: int = None, **kwargs) -> str:
    """
    Unified shuffle interface supporting multiple shuffle modes

    Args:
        reasoning: Original reasoning content
        mode: Type of shuffle - 'line', 'word', 'in-line-word', or 'token'
        seed: Random seed for reproducibility (optional)
        **kwargs: Additional arguments (e.g., tokenizer_model or model_type for token mode)

    Returns:
        Shuffled reasoning content

    Raises:
        ValueError: If mode is not one of 'line', 'word', 'in-line-word', or 'token'
    """
    if mode == 'line':
        return shuffle_lines(reasoning, seed=seed)
    elif mode == 'word':
        return shuffle_reasoning_words(reasoning, seed=seed)
    elif mode == 'in-line-word':
        return shuffle_words_in_line(reasoning, seed=seed)
    elif mode == 'token':
        tokenizer_model = kwargs.get('tokenizer_model')
        model_type = kwargs.get('model_type')
        return shuffle_reasoning_tokens(reasoning, tokenizer_model=tokenizer_model, model_type=model_type, seed=seed)
    else:
        raise ValueError(f"Invalid shuffle mode: {mode}. Must be 'line', 'word', 'in-line-word', or 'token'")


# =============================================================================
# Padding Operations (Random Token/Word Generation)
# =============================================================================

# Cache for English token IDs per tokenizer
_ENGLISH_TOKEN_CACHE = {}

# Cache for word frequency data
_WORD_FREQ_CACHE = None


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

    debug_print(f"Filtering English tokens from vocabulary (size: {vocab_size})...")
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

    debug_print(f"Found {len(english_token_ids)} English tokens out of {vocab_size}")

    # Cache the result
    _ENGLISH_TOKEN_CACHE[tokenizer_name] = english_token_ids

    return english_token_ids


def load_word_frequency(words_tsv_path: str) -> Tuple[List[str], List[float]]:
    """
    Load word frequency from TSV file and prepare for weighted sampling

    Args:
        words_tsv_path: Path to words.tsv file (word\\tfrequency format)

    Returns:
        (words_list, weights_list) for random.choices
    """
    global _WORD_FREQ_CACHE

    if _WORD_FREQ_CACHE is not None:
        return _WORD_FREQ_CACHE

    debug_print(f"Loading word frequency from {words_tsv_path}...")
    words = []
    frequencies = []

    with open(words_tsv_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                word = parts[0]
                freq = int(parts[1])
                words.append(word)
                frequencies.append(freq)

    debug_print(f"Loaded {len(words)} unique words")

    # Cache the result
    _WORD_FREQ_CACHE = (words, frequencies)

    return words, frequencies


def generate_random_tokens(num_tokens: int, tokenizer, seed: int = None) -> str:
    """
    Generate a string of random English-only tokens

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


def generate_random_words(num_words: int, words_tsv_path: str, seed: int = None) -> str:
    """
    Generate a string of random words sampled from word frequency distribution

    Args:
        num_words: Number of words to generate
        words_tsv_path: Path to words.tsv file
        seed: Random seed for reproducibility

    Returns:
        Text with random words separated by spaces
    """
    if seed is not None:
        random.seed(seed)

    # Load word frequency
    words, frequencies = load_word_frequency(words_tsv_path)

    if not words:
        raise ValueError("No words found in word frequency file")

    # Generate random words using frequency as weights
    random_words = random.choices(words, weights=frequencies, k=num_words)

    # Join with spaces
    random_text = ' '.join(random_words)

    return random_text


# =============================================================================
# JSONL File Operations
# =============================================================================

def append_to_jsonl(filepath, data: dict):
    """
    Append single result to JSONL file (append-only, never corrupts)

    Args:
        filepath: Path to JSONL file (str or Path)
        data: Single result dictionary
    """
    from pathlib import Path
    filepath = Path(filepath)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()  # Force immediate write to disk


def load_from_jsonl(filepath) -> List[dict]:
    """
    Load all results from JSONL file

    Args:
        filepath: Path to JSONL file (str or Path)

    Returns:
        List of result dictionaries. Empty list if file doesn't exist.

    Note:
        Lines that fail to parse are logged and skipped.
    """
    from pathlib import Path
    filepath = Path(filepath)
    results = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Note: logger is not available in core.py, so we use print
                        print(f"Warning: Failed to parse JSONL line: {e}")
                        continue
    return results


# =============================================================================
# C++ Code Execution (for CodeElo integration)
# =============================================================================

# Import CodeElo's extract_code_blocks function
import sys
sys.path.append('datasets/CodeElo')
try:
    from main import extract_code_blocks
except ImportError:
    # Fallback: define extract_code_blocks if CodeElo is not available
    def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks from text using regex

        Args:
            text: Text containing code blocks marked with ```

        Returns:
            List of (language, code) tuples
        """
        import re
        pattern = r'```(\w*)\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches


def normalize_output(output: str) -> str:
    """
    標準化輸出用於比對（去除每行尾部空白）

    Args:
        output: 原始輸出字符串

    Returns:
        標準化後的輸出字符串

    Examples:
        >>> normalize_output("hello   \\n")
        'hello'
        >>> normalize_output("  line1  \\n  line2  ")
        'line1\\nline2'
    """
    return '\n'.join(line.strip() for line in output.strip().split('\n'))


def _detect_cpp_compiler() -> str:
    """
    檢測並返回最佳的 C++ 編譯器路徑

    優先順序：
    1. GCC (g++-14, g++-13, etc.) - 支援 bits/stdc++.h
    2. 系統默認 g++ (可能是 Clang)

    Returns:
        編譯器路徑
    """
    import shutil
    import subprocess

    # Try to find GCC (真正的 GCC, 不是 Clang 別名)
    # 檢查 Homebrew GCC
    gcc_candidates = [
        '/opt/homebrew/bin/g++-14',
        '/opt/homebrew/bin/g++-13',
        '/opt/homebrew/bin/g++-12',
        '/usr/local/bin/g++-14',
        '/usr/local/bin/g++-13',
        '/usr/local/bin/g++-12',
    ]

    for gcc_path in gcc_candidates:
        if shutil.which(gcc_path):
            # 驗證是否真的是 GCC (不是 Clang 別名)
            try:
                result = subprocess.run(
                    [gcc_path, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if 'GCC' in result.stdout or 'gcc' in result.stdout.lower():
                    return gcc_path
            except:
                continue

    # Fallback to system g++ (可能是 Clang)
    if shutil.which('g++'):
        return 'g++'

    return None


def compile_and_execute_cpp(code: str, test_input: str, timeout: int = 2) -> Dict:
    """
    編譯並執行 C++ 代碼

    Args:
        code: C++ 源代碼
        test_input: 測試輸入（通過 stdin 傳入）
        timeout: 執行超時（秒）

    Returns:
        {
            'status': 'AC' | 'WA' | 'CE' | 'RTE' | 'TLE',
            'output': str,
            'error': str (if any),
            'compile_error': str (if CE)
        }

    Status codes:
        - AC: Accepted (成功執行)
        - WA: Wrong Answer (輸出不匹配，需在更高層判定)
        - CE: Compilation Error (編譯錯誤)
        - RTE: Runtime Error (運行時錯誤)
        - TLE: Time Limit Exceeded (超時)
    """
    import subprocess
    import tempfile
    import shutil
    from pathlib import Path

    # Detect best C++ compiler
    compiler = _detect_cpp_compiler()
    if not compiler:
        return {
            'status': 'CE',
            'output': '',
            'error': 'g++ compiler not found',
            'compile_error': 'g++ compiler not found. Please install: brew install gcc'
        }

    # Create temporary directory for compilation and execution
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        source_file = tmpdir_path / 'solution.cpp'
        executable_file = tmpdir_path / 'solution'

        # Write source code to file
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(code)

        # Compile C++ code
        try:
            compile_result = subprocess.run(
                [compiler, '-o', str(executable_file), str(source_file), '-std=c++17'],
                capture_output=True,
                text=True,
                timeout=10  # Compilation timeout
            )

            if compile_result.returncode != 0:
                return {
                    'status': 'CE',
                    'output': '',
                    'error': compile_result.stderr,
                    'compile_error': compile_result.stderr
                }
        except subprocess.TimeoutExpired:
            return {
                'status': 'CE',
                'output': '',
                'error': 'Compilation timeout',
                'compile_error': 'Compilation timeout (> 10s)'
            }
        except Exception as e:
            return {
                'status': 'CE',
                'output': '',
                'error': str(e),
                'compile_error': str(e)
            }

        # Execute compiled program
        try:
            execute_result = subprocess.run(
                [str(executable_file)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if execute_result.returncode != 0:
                return {
                    'status': 'RTE',
                    'output': execute_result.stdout,
                    'error': execute_result.stderr
                }

            return {
                'status': 'AC',
                'output': execute_result.stdout,
                'error': ''
            }

        except subprocess.TimeoutExpired:
            return {
                'status': 'TLE',
                'output': '',
                'error': f'Time limit exceeded (> {timeout}s)'
            }
        except Exception as e:
            return {
                'status': 'RTE',
                'output': '',
                'error': str(e)
            }
