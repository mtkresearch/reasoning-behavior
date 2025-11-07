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
from typing import List, Dict
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
