"""Pytest configuration and fixtures for reasoning-behavior tests."""

import pytest


@pytest.fixture
def sample_reasoning_text():
    """Sample reasoning text for testing."""
    return """Let's solve this step by step.
First, we calculate 2 + 2 = 4.
Then, we multiply by 3: 4 × 3 = 12.
Therefore, the answer is 12."""


@pytest.fixture
def sample_reasoning_with_answer():
    """Sample reasoning text with answer marker."""
    return """Let's solve this step by step.
First, we calculate 2 + 2 = 4.
Then, we multiply by 3: 4 × 3 = 12.
Therefore, the answer is \\boxed{12}."""


@pytest.fixture
def sample_question():
    """Sample math question."""
    return "What is (2 + 2) × 3?"


@pytest.fixture
def sample_answer():
    """Sample answer."""
    return "12"


@pytest.fixture
def sample_context():
    """Sample context dictionary for processors."""
    return {
        "question": "What is (2 + 2) × 3?",
        "answer": "12",
        "ground_truth": "12"
    }


@pytest.fixture
def multiline_reasoning():
    """Multi-line reasoning text for testing line operations."""
    return """Step 1: Understand the problem
Step 2: Calculate 5 + 3 = 8
Step 3: Multiply 8 by 2 = 16
Step 4: Subtract 4 from 16 = 12
Step 5: The final answer is 12
Therefore, \\boxed{12}"""
