#!/usr/bin/env python3
"""
Unit tests for preprocessing functions

This test file covers all preprocessing functions that will be extracted to core.py:
- remove_answer_and_after()
- shuffle_lines()
"""

import pytest
from core import remove_answer_and_after, shuffle_lines


class TestRemoveAnswerAndAfter:
    """Tests for remove_answer_and_after()"""

    def test_remove_answer_line_and_after(self):
        """Test removing answer line and all lines after it"""
        reasoning = "Line 1\nLine 2\nAnswer is 42\nLine 4\nLine 5"
        answer = "42"
        result = remove_answer_and_after(reasoning, answer)
        assert result == "Line 1\nLine 2"

    def test_answer_in_first_line(self):
        """Test when answer is in the first line"""
        reasoning = "Answer is 42\nLine 2\nLine 3"
        answer = "42"
        result = remove_answer_and_after(reasoning, answer)
        assert result == ""

    def test_answer_in_last_line(self):
        """Test when answer is in the last line"""
        reasoning = "Line 1\nLine 2\nAnswer is 42"
        answer = "42"
        result = remove_answer_and_after(reasoning, answer)
        assert result == "Line 1\nLine 2"

    def test_answer_not_present(self):
        """Test when answer doesn't exist (keep original)"""
        reasoning = "Line 1\nLine 2\nLine 3"
        answer = "99"
        result = remove_answer_and_after(reasoning, answer)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_multiple_answer_occurrences(self):
        """Test removing only from first occurrence"""
        reasoning = "Line 1 with 42\nLine 2\nLine 3 with 42\nLine 4"
        answer = "42"
        result = remove_answer_and_after(reasoning, answer)
        # Should remove from first occurrence
        assert result == ""

    def test_word_boundary_protection(self):
        """Test word boundary prevents partial matches"""
        reasoning = "Line 1 with 123\nLine 2 with 23\nLine 3"
        answer = "23"
        result = remove_answer_and_after(reasoning, answer)
        # Should remove from line with "23", not "123"
        assert result == "Line 1 with 123"

    def test_empty_string(self):
        """Test empty string handling"""
        reasoning = ""
        answer = "42"
        result = remove_answer_and_after(reasoning, answer)
        assert result == ""

    def test_answer_with_special_chars(self):
        """Test answer with special regex characters"""
        reasoning = "Line 1\nValue is 3.14\nLine 3"
        answer = "3.14"
        result = remove_answer_and_after(reasoning, answer)
        assert result == "Line 1"


class TestShuffleLines:
    """Tests for shuffle_lines()"""

    def test_shuffle_with_seed(self):
        """Test shuffling with fixed seed gives reproducible results"""
        reasoning = "Line 1\nLine 2\nLine 3\nLine 4"
        result1 = shuffle_lines(reasoning, seed=42)
        result2 = shuffle_lines(reasoning, seed=42)
        # Same seed should give same result
        assert result1 == result2

    def test_shuffle_changes_order(self):
        """Test that shuffling actually changes line order"""
        reasoning = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        result = shuffle_lines(reasoning, seed=42)
        # Result should have all lines
        result_lines = set(result.split('\n'))
        original_lines = set(reasoning.split('\n'))
        assert result_lines == original_lines
        # Order should be different (with very high probability)
        assert result != reasoning

    def test_remove_empty_lines(self):
        """Test that empty lines are removed"""
        reasoning = "Line 1\n\n\nLine 2\nLine 3\n\n"
        result = shuffle_lines(reasoning, seed=42)
        result_lines = result.split('\n')
        # Should only have 3 non-empty lines
        assert len(result_lines) == 3
        assert all(line.strip() for line in result_lines)

    def test_single_line(self):
        """Test shuffling with single line (no shuffle needed)"""
        reasoning = "Only one line"
        result = shuffle_lines(reasoning, seed=42)
        assert result == "Only one line"

    def test_empty_string(self):
        """Test empty string handling"""
        reasoning = ""
        result = shuffle_lines(reasoning, seed=42)
        assert result == ""

    def test_two_lines(self):
        """Test shuffling with two lines"""
        reasoning = "Line A\nLine B"
        result = shuffle_lines(reasoning, seed=42)
        # Should have both lines
        result_lines = set(result.split('\n'))
        assert result_lines == {"Line A", "Line B"}

    def test_different_seeds_give_different_results(self):
        """Test that different seeds give different shuffles"""
        reasoning = "L1\nL2\nL3\nL4\nL5"
        result1 = shuffle_lines(reasoning, seed=1)
        result2 = shuffle_lines(reasoning, seed=2)
        # Different seeds should give different results (highly probable)
        # Both should have all lines
        assert set(result1.split('\n')) == set(result2.split('\n'))
