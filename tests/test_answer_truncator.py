"""
Tests for AnswerTruncator class in run_attn_visual.py

Following TDD Red-Green-Refactor cycle for Phase 2
"""

import pytest


class TestAnswerTruncator:
    """Test suite for AnswerTruncator class"""

    def test_find_answer_position_simple(self):
        """Test finding answer position in simple text"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        text = "The answer is 42 and that's correct."
        ground_truth = "42"

        position = truncator.find_answer_position(text, ground_truth)

        assert position == 14  # Position of '42'

    def test_find_answer_position_not_found(self):
        """Test when answer is not found in text"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        text = "The answer is forty-two."
        ground_truth = "42"

        position = truncator.find_answer_position(text, ground_truth)

        assert position is None

    def test_find_answer_position_word_boundary(self):
        """Test that word boundary is respected"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        # Should not match '4' in '42'
        text = "The answer is 42"
        ground_truth = "4"

        position = truncator.find_answer_position(text, ground_truth)

        # Should not find '4' as a separate word
        assert position is None

    def test_find_answer_position_multiple_occurrences(self):
        """Test finding first occurrence when answer appears multiple times"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        text = "First 42 then another 42 appears."
        ground_truth = "42"

        position = truncator.find_answer_position(text, ground_truth)

        assert position == 6  # Position of first '42'

    def test_truncate_at_answer(self):
        """Test truncating text at specified position"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        text = "The answer is 42 and that's correct."
        position = 14

        truncated = truncator.truncate_at_answer(text, position)

        assert truncated == "The answer is "

    def test_truncate_at_beginning(self):
        """Test truncating at position 0"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        text = "42 is the answer"
        position = 0

        truncated = truncator.truncate_at_answer(text, position)

        assert truncated == ""

    def test_process_complete_workflow(self):
        """Test complete truncation workflow"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        generated_answer = "Thus, the answer is 42"
        ground_truth = "42"

        result = truncator.process(generated_answer, ground_truth)

        assert result == "Thus, the answer is "

    def test_process_answer_not_found(self):
        """Test process when answer is not found - returns full text"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        generated_answer = "Thus, the answer is forty-two"
        ground_truth = "42"

        result = truncator.process(generated_answer, ground_truth)

        assert result == generated_answer  # Returns full text when not found

    def test_process_with_special_characters(self):
        """Test processing with special characters in answer"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        generated_answer = "The result is $100.50 dollars"
        ground_truth = "$100.50"

        result = truncator.process(generated_answer, ground_truth)

        # Should properly escape regex special characters
        assert result == "The result is "

    def test_process_with_whitespace(self):
        """Test processing with whitespace handling"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        generated_answer = "Answer:   42   with spaces"
        ground_truth = "42"

        result = truncator.process(generated_answer, ground_truth)

        assert result == "Answer:   "

    def test_find_answer_position_case_sensitive(self):
        """Test that search is case-sensitive"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        text = "The answer is ABC"
        ground_truth = "abc"

        position = truncator.find_answer_position(text, ground_truth)

        # Should not match due to case difference
        assert position is None

    def test_process_empty_answer(self):
        """Test processing with empty ground truth"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        generated_answer = "Some text here"
        ground_truth = ""

        # Should handle empty ground truth gracefully
        result = truncator.process(generated_answer, ground_truth)

        # Implementation detail: empty pattern may match at start or return full text
        # Either behavior is acceptable for edge case
        assert isinstance(result, str)

    def test_find_answer_complex_number(self):
        """Test finding complex numerical answer"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        text = "The final answer is 3.14159 approximately"
        ground_truth = "3.14159"

        position = truncator.find_answer_position(text, ground_truth)

        assert position == 20

    def test_find_answer_with_parentheses(self):
        """Test finding answer with parentheses (special regex chars)"""
        from run_attn_visual import AnswerTruncator

        truncator = AnswerTruncator()

        text = "The answer is (x+y) based on formula"
        ground_truth = "(x+y)"

        position = truncator.find_answer_position(text, ground_truth)

        assert position == 14  # Should properly escape parentheses
