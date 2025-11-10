#!/usr/bin/env python3
"""
Unit tests for masking functions

This test file covers all masking functions that will be extracted to core.py:
- mask_numbers_in_reasoning()
- mask_answer_only_in_reasoning()
- mask_numbers_in_lines_with_answer()
- mask_numbers_in_nlines_with_answer()
- mask_alphabet_in_reasoning()
- mask_alphabet_and_answer_in_reasoning()
- mask_numbers_advance()
"""

import pytest
from core import (
    mask_numbers_in_reasoning,
    mask_answer_only_in_reasoning,
    mask_numbers_in_lines_with_answer,
    mask_numbers_in_nlines_with_answer,
    mask_alphabet_in_reasoning,
    mask_alphabet_and_answer_in_reasoning,
    mask_numbers_advance
)


class TestMaskNumbersInReasoning:
    """Tests for mask_numbers_in_reasoning()"""

    def test_basic_number_masking(self):
        """Test basic single-digit masking"""
        reasoning = "The answer is 5"
        result = mask_numbers_in_reasoning(reasoning)
        assert result == "The answer is █"

    def test_multi_digit_masking(self):
        """Test multi-digit number masking"""
        reasoning = "The answer is 42"
        result = mask_numbers_in_reasoning(reasoning)
        assert result == "The answer is ██"

    def test_mixed_text_and_numbers(self):
        """Test masking in mixed text with multiple numbers"""
        reasoning = "Calculate 1 + 2 = 3 and then multiply by 10"
        result = mask_numbers_in_reasoning(reasoning)
        assert result == "Calculate █ + █ = █ and then multiply by ██"

    def test_empty_string(self):
        """Test empty string handling"""
        reasoning = ""
        result = mask_numbers_in_reasoning(reasoning)
        assert result == ""

    def test_custom_mask_char(self):
        """Test with custom mask character"""
        reasoning = "The value is 123"
        result = mask_numbers_in_reasoning(reasoning, mask_char='*')
        assert result == "The value is ***"

    def test_no_numbers(self):
        """Test text with no numbers"""
        reasoning = "This text has no digits"
        result = mask_numbers_in_reasoning(reasoning)
        assert result == "This text has no digits"

    def test_numbers_at_boundaries(self):
        """Test numbers at start and end of text"""
        reasoning = "5 is the answer 10"
        result = mask_numbers_in_reasoning(reasoning)
        assert result == "█ is the answer ██"

    def test_consecutive_numbers(self):
        """Test consecutive numbers like years or large numbers"""
        reasoning = "In year 2025, the value was 1000000"
        result = mask_numbers_in_reasoning(reasoning)
        assert result == "In year ████, the value was ███████"


class TestMaskAnswerOnlyInReasoning:
    """Tests for mask_answer_only_in_reasoning()"""

    def test_mask_single_answer_occurrence(self):
        """Test masking single answer occurrence"""
        reasoning = "The final answer is 42"
        answer = "42"
        result = mask_answer_only_in_reasoning(reasoning, answer)
        assert result == "The final answer is ██"

    def test_mask_multiple_answer_occurrences(self):
        """Test masking multiple answer occurrences"""
        reasoning = "First we see 42, then calculate 42 + 10 = 52. The answer is 42."
        answer = "42"
        result = mask_answer_only_in_reasoning(reasoning, answer)
        assert result == "First we see ██, then calculate ██ + 10 = 52. The answer is ██."

    def test_answer_not_in_reasoning(self):
        """Test when answer doesn't exist in reasoning"""
        reasoning = "The calculation gives us 100"
        answer = "99"
        result = mask_answer_only_in_reasoning(reasoning, answer)
        assert result == "The calculation gives us 100"

    def test_word_boundary_handling(self):
        """Test word boundary handling to avoid partial matches"""
        reasoning = "We have 123 and 23, and the result is 3"
        answer = "23"
        result = mask_answer_only_in_reasoning(reasoning, answer)
        # Should mask "23" but not the "23" in "123"
        assert result == "We have 123 and ██, and the result is 3"

    def test_answer_with_special_chars(self):
        """Test answer containing special regex characters"""
        # Note: Word boundaries (\b) don't work with parentheses
        # This test documents the current limitation
        reasoning = "The pattern is (1+2) which equals (1+2)"
        answer = "(1+2)"
        result = mask_answer_only_in_reasoning(reasoning, answer)
        # Current implementation doesn't match due to \b word boundary limitation
        assert result == "The pattern is (1+2) which equals (1+2)"

    def test_answer_with_dots(self):
        """Test answer containing dots (special regex char)"""
        reasoning = "The value is 3.14 and repeats 3.14"
        answer = "3.14"
        result = mask_answer_only_in_reasoning(reasoning, answer)
        # Dots are special chars but work with word boundaries
        assert result == "The value is ████ and repeats ████"

    def test_custom_mask_char(self):
        """Test with custom mask character"""
        reasoning = "The answer is 42"
        answer = "42"
        result = mask_answer_only_in_reasoning(reasoning, answer, mask_char='*')
        assert result == "The answer is **"

    def test_answer_with_whitespace(self):
        """Test answer with leading/trailing whitespace"""
        reasoning = "The value is 42 exactly"
        answer = " 42 "  # with spaces
        result = mask_answer_only_in_reasoning(reasoning, answer)
        # Should still match after cleaning
        assert result == "The value is ██ exactly"


class TestMaskNumbersInLinesWithAnswer:
    """Tests for mask_numbers_in_lines_with_answer()"""

    def test_single_line_with_answer(self):
        """Test masking numbers in a single line containing the answer"""
        reasoning = "The answer is 42"
        answer = "42"
        result = mask_numbers_in_lines_with_answer(reasoning, answer)
        assert result == "The answer is ██"

    def test_multiple_lines_with_answer(self):
        """Test masking only lines that contain the answer"""
        reasoning = "Line 1 has number 10\nLine 2 has answer 42\nLine 3 has number 20"
        answer = "42"
        result = mask_numbers_in_lines_with_answer(reasoning, answer)
        assert result == "Line 1 has number 10\nLine █ has answer ██\nLine 3 has number 20"

    def test_multiple_answer_lines(self):
        """Test masking multiple lines that contain the answer"""
        reasoning = "First 42 is here\nNo answer here with 10\nSecond 42 is here\n100 is not the answer"
        answer = "42"
        result = mask_numbers_in_lines_with_answer(reasoning, answer)
        assert result == "First ██ is here\nNo answer here with 10\nSecond ██ is here\n100 is not the answer"

    def test_no_answer_in_reasoning(self):
        """Test when answer doesn't appear in any line"""
        reasoning = "Line 1 has 10\nLine 2 has 20\nLine 3 has 30"
        answer = "99"
        result = mask_numbers_in_lines_with_answer(reasoning, answer)
        assert result == "Line 1 has 10\nLine 2 has 20\nLine 3 has 30"

    def test_word_boundary_protection(self):
        """Test that word boundaries prevent partial matches"""
        reasoning = "We have 123 here\nWe have 23 here\nWe have 3 here"
        answer = "23"
        result = mask_numbers_in_lines_with_answer(reasoning, answer)
        # Only the line with standalone "23" should be masked
        assert result == "We have 123 here\nWe have ██ here\nWe have 3 here"

    def test_custom_mask_char(self):
        """Test with custom mask character"""
        reasoning = "Line 1 has 10\nLine 2 has answer 42"
        answer = "42"
        result = mask_numbers_in_lines_with_answer(reasoning, answer, mask_char='*')
        assert result == "Line 1 has 10\nLine * has answer **"

    def test_empty_string(self):
        """Test empty string handling"""
        reasoning = ""
        answer = "42"
        result = mask_numbers_in_lines_with_answer(reasoning, answer)
        assert result == ""


class TestMaskNumbersInNLinesWithAnswer:
    """Tests for mask_numbers_in_nlines_with_answer()"""

    def test_mask_answer_line_and_one_previous(self):
        """Test masking answer line and 1 previous non-empty line (default)"""
        reasoning = "Line 1 has 10\nLine 2 has 20\nLine 3 has answer 42"
        answer = "42"
        result = mask_numbers_in_nlines_with_answer(reasoning, answer, n=1)
        assert result == "Line 1 has 10\nLine █ has ██\nLine █ has answer ██"

    def test_mask_answer_line_and_two_previous(self):
        """Test masking answer line and 2 previous non-empty lines"""
        reasoning = "Line 1 has 10\nLine 2 has 20\nLine 3 has 30\nLine 4 has answer 42"
        answer = "42"
        result = mask_numbers_in_nlines_with_answer(reasoning, answer, n=2)
        assert result == "Line 1 has 10\nLine █ has ██\nLine █ has ██\nLine █ has answer ██"

    def test_skip_empty_lines_when_counting(self):
        """Test that empty lines are skipped when counting previous lines"""
        reasoning = "Line 1 has 10\n\n\nLine 2 has 20\nLine 3 has answer 42"
        answer = "42"
        result = mask_numbers_in_nlines_with_answer(reasoning, answer, n=1)
        # Should mask Line 2 and Line 3, skipping empty lines
        assert result == "Line 1 has 10\n\n\nLine █ has ██\nLine █ has answer ██"

    def test_insufficient_previous_lines(self):
        """Test when there are fewer previous lines than requested"""
        reasoning = "Line 1 has answer 42"
        answer = "42"
        result = mask_numbers_in_nlines_with_answer(reasoning, answer, n=5)
        # Should only mask the answer line (no previous lines available)
        assert result == "Line █ has answer ██"

    def test_multiple_answer_occurrences(self):
        """Test masking around multiple lines containing the answer"""
        reasoning = "Line 1 has 10\nLine 2 has 42\nLine 3 has 30\nLine 4 has 42\nLine 5 has 50"
        answer = "42"
        result = mask_numbers_in_nlines_with_answer(reasoning, answer, n=1)
        # Should mask Lines 1-2 (around first 42) and Lines 3-4 (around second 42)
        assert result == "Line █ has ██\nLine █ has ██\nLine █ has ██\nLine █ has ██\nLine 5 has 50"

    def test_no_answer_in_reasoning(self):
        """Test when answer doesn't appear in reasoning"""
        reasoning = "Line 1 has 10\nLine 2 has 20\nLine 3 has 30"
        answer = "99"
        result = mask_numbers_in_nlines_with_answer(reasoning, answer, n=1)
        assert result == "Line 1 has 10\nLine 2 has 20\nLine 3 has 30"

    def test_custom_mask_char(self):
        """Test with custom mask character"""
        reasoning = "Line 1 has 10\nLine 2 has answer 42"
        answer = "42"
        result = mask_numbers_in_nlines_with_answer(reasoning, answer, n=1, mask_char='*')
        assert result == "Line * has **\nLine * has answer **"


class TestMaskAlphabetInReasoning:
    """Tests for mask_alphabet_in_reasoning()"""

    def test_basic_alphabet_masking(self):
        """Test basic alphabetic character masking"""
        reasoning = "The answer is ABC"
        result = mask_alphabet_in_reasoning(reasoning)
        assert result == "███ ██████ ██ ███"

    def test_mixed_case_masking(self):
        """Test masking both uppercase and lowercase"""
        reasoning = "Calculate A + b = C"
        result = mask_alphabet_in_reasoning(reasoning)
        assert result == "█████████ █ + █ = █"

    def test_preserve_numbers(self):
        """Test that numbers are preserved while masking letters"""
        reasoning = "The value is 42 and x equals 10"
        result = mask_alphabet_in_reasoning(reasoning)
        assert result == "███ █████ ██ 42 ███ █ ██████ 10"

    def test_preserve_special_chars(self):
        """Test that special characters are preserved"""
        reasoning = "f(x) = 2x + 3, where x > 5"
        result = mask_alphabet_in_reasoning(reasoning)
        assert result == "█(█) = 2█ + 3, █████ █ > 5"

    def test_empty_string(self):
        """Test empty string handling"""
        reasoning = ""
        result = mask_alphabet_in_reasoning(reasoning)
        assert result == ""

    def test_no_alphabetic_chars(self):
        """Test text with no alphabetic characters"""
        reasoning = "123 + 456 = 579"
        result = mask_alphabet_in_reasoning(reasoning)
        assert result == "123 + 456 = 579"

    def test_custom_mask_char(self):
        """Test with custom mask character"""
        reasoning = "Hello World"
        result = mask_alphabet_in_reasoning(reasoning, mask_char='*')
        assert result == "***** *****"


class TestMaskAlphabetAndAnswerInReasoning:
    """Tests for mask_alphabet_and_answer_in_reasoning()"""

    def test_mask_both_alphabet_and_answer(self):
        """Test masking both alphabetic characters and answer number"""
        reasoning = "The answer is 42"
        answer = "42"
        result = mask_alphabet_and_answer_in_reasoning(reasoning, answer)
        assert result == "███ ██████ ██ ██"

    def test_multiple_answer_occurrences(self):
        """Test masking alphabet and multiple answer occurrences"""
        reasoning = "First 42, then calculate 42 + 10 = 52. Answer is 42."
        answer = "42"
        result = mask_alphabet_and_answer_in_reasoning(reasoning, answer)
        assert result == "█████ ██, ████ █████████ ██ + 10 = 52. ██████ ██ ██."

    def test_word_boundary_for_answer(self):
        """Test that answer masking respects word boundaries"""
        reasoning = "We have 123, 42, and 3"
        answer = "42"
        result = mask_alphabet_and_answer_in_reasoning(reasoning, answer)
        # 42 should be masked, but 123 and 3 should not (only letters masked there)
        assert result == "██ ████ 123, ██, ███ 3"

    def test_answer_not_in_reasoning(self):
        """Test when answer doesn't appear in reasoning"""
        reasoning = "The value is 100"
        answer = "99"
        result = mask_alphabet_and_answer_in_reasoning(reasoning, answer)
        # Only letters should be masked
        assert result == "███ █████ ██ 100"

    def test_answer_with_special_chars(self):
        """Test answer with special characters"""
        reasoning = "The result is 3.14 exactly"
        answer = "3.14"
        result = mask_alphabet_and_answer_in_reasoning(reasoning, answer)
        # Both alphabet and answer should be masked
        assert result == "███ ██████ ██ ████ ███████"

    def test_custom_mask_char(self):
        """Test with custom mask character"""
        reasoning = "Answer is 42"
        answer = "42"
        result = mask_alphabet_and_answer_in_reasoning(reasoning, answer, mask_char='*')
        assert result == "****** ** **"

    def test_empty_string(self):
        """Test empty string handling"""
        reasoning = ""
        answer = "42"
        result = mask_alphabet_and_answer_in_reasoning(reasoning, answer)
        assert result == ""


class TestMaskNumbersAllAdvance:
    """Tests for mask_numbers_advance()"""

    def test_preserve_algebraic_with_letter_before(self):
        """Test preserving algebraic notation with letter before number"""
        reasoning = "We have A12 and B5"
        result = mask_numbers_advance(reasoning)
        assert result == "We have A12 and B5"

    def test_preserve_algebraic_with_letter_after(self):
        """Test preserving algebraic notation with letter after number"""
        reasoning = "Calculate 3x and 2y"
        result = mask_numbers_advance(reasoning)
        assert result == "Calculate 3x and 2y"

    def test_preserve_algebraic_with_underscore(self):
        """Test preserving algebraic notation with underscore"""
        reasoning = "Use x_1 and y_2"
        result = mask_numbers_advance(reasoning)
        assert result == "Use x_1 and y_2"

    def test_preserve_ordinal_numbers(self):
        """Test preserving ordinal numbers like 1st, 2nd"""
        reasoning = "The 1st and 2nd steps"
        result = mask_numbers_advance(reasoning)
        assert result == "The 1st and 2nd steps"

    def test_preserve_inequality_expressions(self):
        """Test preserving numbers in inequality expressions"""
        reasoning = "n < 5 and x > 10 where 1 ≤ y ≤ 20"
        result = mask_numbers_advance(reasoning)
        assert result == "n < 5 and x > 10 where 1 ≤ y ≤ 20"

    def test_mask_multiplication_pattern(self):
        """Test masking multiplication pattern like 3x3"""
        reasoning = "Calculate 3x3 and 10x5"
        result = mask_numbers_advance(reasoning)
        assert result == "Calculate █x█ and ██x█"

    def test_mask_computational_numbers(self):
        """Test masking computational numbers"""
        reasoning = "Add 1 + 2 = 3 and multiply by 10"
        result = mask_numbers_advance(reasoning)
        assert result == "Add █ + █ = █ and multiply by ██"

    def test_mask_function_arguments(self):
        """Test masking function arguments"""
        reasoning = "f(3) and g(42)"
        result = mask_numbers_advance(reasoning)
        assert result == "f(█) and g(██)"

    def test_mask_exponents(self):
        """Test masking exponents"""
        reasoning = "x^2 and y^10"
        result = mask_numbers_advance(reasoning)
        assert result == "x^█ and y^██"

    def test_answer_always_masked(self):
        """Test that answer is ALWAYS masked even when adjacent to letter"""
        reasoning = "We have x42 and A42"
        answer = "42"
        result = mask_numbers_advance(reasoning, answer=answer)
        assert result == "We have x██ and A██"

    def test_mixed_algebraic_and_computational(self):
        """Test mixed algebraic and computational numbers"""
        reasoning = "For A12, calculate 12 + 3 = 15"
        result = mask_numbers_advance(reasoning)
        assert result == "For A12, calculate ██ + █ = ██"

    def test_custom_mask_char(self):
        """Test with custom mask character"""
        reasoning = "Calculate 1 + 2 = 3"
        result = mask_numbers_advance(reasoning, mask_char='*')
        assert result == "Calculate * + * = *"

    def test_empty_string(self):
        """Test empty string handling"""
        reasoning = ""
        result = mask_numbers_advance(reasoning)
        assert result == ""

    def test_no_numbers(self):
        """Test text with no numbers"""
        reasoning = "This has no digits"
        result = mask_numbers_advance(reasoning)
        assert result == "This has no digits"
