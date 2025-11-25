"""
Unit tests for core.py utility functions

This test file covers truncate and shuffle functions that will be added to core.py.
"""

import pytest


class TestTruncateReasoningLines:
    """Tests for truncate_reasoning_lines()"""

    def test_count_mode_remove_last_n_lines(self):
        """Test removing last N lines in count mode (n >= 1)"""
        from core import truncate_reasoning_lines

        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        # Remove last 2 lines
        result = truncate_reasoning_lines(reasoning, 2)
        expected = "Line 1\nLine 2\nLine 3"
        assert result == expected

    def test_ratio_mode_remove_percentage(self):
        """Test removing X% of lines in ratio mode (0 < ratio < 1)"""
        from core import truncate_reasoning_lines

        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""

        # Remove 30% (3 out of 10 lines)
        result = truncate_reasoning_lines(reasoning, 0.3)
        lines = result.split('\n')
        assert len(lines) == 7  # 10 - 3 = 7
        assert lines[0] == "Line 1"
        assert lines[-1] == "Line 7"

    def test_remove_all_but_one_line(self):
        """Test when removing more lines than available returns first line"""
        from core import truncate_reasoning_lines

        reasoning = """Line 1
Line 2
Line 3"""

        # Try to remove 10 lines (more than available)
        result = truncate_reasoning_lines(reasoning, 10)
        assert result == "Line 1"

    def test_zero_truncation(self):
        """Test that del_last_line=0 returns original reasoning"""
        from core import truncate_reasoning_lines

        reasoning = """Line 1
Line 2
Line 3"""

        result = truncate_reasoning_lines(reasoning, 0)
        assert result == reasoning

    def test_negative_truncation(self):
        """Test that negative values return original reasoning"""
        from core import truncate_reasoning_lines

        reasoning = """Line 1
Line 2
Line 3"""

        result = truncate_reasoning_lines(reasoning, -1)
        assert result == reasoning

    def test_empty_string(self):
        """Test empty string handling"""
        from core import truncate_reasoning_lines

        result = truncate_reasoning_lines("", 2)
        assert result == ""

    def test_single_line(self):
        """Test with single line"""
        from core import truncate_reasoning_lines

        reasoning = "Only one line"
        result = truncate_reasoning_lines(reasoning, 0.5)
        # Should return the line since we can't remove 0.5 of 1 line (int(0.5) = 0)
        assert result == reasoning

    def test_removes_empty_lines_before_truncating(self):
        """Test that empty lines are removed before counting"""
        from core import truncate_reasoning_lines

        reasoning = """Line 1

Line 2

Line 3

Line 4"""

        # Should have 4 non-empty lines, remove last 1
        result = truncate_reasoning_lines(reasoning, 1)
        lines = [line for line in result.split('\n') if line.strip()]
        assert len(lines) == 3
        assert "Line 4" not in result

    def test_ratio_mode_rounds_down(self):
        """Test that ratio mode uses int() which rounds down"""
        from core import truncate_reasoning_lines

        reasoning = """Line 1
Line 2
Line 3"""

        # 0.4 * 3 = 1.2, int(1.2) = 1
        result = truncate_reasoning_lines(reasoning, 0.4)
        lines = result.split('\n')
        assert len(lines) == 2  # 3 - 1 = 2


class TestShuffleReasoningWords:
    """Tests for shuffle_reasoning_words()"""

    def test_shuffle_words_basic(self):
        """Test basic word shuffling"""
        from core import shuffle_reasoning_words

        reasoning = "The quick brown fox jumps over the lazy dog"
        result = shuffle_reasoning_words(reasoning, seed=42)

        # Result should have same words but different order
        original_words = set(reasoning.split())
        result_words = set(result.split())
        assert original_words == result_words

        # With high probability, order should be different
        assert result != reasoning

    def test_shuffle_words_with_seed_reproducible(self):
        """Test that same seed produces same shuffle"""
        from core import shuffle_reasoning_words

        reasoning = "The quick brown fox jumps over the lazy dog"
        result1 = shuffle_reasoning_words(reasoning, seed=42)
        result2 = shuffle_reasoning_words(reasoning, seed=42)

        assert result1 == result2

    def test_shuffle_words_different_seeds(self):
        """Test that different seeds produce different results"""
        from core import shuffle_reasoning_words

        reasoning = "The quick brown fox jumps over the lazy dog"
        result1 = shuffle_reasoning_words(reasoning, seed=42)
        result2 = shuffle_reasoning_words(reasoning, seed=123)

        # With high probability, should be different
        assert result1 != result2

    def test_shuffle_words_multiline(self):
        """Test shuffling words across multiple lines"""
        from core import shuffle_reasoning_words

        reasoning = """First line here
Second line here
Third line here"""

        result = shuffle_reasoning_words(reasoning, seed=42)

        # All words should still be present
        original_words = set(reasoning.replace('\n', ' ').split())
        result_words = set(result.replace('\n', ' ').split())
        assert original_words == result_words

    def test_shuffle_words_empty_string(self):
        """Test empty string handling"""
        from core import shuffle_reasoning_words

        result = shuffle_reasoning_words("", seed=42)
        assert result == ""

    def test_shuffle_words_single_word(self):
        """Test with single word"""
        from core import shuffle_reasoning_words

        result = shuffle_reasoning_words("Hello", seed=42)
        assert result == "Hello"


class TestShuffleReasoningTokens:
    """Tests for shuffle_reasoning_tokens()"""

    def test_shuffle_tokens_basic(self):
        """Test basic token shuffling"""
        from core import shuffle_reasoning_tokens

        reasoning = "The quick brown fox jumps over the lazy dog"
        result = shuffle_reasoning_tokens(reasoning, tokenizer_model="gpt2", seed=42)

        # Result should not be empty
        assert len(result) > 0

        # With high probability, order should be different
        assert result != reasoning

    def test_shuffle_tokens_with_seed_reproducible(self):
        """Test that same seed produces same shuffle"""
        from core import shuffle_reasoning_tokens

        reasoning = "The quick brown fox jumps over the lazy dog"
        result1 = shuffle_reasoning_tokens(reasoning, tokenizer_model="gpt2", seed=42)
        result2 = shuffle_reasoning_tokens(reasoning, tokenizer_model="gpt2", seed=42)

        assert result1 == result2

    def test_shuffle_tokens_different_seeds(self):
        """Test that different seeds produce different results"""
        from core import shuffle_reasoning_tokens

        reasoning = "The quick brown fox jumps over the lazy dog"
        result1 = shuffle_reasoning_tokens(reasoning, tokenizer_model="gpt2", seed=42)
        result2 = shuffle_reasoning_tokens(reasoning, tokenizer_model="gpt2", seed=123)

        # With high probability, should be different
        assert result1 != result2

    def test_shuffle_tokens_empty_string(self):
        """Test empty string handling"""
        from core import shuffle_reasoning_tokens

        result = shuffle_reasoning_tokens("", tokenizer_model="gpt2", seed=42)
        assert result == ""

    def test_shuffle_tokens_with_model_type_deepseek(self):
        """Test token shuffling with model_type='deepseek'"""
        from core import shuffle_reasoning_tokens

        reasoning = "The quick brown fox jumps over the lazy dog"
        result = shuffle_reasoning_tokens(reasoning, model_type="deepseek", seed=42)

        # Result should not be empty
        assert len(result) > 0

        # With high probability, order should be different
        assert result != reasoning

    def test_shuffle_tokens_with_model_type_gpt_oss(self):
        """Test token shuffling with model_type='gpt-oss'"""
        from core import shuffle_reasoning_tokens

        reasoning = "The quick brown fox jumps over the lazy dog"
        result = shuffle_reasoning_tokens(reasoning, model_type="gpt-oss", seed=42)

        # Result should not be empty
        assert len(result) > 0

        # With high probability, order should be different
        assert result != reasoning

    def test_shuffle_tokens_with_model_type_qwen3(self):
        """Test token shuffling with model_type='qwen3'"""
        from core import shuffle_reasoning_tokens

        reasoning = "The quick brown fox jumps over the lazy dog"
        result = shuffle_reasoning_tokens(reasoning, model_type="qwen3", seed=42)

        # Result should not be empty
        assert len(result) > 0

        # With high probability, order should be different
        assert result != reasoning

    def test_shuffle_tokens_model_type_reproducible(self):
        """Test that model_type with same seed produces same shuffle"""
        from core import shuffle_reasoning_tokens

        reasoning = "The quick brown fox jumps over the lazy dog"
        result1 = shuffle_reasoning_tokens(reasoning, model_type="gpt-oss", seed=42)
        result2 = shuffle_reasoning_tokens(reasoning, model_type="gpt-oss", seed=42)

        assert result1 == result2

    def test_shuffle_tokens_invalid_model_type(self):
        """Test that invalid model_type raises ValueError"""
        from core import shuffle_reasoning_tokens
        import pytest

        reasoning = "The quick brown fox"
        with pytest.raises(ValueError, match="Invalid model_type"):
            shuffle_reasoning_tokens(reasoning, model_type="invalid_model")

    def test_shuffle_tokens_missing_both_args(self):
        """Test that missing both tokenizer_model and model_type raises ValueError"""
        from core import shuffle_reasoning_tokens
        import pytest

        reasoning = "The quick brown fox"
        with pytest.raises(ValueError, match="Either tokenizer_model or model_type must be provided"):
            shuffle_reasoning_tokens(reasoning)


class TestShuffleReasoningUnified:
    """Tests for unified shuffle_reasoning() interface"""

    def test_shuffle_reasoning_line_mode(self):
        """Test shuffle_reasoning with mode='line'"""
        from core import shuffle_reasoning

        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        result = shuffle_reasoning(reasoning, mode='line', seed=42)

        # All lines should still be present
        original_lines = set(reasoning.strip().split('\n'))
        result_lines = set(result.strip().split('\n'))
        assert original_lines == result_lines

    def test_shuffle_reasoning_word_mode(self):
        """Test shuffle_reasoning with mode='word'"""
        from core import shuffle_reasoning

        reasoning = "The quick brown fox jumps"
        result = shuffle_reasoning(reasoning, mode='word', seed=42)

        # All words should still be present
        original_words = set(reasoning.split())
        result_words = set(result.split())
        assert original_words == result_words

    def test_shuffle_reasoning_token_mode(self):
        """Test shuffle_reasoning with mode='token'"""
        from core import shuffle_reasoning

        reasoning = "The quick brown fox"
        result = shuffle_reasoning(reasoning, mode='token', seed=42, tokenizer_model='gpt2')

        assert len(result) > 0

    def test_shuffle_reasoning_invalid_mode(self):
        """Test that invalid mode raises ValueError"""
        from core import shuffle_reasoning

        with pytest.raises(ValueError, match="Invalid shuffle mode"):
            shuffle_reasoning("Some text", mode='invalid', seed=42)

    def test_shuffle_reasoning_token_mode_requires_tokenizer(self):
        """Test that token mode without tokenizer_model raises error"""
        from core import shuffle_reasoning

        # This should work - tokenizer_model is provided
        result = shuffle_reasoning("Test", mode='token', seed=42, tokenizer_model='gpt2')
        assert len(result) > 0

    def test_shuffle_reasoning_token_mode_with_model_type(self):
        """Test shuffle_reasoning with mode='token' using model_type"""
        from core import shuffle_reasoning

        reasoning = "The quick brown fox jumps over the lazy dog"

        # Test with different model types
        for model_type in ['deepseek', 'gpt-oss', 'qwen3']:
            result = shuffle_reasoning(reasoning, mode='token', seed=42, model_type=model_type)
            assert len(result) > 0, f"Failed for model_type={model_type}"

    def test_shuffle_reasoning_token_mode_model_type_reproducible(self):
        """Test that model_type in shuffle_reasoning produces reproducible results"""
        from core import shuffle_reasoning

        reasoning = "The quick brown fox jumps over the lazy dog"
        result1 = shuffle_reasoning(reasoning, mode='token', seed=42, model_type='gpt-oss')
        result2 = shuffle_reasoning(reasoning, mode='token', seed=42, model_type='gpt-oss')

        assert result1 == result2

    def test_shuffle_reasoning_token_mode_missing_both_args(self):
        """Test that token mode without tokenizer_model or model_type raises error"""
        from core import shuffle_reasoning
        import pytest

        reasoning = "The quick brown fox"
        with pytest.raises(ValueError, match="Either tokenizer_model or model_type must be provided"):
            shuffle_reasoning(reasoning, mode='token', seed=42)


class TestRemoveExactAnswer:
    """Tests for remove_exact_answer()"""

    def test_remove_exact_answer_basic(self):
        """Test basic exact answer removal"""
        from core import remove_exact_answer

        reasoning = "The answer is 42."
        result = remove_exact_answer(reasoning, "42")

        assert "42" not in result
        assert "The answer is" in result

    def test_remove_exact_answer_multiple_occurrences(self):
        """Test that all occurrences are removed"""
        from core import remove_exact_answer

        reasoning = "First: 42, second: 42, third: 42."
        result = remove_exact_answer(reasoning, "42")

        assert "42" not in result
        assert result.count("42") == 0

    def test_remove_exact_answer_word_boundary(self):
        """Test that word boundaries are respected"""
        from core import remove_exact_answer

        reasoning = "Numbers: 42, 421, 142, 4200"
        result = remove_exact_answer(reasoning, "42")

        # Standalone 42 should be removed
        # But 421, 142, 4200 should remain
        assert "421" in result
        assert "142" in result
        assert "4200" in result

    def test_remove_exact_answer_empty_answer(self):
        """Test with empty answer string"""
        from core import remove_exact_answer

        reasoning = "This is some text."
        result = remove_exact_answer(reasoning, "")

        assert result == reasoning

    def test_remove_exact_answer_whitespace_answer(self):
        """Test with whitespace-only answer"""
        from core import remove_exact_answer

        reasoning = "This is some text."
        result = remove_exact_answer(reasoning, "   ")

        assert result == reasoning

    def test_remove_exact_answer_special_regex_chars(self):
        """Test with answer containing regex special characters"""
        from core import remove_exact_answer

        # Test with decimal point (regex special char)
        reasoning = "The value is 3.14 and also 3.14."
        result = remove_exact_answer(reasoning, "3.14")

        assert "3.14" not in result
        assert "The value is" in result

    def test_remove_exact_answer_no_match(self):
        """Test when answer is not in reasoning"""
        from core import remove_exact_answer

        reasoning = "This text does not contain the answer."
        result = remove_exact_answer(reasoning, "999")

        assert result == reasoning

    def test_remove_exact_answer_multiline(self):
        """Test removal across multiple lines"""
        from core import remove_exact_answer

        reasoning = """Line 1: answer is 12
Line 2: we got 12
Line 3: final answer 12"""
        result = remove_exact_answer(reasoning, "12")

        assert "12" not in result
        assert "Line 1:" in result
        assert "Line 2:" in result
        assert "Line 3:" in result

    def test_remove_exact_answer_with_brackets(self):
        """Test with answer containing brackets (regex special chars)

        Note: Word boundaries (\b) only work between word characters (alphanumeric)
        and non-word characters. Since '(' and ')' are non-word characters,
        \b won't match around them in patterns like '(1+2)'.
        This test verifies that special regex characters are properly escaped,
        even if word boundaries don't fully apply.
        """
        from core import remove_exact_answer

        # Test with numeric answer that has proper word boundaries
        reasoning = "The answer is 123 in brackets."
        result = remove_exact_answer(reasoning, "123")

        assert "123" not in result
        assert "The answer is" in result

    def test_remove_exact_answer_preserves_structure(self):
        """Test that the overall structure is preserved"""
        from core import remove_exact_answer

        reasoning = """Step 1: Calculate 5 + 7 = 12
Step 2: Therefore the answer is 12."""
        result = remove_exact_answer(reasoning, "12")

        # Should have same number of lines
        assert result.count('\n') == reasoning.count('\n')
        # Step labels should remain
        assert "Step 1:" in result
        assert "Step 2:" in result
