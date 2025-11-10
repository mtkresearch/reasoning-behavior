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
