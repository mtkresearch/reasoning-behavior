"""
Unit tests for processors.py

This test file covers all Processor classes:
- Processor (base class)
- MaskProcessor
- TruncateProcessor
- ShuffleProcessor
- InsertProcessor
- QuestionProcessor
- RemoveProcessor
- ReplaceProcessor
"""

import pytest


class TestProcessorBase:
    """Tests for Processor base class interface"""

    def test_processor_base_interface(self):
        """Test that Processor base class has required methods"""
        from processors import Processor

        # Verify it's an abstract base class
        assert hasattr(Processor, 'process')
        assert hasattr(Processor, 'get_metadata')

        # Verify we cannot instantiate it directly
        with pytest.raises(TypeError):
            Processor()


class TestMaskProcessor:
    """Tests for MaskProcessor"""

    def test_mask_processor_number_mode(self, sample_context):
        """Test MaskProcessor with mode='number'"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='number')
        reasoning = "First, we calculate 2 + 2 = 4. Then multiply by 3: 4 × 3 = 12."

        result = processor.process(reasoning, sample_context)

        # All numbers should be masked
        assert '2' not in result
        assert '4' not in result
        assert '3' not in result
        assert '12' not in result
        assert '█' in result

    def test_mask_processor_answer_mode(self, sample_context):
        """Test MaskProcessor with mode='answer'"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='answer')
        reasoning = "First, we calculate 2 + 2 = 4. The answer is 12."
        context = {"answer": "12", "question": "What is the answer?"}

        result = processor.process(reasoning, context)

        # Only answer (12) should be masked
        assert '2' in result
        assert '4' in result
        assert '12' not in result
        assert '█' in result

    def test_mask_processor_custom_mask_char(self, sample_context):
        """Test MaskProcessor with custom mask character"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='number', mask_char='*')
        reasoning = "Calculate 2 + 2 = 4"

        result = processor.process(reasoning, sample_context)

        assert '*' in result
        assert '█' not in result

    def test_mask_processor_get_metadata(self, sample_context):
        """Test MaskProcessor.get_metadata()"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='number', mask_char='█')
        reasoning = "Calculate 2 + 2 = 4"

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'mask'
        assert metadata['mode'] == 'number'
        assert metadata['mask_char'] == '█'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_mask_processor_nlines_mode(self, sample_context):
        """Test MaskProcessor with mode='n-lines' and num_prev_lines parameter"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='n-lines', num_prev_lines=2)
        reasoning = """Line 1: compute 5
Line 2: compute 10
Line 3: compute 15
Line 4: The answer is 12."""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # Last 3 lines (answer line + 2 prev) should have numbers masked
        assert '5' in result  # Line 1 not masked
        assert '10' not in result  # Line 2 masked
        assert '15' not in result  # Line 3 masked
        assert '12' not in result  # Line 4 masked

    def test_mask_processor_alphabet_mode(self, sample_context):
        """Test MaskProcessor with mode='alphabet'"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='alphabet')
        reasoning = "Calculate 2 + 2 = 4"

        result = processor.process(reasoning, sample_context)

        # Letters should be masked, numbers preserved
        assert '2' in result
        assert '4' in result
        assert 'Calculate' not in result
        assert '█' in result


class TestTruncateProcessor:
    """Tests for TruncateProcessor"""

    def test_truncate_processor_answer_and_after(self, sample_context):
        """Test TruncateProcessor with mode='answer_and_after'"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer_and_after')
        reasoning = """Step 1: Calculate something
Step 2: Do more work
The answer is 12.
This should be removed."""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        assert "Step 1" in result
        assert "Step 2" in result
        assert "The answer is 12" not in result
        assert "This should be removed" not in result

    def test_truncate_processor_before_answer(self, sample_context):
        """Test TruncateProcessor with mode='before_answer'"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_answer')
        reasoning = """Step 1: Calculate something
Step 2: Do more work
The answer is 12.
This should be kept."""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        assert "Step 1" not in result
        assert "Step 2" not in result
        assert "The answer is 12" in result
        assert "This should be kept" in result

    def test_truncate_processor_last_n_lines(self, sample_context):
        """Test TruncateProcessor with mode='last_n_lines'"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='last_n_lines', n=2)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" not in result
        assert "Line 5" not in result

    def test_truncate_processor_last_ratio(self, sample_context):
        """Test TruncateProcessor with mode='last_ratio'"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='last_ratio', ratio=0.3)
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

        result = processor.process(reasoning, sample_context)

        lines = [l for l in result.split('\n') if l.strip()]
        # 30% of 10 = 3, so should have 7 lines remaining
        assert len(lines) == 7

    def test_truncate_processor_get_metadata(self, sample_context):
        """Test TruncateProcessor.get_metadata()"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='last_n_lines', n=2)
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'truncate'
        assert metadata['mode'] == 'last_n_lines'
        assert metadata['n'] == 2
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert 'removed_lines' in metadata

    def test_truncate_processor_answer_mode_basic(self, sample_context):
        """Test TruncateProcessor with mode='answer' removes exact answer"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = """Step 1: Calculate something
Step 2: The answer is 12.
Step 3: Therefore 12 is correct."""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # Answer "12" should be removed
        assert "12" not in result
        # Other content should be preserved
        assert "Step 1: Calculate something" in result
        assert "Step 2: The answer is" in result
        assert "Step 3: Therefore" in result

    def test_truncate_processor_answer_mode_multiple_occurrences(self, sample_context):
        """Test TruncateProcessor with mode='answer' removes all occurrences"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = """First: 12
Second: 12 again
Third: the value is 12!"""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # All occurrences of "12" should be removed
        assert "12" not in result
        assert "First:" in result
        assert "Second:" in result
        assert "again" in result
        assert "Third: the value is" in result

    def test_truncate_processor_answer_mode_word_boundary(self, sample_context):
        """Test TruncateProcessor with mode='answer' respects word boundaries"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = """Calculate 42 and 421 and 142.
The number is 42."""
        context = {"answer": "42"}

        result = processor.process(reasoning, context)

        # Only standalone "42" should be removed (word boundary)
        assert "421" in result  # 421 should remain
        assert "142" in result  # 142 should remain
        # But standalone 42 should be removed
        assert "Calculate  and 421" in result or "Calculate  " in result

    def test_truncate_processor_answer_mode_empty_answer(self, sample_context):
        """Test TruncateProcessor with mode='answer' handles empty answer"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = "This is some reasoning text."
        context = {"answer": ""}

        result = processor.process(reasoning, context)

        # Reasoning should be unchanged
        assert result == reasoning

    def test_truncate_processor_answer_mode_no_match(self, sample_context):
        """Test TruncateProcessor with mode='answer' when answer not in text"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = "This text does not contain the answer."
        context = {"answer": "999"}

        result = processor.process(reasoning, context)

        # Reasoning should be unchanged
        assert result == reasoning

    def test_truncate_processor_answer_mode_special_chars(self, sample_context):
        """Test TruncateProcessor with mode='answer' handles special regex chars"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = "The answer is 3.14 which is pi."
        context = {"answer": "3.14"}

        result = processor.process(reasoning, context)

        # Answer with special regex char (.) should be removed safely
        assert "3.14" not in result
        assert "The answer is" in result
        assert "which is pi" in result

    def test_truncate_processor_answer_mode_get_metadata(self, sample_context):
        """Test TruncateProcessor mode='answer' metadata"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = "The answer is 12. Final: 12."
        context = {"answer": "12"}

        _ = processor.process(reasoning, context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'truncate'
        assert metadata['mode'] == 'answer'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata


class TestShuffleProcessor:
    """Tests for ShuffleProcessor"""

    def test_shuffle_processor_line_mode(self, sample_context):
        """Test ShuffleProcessor with mode='line'"""
        from processors import ShuffleProcessor

        processor = ShuffleProcessor(mode='line', seed=42)
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        result = processor.process(reasoning, sample_context)

        # All lines should still be present
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" in result

    def test_shuffle_processor_word_mode(self, sample_context):
        """Test ShuffleProcessor with mode='word'"""
        from processors import ShuffleProcessor

        processor = ShuffleProcessor(mode='word', seed=42)
        reasoning = "The quick brown fox jumps"

        result = processor.process(reasoning, sample_context)

        # All words should still be present
        words = set(reasoning.split())
        result_words = set(result.split())
        assert words == result_words

    def test_shuffle_processor_token_mode(self, sample_context):
        """Test ShuffleProcessor with mode='token'"""
        from processors import ShuffleProcessor

        processor = ShuffleProcessor(mode='token', seed=42, tokenizer_model='gpt2')
        reasoning = "The quick brown fox"

        result = processor.process(reasoning, sample_context)

        assert len(result) > 0

    def test_shuffle_processor_seed_reproducible(self, sample_context):
        """Test that same seed produces same shuffle"""
        from processors import ShuffleProcessor

        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        processor1 = ShuffleProcessor(mode='line', seed=42)
        processor2 = ShuffleProcessor(mode='line', seed=42)

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        assert result1 == result2

    def test_shuffle_processor_get_metadata(self, sample_context):
        """Test ShuffleProcessor.get_metadata()"""
        from processors import ShuffleProcessor

        processor = ShuffleProcessor(mode='line', seed=42)
        reasoning = """Line 1
Line 2
Line 3"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'shuffle'
        assert metadata['mode'] == 'line'
        assert metadata['seed'] == 42
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_shuffle_processor_token_mode_with_model_type(self, sample_context):
        """Test ShuffleProcessor with mode='token' using model_type parameter"""
        from processors import ShuffleProcessor

        # Test with different model types
        for model_type in ['deepseek', 'gpt-oss', 'qwen3']:
            processor = ShuffleProcessor(mode='token', seed=42, model_type=model_type)
            reasoning = "The quick brown fox jumps over the lazy dog"

            result = processor.process(reasoning, sample_context)

            assert len(result) > 0, f"Failed for model_type={model_type}"

            # Verify metadata includes model_type
            metadata = processor.get_metadata()
            assert metadata['model_type'] == model_type

    def test_shuffle_processor_token_mode_model_type_reproducible(self, sample_context):
        """Test that ShuffleProcessor with model_type produces reproducible results"""
        from processors import ShuffleProcessor

        reasoning = "The quick brown fox jumps over the lazy dog"

        processor1 = ShuffleProcessor(mode='token', seed=42, model_type='gpt-oss')
        processor2 = ShuffleProcessor(mode='token', seed=42, model_type='gpt-oss')

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        assert result1 == result2


class TestInsertProcessor:
    """Tests for InsertProcessor"""

    def test_insert_processor_fix_mode_random_position(self, sample_context):
        """Test InsertProcessor with mode='fix' and position='random'"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Maybe the answer is 123.',
            position='random',
            count=3,
            seed=42
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        result = processor.process(reasoning, sample_context)

        # Inserted sentence should appear
        assert 'Maybe the answer is 123.' in result

        # All original lines should still be present
        assert 'Line 1' in result
        assert 'Line 2' in result
        assert 'Line 3' in result
        assert 'Line 4' in result

        # Should have more lines after insertion
        original_lines = len([l for l in reasoning.split('\n') if l.strip()])
        result_lines = len([l for l in result.split('\n') if l.strip()])
        assert result_lines == original_lines + 3

    def test_insert_processor_seed_reproducible(self, sample_context):
        """Test that same seed produces same insertion positions"""
        from processors import InsertProcessor

        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        processor1 = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=3,
            seed=42
        )
        processor2 = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=3,
            seed=42
        )

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        # Same seed should produce identical results
        assert result1 == result2

    def test_insert_processor_different_seeds(self, sample_context):
        """Test that different seeds produce different insertion positions"""
        from processors import InsertProcessor

        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        processor1 = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=3,
            seed=42
        )
        processor2 = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=3,
            seed=99
        )

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        # Different seeds should likely produce different results
        # (not guaranteed, but highly probable)
        assert result1 != result2

    def test_insert_processor_default_parameters(self, sample_context):
        """Test InsertProcessor with default parameters"""
        from processors import InsertProcessor

        processor = InsertProcessor(mode='fix')
        reasoning = """Line 1
Line 2"""

        result = processor.process(reasoning, sample_context)

        # Default sentence should appear
        assert 'Maybe the answer is 123.' in result

        # Original lines should be present
        assert 'Line 1' in result
        assert 'Line 2' in result

    def test_insert_processor_get_metadata(self, sample_context):
        """Test InsertProcessor.get_metadata()"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Test insertion.',
            position='random',
            count=2,
            seed=42
        )
        reasoning = """Line 1
Line 2
Line 3"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'insert'
        assert metadata['mode'] == 'fix'
        assert metadata['sentence'] == 'Test insertion.'
        assert metadata['position'] == 'random'
        assert metadata['count'] == 2
        assert metadata['seed'] == 42
        assert 'insertion_positions' in metadata
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

        # Check that insertion_positions is a list with correct length
        assert isinstance(metadata['insertion_positions'], list)
        assert len(metadata['insertion_positions']) == 2

    def test_insert_processor_invalid_mode(self, sample_context):
        """Test InsertProcessor with invalid mode raises error"""
        from processors import InsertProcessor

        processor = InsertProcessor(mode='invalid_mode')
        reasoning = "Line 1"

        with pytest.raises(ValueError, match="Invalid insert mode"):
            processor.process(reasoning, sample_context)

    def test_insert_processor_invalid_position(self, sample_context):
        """Test InsertProcessor with invalid position raises error"""
        from processors import InsertProcessor

        processor = InsertProcessor(mode='fix', position='invalid_position')
        reasoning = "Line 1"

        with pytest.raises(ValueError, match="Invalid position strategy"):
            processor.process(reasoning, sample_context)

    def test_insert_processor_count_zero(self, sample_context):
        """Test InsertProcessor with count=0 (no insertions)"""
        from processors import InsertProcessor

        processor = InsertProcessor(mode='fix', count=0)
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # No insertions should occur
        assert result == reasoning

    def test_insert_processor_count_multiple(self, sample_context):
        """Test InsertProcessor with large count"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=10,
            seed=42
        )
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Should have 10 insertions
        original_lines = len([l for l in reasoning.split('\n') if l.strip()])
        result_lines = len([l for l in result.split('\n') if l.strip()])
        assert result_lines == original_lines + 10

        # Count occurrences of inserted sentence
        assert result.count('Noise') == 10

    def test_insert_processor_custom_sentence(self, sample_context):
        """Test InsertProcessor with custom sentence"""
        from processors import InsertProcessor

        custom_sentence = "This is a custom noise sentence with numbers 999."
        processor = InsertProcessor(
            mode='fix',
            sentence=custom_sentence,
            position='random',
            count=1,
            seed=42
        )
        reasoning = "Line 1"

        result = processor.process(reasoning, sample_context)

        # Custom sentence should appear
        assert custom_sentence in result

    def test_insert_processor_empty_reasoning(self, sample_context):
        """Test InsertProcessor with empty reasoning"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=1,
            seed=42
        )
        reasoning = ""

        result = processor.process(reasoning, sample_context)

        # Should insert into empty text
        assert 'Noise' in result

    def test_insert_processor_single_line(self, sample_context):
        """Test InsertProcessor with single-line reasoning"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=2,
            seed=42
        )
        reasoning = "Single line"

        result = processor.process(reasoning, sample_context)

        # Original line should be present
        assert 'Single line' in result

        # Should have insertions
        assert result.count('Noise') == 2


class TestQuestionProcessor:
    """Tests for QuestionProcessor"""

    def test_question_remove_mode(self, sample_context):
        """Test QuestionProcessor with mode='remove'"""
        from processors import QuestionProcessor

        processor = QuestionProcessor(mode='remove')
        reasoning = "Step 1: Calculate something\nStep 2: Get answer"

        # Context should be modified in place
        context = sample_context.copy()
        original_question = context['question']

        result = processor.process(reasoning, context)

        # Reasoning should be unchanged
        assert result == reasoning

        # Question should be empty
        assert context['question'] == ''

        # Original question should be stored in metadata
        metadata = processor.get_metadata()
        assert metadata['processor'] == 'question'
        assert metadata['mode'] == 'remove'
        assert metadata['original_question'] == original_question

    def test_question_invalid_mode(self, sample_context):
        """Test QuestionProcessor with invalid mode"""
        from processors import QuestionProcessor

        processor = QuestionProcessor(mode='invalid')
        reasoning = "Test reasoning"

        with pytest.raises(ValueError, match="Invalid question mode"):
            processor.process(reasoning, sample_context)


class TestRemoveProcessor:
    """Tests for RemoveProcessor"""

    def test_remove_blank_mode_basic(self, sample_context):
        """Test RemoveProcessor with mode='blank' on basic text"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "hello\n\n   world   \t\n\n   test"

        result = processor.process(reasoning, sample_context)

        # All continuous blanks should be consolidated to single space
        assert result == "hello world test"

    def test_remove_blank_mode_reasoning_text(self, sample_context):
        """Test RemoveProcessor with mode='blank' on reasoning-like text"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = """Let me solve this problem step by step.


First, I will analyze the question.


   Then,   I calculate:

x = 5


Therefore, the answer is 5."""

        result = processor.process(reasoning, sample_context)

        # All continuous blanks should be consolidated
        expected = "Let me solve this problem step by step. First, I will analyze the question. Then, I calculate: x = 5 Therefore, the answer is 5."
        assert result == expected

    def test_remove_blank_mode_multiple_spaces(self, sample_context):
        """Test RemoveProcessor removes multiple spaces"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "word1    word2     word3      word4"

        result = processor.process(reasoning, sample_context)

        assert result == "word1 word2 word3 word4"

    def test_remove_blank_mode_multiple_newlines(self, sample_context):
        """Test RemoveProcessor removes multiple newlines"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "Line 1\n\n\n\nLine 2\n\n\nLine 3"

        result = processor.process(reasoning, sample_context)

        assert result == "Line 1 Line 2 Line 3"

    def test_remove_blank_mode_tabs(self, sample_context):
        """Test RemoveProcessor removes tabs"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "word1\t\t\tword2\t\tword3"

        result = processor.process(reasoning, sample_context)

        assert result == "word1 word2 word3"

    def test_remove_blank_mode_mixed_whitespace(self, sample_context):
        """Test RemoveProcessor with mixed whitespace characters"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "word1 \t\n\n  \t  word2   \n\t   word3"

        result = processor.process(reasoning, sample_context)

        assert result == "word1 word2 word3"

    def test_remove_blank_mode_leading_trailing_whitespace(self, sample_context):
        """Test RemoveProcessor strips leading/trailing whitespace"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "   \n\n  hello world  \n\n   "

        result = processor.process(reasoning, sample_context)

        assert result == "hello world"

    def test_remove_blank_mode_empty_string(self, sample_context):
        """Test RemoveProcessor with empty string"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = ""

        result = processor.process(reasoning, sample_context)

        assert result == ""

    def test_remove_blank_mode_only_whitespace(self, sample_context):
        """Test RemoveProcessor with only whitespace"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "   \n\n\t\t   \n   "

        result = processor.process(reasoning, sample_context)

        assert result == ""

    def test_remove_blank_mode_single_word(self, sample_context):
        """Test RemoveProcessor with single word"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "hello"

        result = processor.process(reasoning, sample_context)

        assert result == "hello"

    def test_remove_blank_mode_preserves_content(self, sample_context):
        """Test RemoveProcessor preserves non-whitespace content"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "Calculate  \n\n  2 + 2  =   \n  4   and   multiply   \n\n by  3:   \n 4 × 3 = 12"

        result = processor.process(reasoning, sample_context)

        # All non-whitespace characters should be preserved
        assert "Calculate" in result
        assert "2 + 2 = 4" in result
        assert "multiply" in result
        assert "3:" in result
        assert "4 × 3 = 12" in result

    def test_remove_processor_get_metadata(self, sample_context):
        """Test RemoveProcessor.get_metadata()"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "hello\n\n\nworld   test"

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'remove'
        assert metadata['mode'] == 'blank'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

        # Verify stats are computed correctly
        assert metadata['input_stats']['lines'] > 0
        assert metadata['output_stats']['lines'] > 0

    def test_remove_processor_invalid_mode(self, sample_context):
        """Test RemoveProcessor with invalid mode raises error"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='invalid_mode')
        reasoning = "Test reasoning"

        with pytest.raises(ValueError, match="Invalid remove mode"):
            processor.process(reasoning, sample_context)


class TestReplaceProcessor:
    """Tests for ReplaceProcessor"""

    def test_replace_processor_whitespace_basic(self, sample_context):
        """Test ReplaceProcessor replacing all whitespace with single space"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\s', replacement=' ')
        reasoning = "hello\nworld\ttest"

        result = processor.process(reasoning, sample_context)

        # All whitespace should be replaced with single space
        assert result == "hello world test"

    def test_replace_processor_digits(self, sample_context):
        """Test ReplaceProcessor replacing all digits"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d', replacement='X')
        reasoning = "Calculate 2 + 2 = 4"

        result = processor.process(reasoning, sample_context)

        # All digits should be replaced with 'X'
        assert result == "Calculate X + X = X"

    def test_replace_processor_word(self, sample_context):
        """Test ReplaceProcessor replacing specific word"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\banswer\b', replacement='result')
        reasoning = "The answer is 42. Calculate the answer."

        result = processor.process(reasoning, sample_context)

        # Word 'answer' should be replaced with 'result'
        assert result == "The result is 42. Calculate the result."

    def test_replace_processor_remove_pattern(self, sample_context):
        """Test ReplaceProcessor removing pattern (empty replacement)"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d+', replacement='')
        reasoning = "Step 1: Calculate 42 in step 2."

        result = processor.process(reasoning, sample_context)

        # All digits should be removed
        assert result == "Step : Calculate  in step ."

    def test_replace_processor_multiple_replacements(self, sample_context):
        """Test ReplaceProcessor counts multiple replacements"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d', replacement='#')
        reasoning = "1 2 3 4 5"

        result = processor.process(reasoning, sample_context)

        # Should replace 5 digits
        assert result == "# # # # #"

        metadata = processor.get_metadata()
        assert metadata['num_replacements'] == 5

    def test_replace_processor_no_match(self, sample_context):
        """Test ReplaceProcessor when pattern doesn't match"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d', replacement='X')
        reasoning = "No numbers here"

        result = processor.process(reasoning, sample_context)

        # Should be unchanged
        assert result == "No numbers here"

        metadata = processor.get_metadata()
        assert metadata['num_replacements'] == 0

    def test_replace_processor_complex_pattern(self, sample_context):
        """Test ReplaceProcessor with complex regex pattern"""
        from processors import ReplaceProcessor

        # Replace mathematical operators
        processor = ReplaceProcessor(pattern=r'[+\-*/=]', replacement='OP')
        reasoning = "2 + 3 - 1 * 4 / 2 = 8"

        result = processor.process(reasoning, sample_context)

        assert result == "2 OP 3 OP 1 OP 4 OP 2 OP 8"

    def test_replace_processor_newline_to_space(self, sample_context):
        """Test ReplaceProcessor replacing newlines with spaces"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\n', replacement=' ')
        reasoning = "Line 1\nLine 2\nLine 3"

        result = processor.process(reasoning, sample_context)

        assert result == "Line 1 Line 2 Line 3"

    def test_replace_processor_multiline_reasoning(self, sample_context):
        """Test ReplaceProcessor on multiline reasoning text"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\s+', replacement=' ')
        reasoning = """Step 1: Calculate something


Step 2: Do more work

Final answer: 42"""

        result = processor.process(reasoning, sample_context)

        # Multiple spaces/newlines should be replaced with single space
        assert "Step 1: Calculate something Step 2: Do more work Final answer: 42" == result

    def test_replace_processor_get_metadata(self, sample_context):
        """Test ReplaceProcessor.get_metadata()"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d', replacement='X')
        reasoning = "Numbers: 1 2 3"

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'replace'
        assert metadata['pattern'] == r'\d'
        assert metadata['replacement'] == 'X'
        assert metadata['num_replacements'] == 3
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_replace_processor_empty_reasoning(self, sample_context):
        """Test ReplaceProcessor with empty reasoning"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\s', replacement=' ')
        reasoning = ""

        result = processor.process(reasoning, sample_context)

        assert result == ""

        metadata = processor.get_metadata()
        assert metadata['num_replacements'] == 0

    def test_replace_processor_default_replacement(self, sample_context):
        """Test ReplaceProcessor with default empty replacement"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\s')
        reasoning = "a b c"

        result = processor.process(reasoning, sample_context)

        # Should remove all whitespace (default replacement is '')
        assert result == "abc"


class TestReasonIsProcessor:
    """Tests for ReasonIsProcessor - replaces reasoning with answer only"""

    def test_reason_is_answer_mode_basic(self, sample_context):
        """Test ReasonIsProcessor replaces reasoning with pure answer"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer')
        reasoning = """Let's solve this step by step.
First, we calculate 2 + 2 = 4.
Then, we multiply by 3: 4 × 3 = 12.
Therefore, the answer is 12."""

        result = processor.process(reasoning, sample_context)

        # Should replace entire reasoning with just the answer
        assert result == "12"

    def test_reason_is_answer_with_boxed_answer(self, sample_context):
        """Test ReasonIsProcessor with boxed answer format"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer')
        reasoning = """Step 1: Calculate 5 + 3 = 8
Step 2: Multiply by 2 = 16
Final answer: \\boxed{16}"""

        result = processor.process(reasoning, sample_context)

        # Should use ground_truth from context
        assert result == "12"

    def test_reason_is_answer_extracts_from_context(self, sample_context):
        """Test ReasonIsProcessor uses answer from context"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer')
        reasoning = "Some complex reasoning that we don't care about"

        context = {
            "question": "What is 5 + 5?",
            "answer": "10",
            "ground_truth": "10"
        }

        result = processor.process(reasoning, context)

        # Should use answer from context, not parse from reasoning
        assert result == "10"

    def test_reason_is_answer_with_different_answer_formats(self, sample_context):
        """Test ReasonIsProcessor with various answer types"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer')

        # Test with numeric answer
        context1 = sample_context.copy()
        context1['answer'] = "42"
        result1 = processor.process("reasoning...", context1)
        assert result1 == "42"

        # Test with fractional answer
        context2 = sample_context.copy()
        context2['answer'] = "3/4"
        result2 = processor.process("reasoning...", context2)
        assert result2 == "3/4"

        # Test with text answer
        context3 = sample_context.copy()
        context3['answer'] = "impossible"
        result3 = processor.process("reasoning...", context3)
        assert result3 == "impossible"

    def test_reason_is_answer_get_metadata(self, sample_context):
        """Test ReasonIsProcessor.get_metadata()"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer')
        reasoning = "Long reasoning text that will be replaced"

        result = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'reason_is'
        assert metadata['mode'] == 'answer'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert metadata['input_stats']['chars'] > 0
        assert metadata['output_stats']['chars'] == len("12")

    def test_reason_is_invalid_mode(self, sample_context):
        """Test ReasonIsProcessor with invalid mode raises error"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='invalid_mode')
        reasoning = "Some reasoning"

        with pytest.raises(ValueError, match="Invalid mode for reason_is"):
            processor.process(reasoning, sample_context)

    def test_reason_is_missing_answer_in_context(self, sample_context):
        """Test ReasonIsProcessor raises error when answer missing from context"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer')
        reasoning = "Some reasoning"

        # Context without answer
        context = {"question": "What is 2+2?"}

        with pytest.raises(KeyError, match="'answer' not found in context"):
            processor.process(reasoning, context)

    def test_reason_is_in_pipeline(self, sample_context):
        """Test ReasonIsProcessor works in pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "reason_is('answer')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Complex reasoning with multiple steps and calculations"
        context = sample_context.copy()
        result, metadata_list = pipeline.execute(reasoning, context)

        # Should replace reasoning with answer
        assert result == "12"
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'reason_is'
        assert metadata_list[0]['mode'] == 'answer'

    def test_reason_is_combined_with_mask(self, sample_context):
        """Test ReasonIsProcessor combined with mask processor"""
        from pipeline import parse_flow, Pipeline

        # First mask numbers, then replace with answer
        flow_str = "mask('number'),reason_is('answer')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Calculate: 2 + 2 = 4"
        context = sample_context.copy()
        result, metadata_list = pipeline.execute(reasoning, context)

        # After mask and reason_is, should just be the answer
        assert result == "12"
        assert len(metadata_list) == 2
        assert metadata_list[0]['processor'] == 'mask'
        assert metadata_list[1]['processor'] == 'reason_is'

    def test_reason_is_empty_reasoning(self, sample_context):
        """Test ReasonIsProcessor with empty reasoning"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer')
        reasoning = ""

        result = processor.process(reasoning, sample_context)

        # Should still return the answer even with empty reasoning
        assert result == "12"

    def test_reason_is_answer_with_illustrate_mode(self, sample_context):
        """Test ReasonIsProcessor with answer_with_illustrate mode"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer_with_illustrate')
        reasoning = "Long reasoning that will be replaced"

        result = processor.process(reasoning, sample_context)

        # Should return "Thus, the answer is {answer}"
        assert result == "Thus, the answer is 12"

    def test_reason_is_answer_with_illustrate_different_answers(self, sample_context):
        """Test ReasonIsProcessor answer_with_illustrate with various answers"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer_with_illustrate')

        # Test with numeric answer
        context1 = sample_context.copy()
        context1['answer'] = "42"
        result1 = processor.process("reasoning...", context1)
        assert result1 == "Thus, the answer is 42"

        # Test with fractional answer
        context2 = sample_context.copy()
        context2['answer'] = "3/4"
        result2 = processor.process("reasoning...", context2)
        assert result2 == "Thus, the answer is 3/4"

        # Test with text answer
        context3 = sample_context.copy()
        context3['answer'] = "impossible"
        result3 = processor.process("reasoning...", context3)
        assert result3 == "Thus, the answer is impossible"

    def test_reason_is_answer_with_illustrate_in_pipeline(self, sample_context):
        """Test ReasonIsProcessor answer_with_illustrate in pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "reason_is('answer_with_illustrate')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Complex reasoning with multiple steps"
        context = sample_context.copy()
        result, metadata_list = pipeline.execute(reasoning, context)

        # Should replace reasoning with "Thus, the answer is {answer}"
        assert result == "Thus, the answer is 12"
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'reason_is'
        assert metadata_list[0]['mode'] == 'answer_with_illustrate'

    def test_reason_is_answer_with_illustrate_get_metadata(self, sample_context):
        """Test ReasonIsProcessor answer_with_illustrate metadata"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(mode='answer_with_illustrate')
        reasoning = "Some reasoning text"

        result = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'reason_is'
        assert metadata['mode'] == 'answer_with_illustrate'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert metadata['output_stats']['chars'] == len("Thus, the answer is 12")
