"""
Unit tests for processors.py

This test file covers all Processor classes:
- Processor (base class)
- MaskProcessor
- TruncateProcessor
- ShuffleProcessor
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
