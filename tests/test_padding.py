"""
Unit tests for PaddingProcessor

Tests the padding processor that replaces reasoning with random tokens or words.
The count of replacement tokens/words equals the original reasoning's count.
"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_words_tsv(tmp_path):
    """Create a sample words.tsv file for testing"""
    words_file = tmp_path / "words.tsv"
    content = """word\tfrequency
the\t100
is\t80
to\t60
a\t40
and\t20
"""
    words_file.write_text(content)
    return str(words_file)


class TestPaddingProcessor:
    """Tests for PaddingProcessor"""

    def test_padding_processor_token_mode(self, sample_context):
        """Test PaddingProcessor with mode='token' - replacement count equals original token count"""
        from processors import PaddingProcessor
        from transformers import AutoTokenizer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        processor = PaddingProcessor(
            mode='token',
            tokenizer_model='gpt2',
            seed=42
        )
        reasoning = "Calculate 2 + 2 = 4."

        # Count original tokens
        tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
        original_tokens = tokenizer.encode(reasoning, add_special_tokens=False)
        original_token_count = len(original_tokens)

        result = processor.process(reasoning, sample_context)

        # Result should NOT contain the original reasoning (it's replaced)
        assert reasoning not in result
        # Result should be random tokens
        assert len(result) > 0
        # Replacement count should equal original token count
        assert processor.padding_count == original_token_count
        # Result token count should equal original token count
        result_tokens = tokenizer.encode(result, add_special_tokens=False)
        assert len(result_tokens) == original_token_count

    def test_padding_processor_word_mode(self, sample_context, sample_words_tsv):
        """Test PaddingProcessor with mode='word' - replacement count equals original word count"""
        from processors import PaddingProcessor

        processor = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            seed=42
        )
        reasoning = "Calculate 2 + 2 = 4."

        # Count original words
        original_word_count = len(reasoning.split())

        result = processor.process(reasoning, sample_context)

        # Result should NOT contain the original reasoning (it's replaced)
        assert reasoning not in result
        # Result should be random words
        assert len(result) > 0
        # Replacement count should equal original word count
        assert processor.padding_count == original_word_count
        # Result word count should equal original word count
        assert len(result.split()) == original_word_count

    def test_padding_processor_seed_reproducibility(self, sample_context, sample_words_tsv):
        """Test that padding is reproducible with same seed"""
        from processors import PaddingProcessor

        reasoning = "Test reasoning text."

        # Create two processors with same seed
        processor1 = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            seed=42
        )
        processor2 = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            seed=42
        )

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        # Results should be identical
        assert result1 == result2

    def test_padding_processor_different_seeds(self, sample_context, sample_words_tsv):
        """Test that different seeds produce different results"""
        from processors import PaddingProcessor

        reasoning = "Test reasoning text."

        processor1 = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            seed=42
        )
        processor2 = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            seed=99
        )

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        # Results should be different (with very high probability)
        assert result1 != result2

    def test_padding_processor_different_reasoning_lengths(self, sample_context, sample_words_tsv):
        """Test that padding count adapts to different reasoning lengths"""
        from processors import PaddingProcessor

        reasoning_short = "Test."
        reasoning_long = "This is a much longer test reasoning with many more words."

        processor = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            seed=42
        )

        # Process short reasoning
        result_short = processor.process(reasoning_short, sample_context)
        count_short = processor.padding_count

        # Process long reasoning
        result_long = processor.process(reasoning_long, sample_context)
        count_long = processor.padding_count

        # Padding count should match original word counts
        assert count_short == len(reasoning_short.split())
        assert count_long == len(reasoning_long.split())
        # Long reasoning should get more padding
        assert count_long > count_short

    def test_padding_processor_get_metadata(self, sample_context, sample_words_tsv):
        """Test PaddingProcessor.get_metadata()"""
        from processors import PaddingProcessor

        processor = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            seed=42
        )
        reasoning = "Calculate 2 + 2 = 4."

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'padding'
        assert metadata['mode'] == 'word'
        assert metadata['padding_count'] == len(reasoning.split())
        assert metadata['seed'] == 42
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_padding_processor_invalid_mode(self, sample_context):
        """Test that invalid mode raises ValueError"""
        from processors import PaddingProcessor

        processor = PaddingProcessor(mode='invalid')
        reasoning = "Test."

        with pytest.raises(ValueError, match="Invalid padding mode"):
            processor.process(reasoning, sample_context)

    def test_padding_processor_word_mode_without_tsv_path(self, sample_context):
        """Test that word mode without words_tsv_path raises error"""
        from processors import PaddingProcessor

        processor = PaddingProcessor(mode='word')
        reasoning = "Test."

        with pytest.raises(ValueError, match="words_tsv_path is required"):
            processor.process(reasoning, sample_context)

    def test_padding_processor_position_parameter_deprecated(self, sample_context, sample_words_tsv):
        """Test that position parameter is ignored (deprecated)"""
        from processors import PaddingProcessor

        reasoning = "Original text."

        # Test 'after' position (should be ignored)
        processor_after = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            position='after',
            seed=42
        )
        result_after = processor_after.process(reasoning, sample_context)

        # Test 'before' position (should be ignored, result should be same)
        processor_before = PaddingProcessor(
            mode='word',
            words_tsv_path=sample_words_tsv,
            position='before',
            seed=42
        )
        result_before = processor_before.process(reasoning, sample_context)

        # Both should produce the same result (position is ignored)
        assert result_after == result_before
        # Result should NOT contain original reasoning
        assert reasoning not in result_after
        assert reasoning not in result_before
        # Result should be random words with same count as original
        assert len(result_after.split()) == len(reasoning.split())


class TestPaddingUtilities:
    """Tests for padding utility functions in core.py"""

    def test_generate_random_tokens_function(self):
        """Test generate_random_tokens function"""
        from core import generate_random_tokens
        from transformers import AutoTokenizer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)

        result = generate_random_tokens(10, tokenizer, seed=42)

        # Result should be a string
        assert isinstance(result, str)
        # Result should not be empty
        assert len(result) > 0

    def test_generate_random_words_function(self, sample_words_tsv):
        """Test generate_random_words function"""
        from core import generate_random_words

        result = generate_random_words(10, sample_words_tsv, seed=42)

        # Result should be a string
        assert isinstance(result, str)
        # Result should contain spaces (words separated by spaces)
        assert ' ' in result
        # Should contain approximately 10 words
        word_count = len(result.split())
        assert word_count == 10

    def test_load_word_frequency_function(self, sample_words_tsv):
        """Test load_word_frequency function"""
        from core import load_word_frequency

        words, frequencies = load_word_frequency(sample_words_tsv)

        # Should return lists
        assert isinstance(words, list)
        assert isinstance(frequencies, list)
        # Lists should have same length
        assert len(words) == len(frequencies)
        # Should contain expected words
        assert 'the' in words
        assert 'is' in words
        # Frequencies should be integers
        assert all(isinstance(f, int) for f in frequencies)

    def test_get_english_token_ids_function(self):
        """Test get_english_token_ids function"""
        from core import get_english_token_ids
        from transformers import AutoTokenizer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)

        token_ids = get_english_token_ids(tokenizer)

        # Should return a list
        assert isinstance(token_ids, list)
        # Should not be empty
        assert len(token_ids) > 0
        # All elements should be integers
        assert all(isinstance(tid, int) for tid in token_ids)


class TestPaddingIntegration:
    """Integration tests for padding with pipeline"""

    def test_padding_in_pipeline(self, sample_context, sample_words_tsv):
        """Test padding processor in a pipeline"""
        from pipeline import parse_flow, Pipeline

        # Create a flow with padding (no count parameter needed)
        flow_str = f"padding('word',words_tsv_path='{sample_words_tsv}')"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].__class__.__name__ == 'PaddingProcessor'

        # Execute pipeline
        reasoning = "Test reasoning text."
        pipeline = Pipeline(processors)
        result, metadata = pipeline.execute(reasoning, sample_context)

        # Result should NOT contain original reasoning
        assert reasoning not in result
        # Result should have the same word count as original
        assert len(result.split()) == len(reasoning.split())
        # Metadata should be collected
        assert len(metadata) == 1
        assert metadata[0]['processor'] == 'padding'
        # Replacement count should equal original word count
        assert metadata[0]['padding_count'] == len(reasoning.split())

    def test_padding_with_other_processors(self, sample_context, sample_words_tsv):
        """Test padding combined with other processors"""
        from pipeline import parse_flow, Pipeline

        # Create a flow: padding + mask (no count parameter needed)
        flow_str = f"padding('word',words_tsv_path='{sample_words_tsv}'),mask('number')"
        processors = parse_flow(flow_str)

        assert len(processors) == 2

        # Execute pipeline
        reasoning = "Calculate 2 + 2 = 4."
        pipeline = Pipeline(processors)
        result, metadata = pipeline.execute(reasoning, sample_context)

        # Result should NOT contain original reasoning (replaced by random words)
        assert reasoning not in result
        # Result should have the same word count as original
        assert len(result.split()) == len(reasoning.split())
        # Any numbers in the random words should be masked
        # (We can't test specific numbers since they're random)
