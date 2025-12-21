"""
Tests for AttentionExtractor class in run_attn_visual.py

Following TDD Red-Green-Refactor cycle for Phase 4

Note: These tests use a very small model (gpt2) for speed.
For production, use the actual model specified by user.
"""

import pytest
import numpy as np


# Use a small model for testing to avoid downloading large models
TEST_MODEL = "gpt2"  # Very small model for fast testing


class TestAttentionExtractor:
    """Test suite for AttentionExtractor class"""

    @pytest.fixture(scope="class")
    def extractor(self):
        """Fixture providing AttentionExtractor instance"""
        from run_attn_visual import AttentionExtractor

        # Use small test model
        return AttentionExtractor(TEST_MODEL)

    def test_initialization(self, extractor):
        """Test AttentionExtractor initialization"""
        assert extractor.model_name == TEST_MODEL
        assert extractor.tokenizer is not None
        assert extractor.model is not None
        assert extractor.device in ["cuda", "cpu"]

    def test_tokenize_simple_text(self, extractor):
        """Test tokenization of simple text"""
        prompt = "Hello, world!"

        result = extractor.tokenize(prompt)

        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert result['input_ids'].shape[0] == 1  # Batch size 1
        assert result['input_ids'].shape[1] > 0  # Has tokens

    def test_tokenize_empty_string(self, extractor):
        """Test tokenization of empty string"""
        prompt = ""

        result = extractor.tokenize(prompt)

        # Should still return valid structure
        assert 'input_ids' in result

    def test_extract_attention_returns_correct_structure(self, extractor):
        """Test that extract_last_token_attention returns correct structure"""
        prompt = "This is a test prompt."

        tokens, attention_maps = extractor.extract_last_token_attention(prompt)

        # Verify tokens
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

        # Verify attention maps
        assert isinstance(attention_maps, list)
        assert len(attention_maps) > 0  # Should have multiple layers

        # Each layer should be a numpy array
        for attn_map in attention_maps:
            assert isinstance(attn_map, np.ndarray)
            assert attn_map.ndim == 1  # Should be 1D array (seq_len,)
            assert len(attn_map) == len(tokens)  # Should match token count

    def test_attention_weights_sum_to_one(self, extractor):
        """Test that attention weights sum to approximately 1.0 for each layer"""
        prompt = "Test attention weights."

        tokens, attention_maps = extractor.extract_last_token_attention(prompt)

        # Check each layer
        for i, attn_map in enumerate(attention_maps):
            weight_sum = attn_map.sum()
            # Allow small floating point errors
            assert abs(weight_sum - 1.0) < 0.01, f"Layer {i} attention sum: {weight_sum}"

    def test_attention_weights_are_non_negative(self, extractor):
        """Test that all attention weights are non-negative"""
        prompt = "Check non-negative weights."

        tokens, attention_maps = extractor.extract_last_token_attention(prompt)

        for i, attn_map in enumerate(attention_maps):
            assert np.all(attn_map >= 0), f"Layer {i} has negative weights"

    def test_different_prompt_lengths(self, extractor):
        """Test attention extraction with different prompt lengths"""
        short_prompt = "Hi"
        long_prompt = "This is a much longer prompt with many more tokens to process."

        tokens_short, attn_short = extractor.extract_last_token_attention(short_prompt)
        tokens_long, attn_long = extractor.extract_last_token_attention(long_prompt)

        # Longer prompt should have more tokens
        assert len(tokens_long) > len(tokens_short)

        # Both should have same number of layers
        assert len(attn_short) == len(attn_long)

        # Attention maps should match token counts
        assert all(len(attn) == len(tokens_short) for attn in attn_short)
        assert all(len(attn) == len(tokens_long) for attn in attn_long)

    def test_attention_shape_matches_sequence_length(self, extractor):
        """Test that attention map shape matches sequence length"""
        prompt = "Testing shape consistency."

        tokens, attention_maps = extractor.extract_last_token_attention(prompt)

        seq_len = len(tokens)

        for attn_map in attention_maps:
            assert attn_map.shape == (seq_len,), \
                f"Expected shape ({seq_len},), got {attn_map.shape}"

    def test_model_output_has_attentions(self, extractor):
        """Test that model is configured to output attentions"""
        # This is a sanity check that the model was loaded correctly
        assert extractor.model.config.output_attentions or \
               hasattr(extractor.model, 'output_attentions')

    def test_extract_attention_consistency(self, extractor):
        """Test that extracting attention twice gives same result"""
        prompt = "Consistency test."

        tokens1, attn1 = extractor.extract_last_token_attention(prompt)
        tokens2, attn2 = extractor.extract_last_token_attention(prompt)

        # Tokens should be identical
        assert tokens1 == tokens2

        # Attention maps should be very similar (allowing for floating point)
        assert len(attn1) == len(attn2)
        for a1, a2 in zip(attn1, attn2):
            np.testing.assert_allclose(a1, a2, rtol=1e-5)

    @pytest.mark.slow
    def test_extract_attention_with_long_sequence(self, extractor):
        """Test attention extraction with very long sequence"""
        # Create a longer prompt
        prompt = " ".join(["word"] * 100)

        tokens, attention_maps = extractor.extract_last_token_attention(prompt)

        # Should still work correctly
        assert len(tokens) > 50
        assert all(len(attn) == len(tokens) for attn in attention_maps)
        assert all(abs(attn.sum() - 1.0) < 0.01 for attn in attention_maps)
