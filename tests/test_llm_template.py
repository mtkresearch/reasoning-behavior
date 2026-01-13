"""Tests for LLM completion template handling"""
import pytest
from llm_client import LLMClient, CompletionRequest


class TestCompletionTemplates:
    """Test completion template application for different models"""

    def test_gpt_oss_template_basic(self):
        """Test basic GPT-OSS completion template application"""
        client = LLMClient(mode="openrouter", api_key="test_key")

        question = "What is 2+2?"
        reasoning = "Let me think about this..."
        answer_prefix = "The answer is"

        formatted = client._apply_completion_template(
            question=question,
            reasoning=reasoning,
            answer_prefix=answer_prefix,
            model_type="gpt-oss"
        )

        # Should use gpt-oss format: <|start|>...<|message|>...<|end|>
        assert "<|start|>system<|message|>" in formatted
        assert "<|start|>user<|message|>" in formatted
        assert "<|start|>assistant<|channel|>analysis<|message|>" in formatted
        assert "<|start|>assistant<|channel|>final<|message|>" in formatted
        assert question in formatted
        assert reasoning in formatted
        assert answer_prefix in formatted
        assert "<|end|>" in formatted

    def test_gpt_oss_template_has_fixed_format(self):
        """Test GPT-OSS template has fixed system message format"""
        client = LLMClient(mode="openrouter", api_key="test_key")

        question = "Solve this problem"
        reasoning = "Step 1..."
        answer_prefix = ""

        formatted = client._apply_completion_template(
            question=question,
            reasoning=reasoning,
            answer_prefix=answer_prefix,
            model_type="gpt-oss"
        )

        # gpt-oss uses fixed format with model identity and metadata
        assert "ChatGPT" in formatted
        assert "Knowledge cutoff" in formatted
        assert "Current date" in formatted
        assert "Reasoning:" in formatted
        assert "Valid channels" in formatted
        assert question in formatted
        assert reasoning in formatted

    def test_local_mode_raises_exception(self):
        """Test that local mode raises exception"""
        client = LLMClient(mode="local")

        question = "What is 2+2?"
        reasoning = "Let me think..."
        answer_prefix = ""

        with pytest.raises(Exception):
            client._apply_completion_template(
                question=question,
                reasoning=reasoning,
                answer_prefix=answer_prefix,
                model_type="gpt-oss"
            )

    def test_deepseek_v3_template(self):
        """Test DeepSeek-v3 completion template application"""
        client = LLMClient(mode="openrouter", api_key="test_key")

        question = "What is 2+2?"
        reasoning = "Let me calculate..."
        answer_prefix = "Answer:"

        formatted = client._apply_completion_template(
            question=question,
            reasoning=reasoning,
            answer_prefix=answer_prefix,
            model_type="deepseek-v3"
        )

        # Should use ChatML format
        assert "<|im_start|>" in formatted
        assert "<|im_end|>" in formatted
        assert question in formatted
        assert reasoning in formatted
        assert answer_prefix in formatted

    def test_unknown_model_raises_exception(self):
        """Test that unknown model types raise exception"""
        client = LLMClient(mode="openrouter", api_key="test_key")

        question = "What is 2+2?"
        reasoning = "Thinking..."
        answer_prefix = ""

        with pytest.raises(Exception):
            client._apply_completion_template(
                question=question,
                reasoning=reasoning,
                answer_prefix=answer_prefix,
                model_type="unknown-model"
            )
