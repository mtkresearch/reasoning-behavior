"""
Tests for OLMo base model support in llm_client.py

Following TDD approach:
- Test template generation with full fields
- Test template generation without reasoning
- Test OpenRouter exception handling
- Test extra_body handling
- Test generate() API exception
"""

import pytest
from llm_client import LLMClient


class TestOLMoBaseTemplate:
    """Test OLMo base model template generation"""

    def test_olmo_base_template_full(self):
        """Test OLMo base model template with all fields"""
        client = LLMClient(mode="local")

        template = client._apply_completion_template(
            question="What is 2+2?",
            reasoning="Let me calculate: 2 + 2 = 4",
            answer_prefix="The answer is ",
            model_type="olmo--base",
            system_prompt="You are a helpful assistant",
            reasoning_on=True,
        )

        # 驗證結構
        assert "You are a helpful assistant" in template
        assert "## Question:\nWhat is 2+2?" in template
        assert "## Reasoning:\nLet me calculate: 2 + 2 = 4" in template
        assert "## Answer:\nThe answer is " in template

        # 確認無 chat tokens
        assert "<|im_start|>" not in template
        assert "<|im_end|>" not in template

    def test_olmo_base_template_no_reasoning(self):
        """Test OLMo base model without reasoning"""
        client = LLMClient(mode="local")

        template = client._apply_completion_template(
            question="What is 2+2?",
            reasoning="",
            answer_prefix="",
            model_type="olmo--base",
            system_prompt="You are helpful",
            reasoning_on=False,
        )

        assert "## Question:" in template
        assert "## Answer:" in template
        assert "## Reasoning:" not in template

    def test_olmo_base_template_structure(self):
        """Test that template follows expected Markdown structure"""
        client = LLMClient(mode="local")

        template = client._apply_completion_template(
            question="Test question",
            reasoning="Test reasoning",
            answer_prefix="Answer: ",
            model_type="olmo--base",
            system_prompt="System",
            reasoning_on=True,
        )

        # 驗證順序：system -> question -> reasoning -> answer
        lines = template.split('\n')
        assert lines[0] == "System"
        assert "## Question:" in template
        assert "## Reasoning:" in template
        assert "## Answer:" in template

        # 驗證順序正確
        q_idx = template.find("## Question:")
        r_idx = template.find("## Reasoning:")
        a_idx = template.find("## Answer:")
        assert q_idx < r_idx < a_idx


class TestOLMoBaseOpenRouter:
    """Test OpenRouter mode error handling for OLMo base"""

    def test_olmo_base_openrouter_not_supported(self):
        """Test that OpenRouter mode raises exception for olmo--base"""
        client = LLMClient(mode="openrouter")

        with pytest.raises(ValueError, match="Unsupported model_type for openrouter"):
            client._get_model("olmo--base")


class TestOLMoBaseExtraBody:
    """Test extra_body handling for OLMo base"""

    def test_olmo_base_extra_body(self):
        """Test that _get_extra_body() doesn't add reasoning flag for olmo--base"""
        from llm_client import CompletionRequest

        client = LLMClient(mode="local")

        request = CompletionRequest(
            question="What is 2+2?",
            reasoning="Test reasoning",
            model_type="olmo--base",
        )

        extra_body = client._get_extra_body(request)

        # Base model 不應該有 reasoning.enabled flag
        if extra_body:
            assert "reasoning" not in extra_body


class TestOLMoBaseChatAPIDisabled:
    """Test that chat API is disabled for OLMo base"""

    def test_olmo_base_generate_not_supported(self):
        """Test that generate() raises exception for olmo--base"""
        from llm_client import Request
        from unittest.mock import MagicMock

        # Mock the client to avoid actual server connection
        client = LLMClient(mode="local")
        client._get_model = MagicMock(return_value="mock-model")

        request = Request(
            queries=["test question"],
            model_type="olmo--base",
        )

        with pytest.raises(Exception):
            client.generate(request)
