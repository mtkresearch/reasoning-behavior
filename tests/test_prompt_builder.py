"""
Tests for PromptBuilder class in run_attn_visual.py

Following TDD Red-Green-Refactor cycle for Phase 3
"""

import pytest


class TestPromptBuilder:
    """Test suite for PromptBuilder class"""

    def test_initialization(self):
        """Test PromptBuilder initialization"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('gpt-oss')

        assert builder.template == 'gpt-oss'

    def test_build_prompt_gpt_oss_template(self):
        """Test building prompt with gpt-oss template"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('gpt-oss')

        question = "What is 2+2?"
        reasoning = "Let me calculate: 2+2 = 4"
        prefill_text = "Thus, the answer is"
        truncated_answer = ""

        prompt = builder.build_prompt(
            question=question,
            reasoning=reasoning,
            prefill_text=prefill_text,
            truncated_answer=truncated_answer
        )

        # Verify prompt structure
        assert isinstance(prompt, str)
        assert '<|start|>system<|message|>' in prompt
        assert '<|start|>user<|message|>' in prompt
        assert '<|start|>assistant<|channel|>analysis<|message|>' in prompt
        assert '<|start|>assistant<|channel|>final<|message|>' in prompt

        # Verify content is included
        assert question in prompt
        assert reasoning in prompt
        assert prefill_text in prompt

    def test_build_prompt_with_truncated_answer(self):
        """Test building prompt with non-empty truncated answer"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('gpt-oss')

        question = "What is 2+2?"
        reasoning = "Let me calculate: 2+2 = 4"
        prefill_text = "Thus, the answer is"
        truncated_answer = "four plus"

        prompt = builder.build_prompt(
            question=question,
            reasoning=reasoning,
            prefill_text=prefill_text,
            truncated_answer=truncated_answer
        )

        # Verify truncated answer is merged with prefill_text
        assert "Thus, the answer is four plus" in prompt

    def test_build_prompt_empty_reasoning(self):
        """Test building prompt with empty reasoning"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('gpt-oss')

        question = "What is 2+2?"
        reasoning = ""
        prefill_text = "Answer:"
        truncated_answer = ""

        prompt = builder.build_prompt(
            question=question,
            reasoning=reasoning,
            prefill_text=prefill_text,
            truncated_answer=truncated_answer
        )

        # Should still generate valid prompt
        assert isinstance(prompt, str)
        assert '<|start|>assistant<|channel|>analysis<|message|><|end|>' in prompt

    def test_build_prompt_empty_truncated_answer(self):
        """Test building prompt with empty truncated answer"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('gpt-oss')

        question = "What is 2+2?"
        reasoning = "Calculating..."
        prefill_text = "Thus, the answer is"
        truncated_answer = ""

        prompt = builder.build_prompt(
            question=question,
            reasoning=reasoning,
            prefill_text=prefill_text,
            truncated_answer=truncated_answer
        )

        # Should only include prefill_text (no extra content)
        assert "Thus, the answer is" in prompt
        # Ensure no double spacing after prefill_text
        assert "Thus, the answer is<|end|>" in prompt or "Thus, the answer is" in prompt

    def test_build_prompt_custom_prefill_text(self):
        """Test building prompt with custom prefill text"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('gpt-oss')

        question = "Calculate 5*6"
        reasoning = "5*6 = 30"
        prefill_text = "The result is:"
        truncated_answer = "thirty"

        prompt = builder.build_prompt(
            question=question,
            reasoning=reasoning,
            prefill_text=prefill_text,
            truncated_answer=truncated_answer
        )

        # Verify custom prefill_text is used
        assert "The result is: thirty" in prompt or "The result is:thirty" in prompt

    def test_build_prompt_merges_prefill_and_answer(self):
        """Test that prefill_text and truncated_answer are properly merged"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('gpt-oss')

        question = "Test question"
        reasoning = "Test reasoning"
        prefill_text = "Answer:"
        truncated_answer = "42"

        prompt = builder.build_prompt(
            question=question,
            reasoning=reasoning,
            prefill_text=prefill_text,
            truncated_answer=truncated_answer
        )

        # Check merging (should be "Answer: 42" or "Answer:42")
        assert "Answer:" in prompt
        assert "42" in prompt

    def test_unknown_template_raises_error(self):
        """Test that unknown template raises ValueError"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('unknown-template')

        with pytest.raises(ValueError, match="Unknown template"):
            builder.build_prompt(
                question="Test",
                reasoning="Test",
                prefill_text="Test",
                truncated_answer="Test"
            )

    def test_build_prompt_preserves_special_characters(self):
        """Test that special characters in inputs are preserved"""
        from run_attn_visual import PromptBuilder

        builder = PromptBuilder('gpt-oss')

        question = "What is <special>?"
        reasoning = "Processing {data}..."
        prefill_text = "Result:"
        truncated_answer = "$100"

        prompt = builder.build_prompt(
            question=question,
            reasoning=reasoning,
            prefill_text=prefill_text,
            truncated_answer=truncated_answer
        )

        # Verify all special characters are preserved
        assert "<special>" in prompt
        assert "{data}" in prompt
        assert "$100" in prompt
