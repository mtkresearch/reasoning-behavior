"""
Integration test for question('remove') processor

This test verifies that when using question('remove') processor,
the final prompt sent to LLM does not contain the question text.
"""

import pytest


def test_question_remove_prompt_integration():
    """Test that question('remove') results in empty question in prompt"""
    from pipeline import parse_flow, Pipeline
    from core import build_gpt_oss_prompt_with_reasoning

    # Setup
    original_question = "What is 2 + 2?"
    reasoning = "Let me think about this step by step."

    # Parse flow and create pipeline
    flow_str = "question('remove')"
    processors = parse_flow(flow_str)
    pipeline = Pipeline(processors)

    # Create context
    context = {
        'question': original_question,
        'answer': '4',
        'ground_truth': '4'
    }

    # Execute pipeline
    processed_reasoning, metadata_list = pipeline.execute(reasoning, context)

    # Verify context was modified
    assert context['question'] == ''
    assert metadata_list[0]['original_question'] == original_question

    # Build prompt with processed context
    final_question = context['question']
    prompt = build_gpt_oss_prompt_with_reasoning(final_question, processed_reasoning)

    # Verify original question is NOT in the prompt
    # The prompt should have empty string for question field
    assert original_question not in prompt

    # Verify the prompt has the correct structure with empty question
    assert '<|start|>user<|message|><|end|>' in prompt

    # Verify reasoning is still present
    assert reasoning in prompt


def test_question_remove_vs_normal_prompt():
    """Compare prompt with and without question('remove')"""
    from pipeline import parse_flow, Pipeline
    from core import build_gpt_oss_prompt_with_reasoning

    question = "What is 2 + 2?"
    reasoning = "Step 1: Add the numbers."

    # Normal prompt (without question removal)
    normal_prompt = build_gpt_oss_prompt_with_reasoning(question, reasoning)

    # Prompt with question removed
    flow_str = "question('remove')"
    processors = parse_flow(flow_str)
    pipeline = Pipeline(processors)

    context = {'question': question, 'answer': '4', 'ground_truth': '4'}
    processed_reasoning, _ = pipeline.execute(reasoning, context)

    removed_prompt = build_gpt_oss_prompt_with_reasoning(context['question'], processed_reasoning)

    # Verify normal prompt contains the question
    assert question in normal_prompt

    # Verify removed prompt does NOT contain the question
    assert question not in removed_prompt

    # Both should contain the reasoning
    assert reasoning in normal_prompt
    assert reasoning in removed_prompt


def test_question_remove_with_other_processors():
    """Test question('remove') combined with other processors"""
    from pipeline import parse_flow, Pipeline
    from core import build_gpt_oss_prompt_with_reasoning

    question = "Calculate 10 + 20"
    reasoning = "Step 1: 10 + 20 = 30"

    # Combine question removal with masking
    flow_str = "question('remove'),mask('number')"
    processors = parse_flow(flow_str)
    pipeline = Pipeline(processors)

    context = {'question': question, 'answer': '30', 'ground_truth': '30'}
    processed_reasoning, metadata_list = pipeline.execute(reasoning, context)

    # Verify both processors were applied
    assert len(metadata_list) == 2
    assert metadata_list[0]['processor'] == 'question'
    assert metadata_list[1]['processor'] == 'mask'

    # Build final prompt
    prompt = build_gpt_oss_prompt_with_reasoning(context['question'], processed_reasoning)

    # Question should not be in prompt
    assert question not in prompt

    # Numbers should be masked in reasoning
    assert '10' not in processed_reasoning
    assert '20' not in processed_reasoning
    assert '30' not in processed_reasoning
