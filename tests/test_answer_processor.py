"""
Tests for AnswerProcessor

Tests the answer processor functionality which adds prefill text
to guide model answer generation.
"""

import pytest
from processors import AnswerProcessor
from pipeline import parse_flow, Pipeline
from core import build_gpt_oss_prompt_with_reasoning


class TestAnswerProcessor:
    """Tests for AnswerProcessor"""

    def test_answer_processor_retrieval_mode(self, sample_context):
        """Test AnswerProcessor with mode='retrieval' adds prefill to context"""
        processor = AnswerProcessor(mode='retrieval')
        reasoning = "Let's solve: 2 + 2 = 4"

        result = processor.process(reasoning, sample_context)

        # Processor should not modify reasoning
        assert result == reasoning

        # Processor should add answer_prefill to context
        assert 'answer_prefill' in sample_context
        assert sample_context['answer_prefill'] == "Thus, the answer is"

    def test_answer_processor_custom_prefill_text(self, sample_context):
        """Test AnswerProcessor auto-determines prefill text (custom param deprecated)"""
        # Custom prefill_text parameter is deprecated and ignored
        # Prefill text is now auto-determined based on dataset_type
        custom_text = "Therefore, the final answer is"
        processor = AnswerProcessor(mode='retrieval', prefill_text=custom_text)
        reasoning = "Test reasoning"
        sample_context['dataset_type'] = 'math'

        result = processor.process(reasoning, sample_context)

        assert result == reasoning
        # Should use auto-determined prefill, not the custom text
        assert sample_context['answer_prefill'] == "Thus, the answer is"
        assert sample_context['max_tokens'] == 50

    def test_answer_processor_get_metadata(self, sample_context):
        """Test AnswerProcessor.get_metadata()"""
        processor = AnswerProcessor(mode='retrieval')
        reasoning = "Test text"

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'answer'
        assert metadata['mode'] == 'retrieval'
        assert metadata['prefill_text'] == "Thus, the answer is"
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_answer_processor_invalid_mode(self, sample_context):
        """Test AnswerProcessor with invalid mode raises error"""
        processor = AnswerProcessor(mode='invalid_mode')
        reasoning = "Test"

        with pytest.raises(ValueError, match="Invalid mode"):
            processor.process(reasoning, sample_context)

    def test_answer_processor_in_pipeline(self, sample_context):
        """Test AnswerProcessor works in pipeline"""
        flow_str = "answer('retrieval')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Step 1: Calculate 2 + 2 = 4"
        context = sample_context.copy()
        result, metadata_list = pipeline.execute(reasoning, context)

        # Reasoning should be unchanged
        assert result == reasoning

        # Context should have answer_prefill
        assert context['answer_prefill'] == "Thus, the answer is"

        # Metadata should be recorded
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'answer'

    def test_answer_processor_combined_with_mask(self, sample_context):
        """Test AnswerProcessor combined with MaskProcessor"""
        flow_str = "mask('number'),answer('retrieval')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Calculate: 2 + 2 = 4"
        context = sample_context.copy()
        result, metadata_list = pipeline.execute(reasoning, context)

        # Numbers should be masked
        assert '2' not in result
        assert '4' not in result
        assert '█' in result

        # Answer prefill should be set
        assert context['answer_prefill'] == "Thus, the answer is"

        # Both processors should be in metadata
        assert len(metadata_list) == 2
        assert metadata_list[0]['processor'] == 'mask'
        assert metadata_list[1]['processor'] == 'answer'


class TestAnswerPrefillPromptBuilding:
    """Tests for prompt building with answer prefill"""

    def test_build_prompt_with_answer_prefill(self):
        """Test build_gpt_oss_prompt_with_reasoning_prefilled_answer"""
        from core import build_gpt_oss_prompt_with_reasoning_prefilled_answer

        question = "What is 2 + 2?"
        reasoning = "Let me calculate: 2 + 2 = 4"
        prefill_text = "Thus, the answer is"

        prompt = build_gpt_oss_prompt_with_reasoning_prefilled_answer(
            question, reasoning, prefill_text
        )

        # Prompt should contain the prefill text in the final channel
        assert '<|channel|>final<|message|>' in prompt
        assert f'<|channel|>final<|message|>{prefill_text}' in prompt

        # Prompt should contain question and reasoning
        assert question in prompt
        assert reasoning in prompt

    def test_build_prompt_default_prefill(self):
        """Test build_gpt_oss_prompt_with_reasoning_prefilled_answer with default prefill"""
        from core import build_gpt_oss_prompt_with_reasoning_prefilled_answer

        question = "What is 3 × 4?"
        reasoning = "Multiply: 3 × 4 = 12"

        prompt = build_gpt_oss_prompt_with_reasoning_prefilled_answer(
            question, reasoning
        )

        # Should use default prefill text
        assert '<|channel|>final<|message|>Thus, the answer is' in prompt

    def test_build_prompt_custom_prefill(self):
        """Test build_gpt_oss_prompt_with_reasoning_prefilled_answer with custom text"""
        from core import build_gpt_oss_prompt_with_reasoning_prefilled_answer

        question = "What is 5 - 3?"
        reasoning = "Subtract: 5 - 3 = 2"
        custom_prefill = "The final answer is"

        prompt = build_gpt_oss_prompt_with_reasoning_prefilled_answer(
            question, reasoning, prefill_text=custom_prefill
        )

        assert f'<|channel|>final<|message|>{custom_prefill}' in prompt

    def test_build_prompt_empty_prefill(self):
        """Test build_gpt_oss_prompt_with_reasoning_prefilled_answer with empty prefill"""
        from core import build_gpt_oss_prompt_with_reasoning_prefilled_answer

        question = "What is 10 / 2?"
        reasoning = "Divide: 10 / 2 = 5"

        prompt = build_gpt_oss_prompt_with_reasoning_prefilled_answer(
            question, reasoning, prefill_text=""
        )

        # Should have final channel tag but no prefill text
        assert '<|channel|>final<|message|>' in prompt
        # Verify the tag appears exactly once and ends immediately
        assert prompt.endswith('<|channel|>final<|message|>')


class TestAnswerProcessorIntegration:
    """Integration tests for answer processor with prepare_task"""

    def test_prepare_task_uses_answer_prefill(self):
        """Test that prepare_task uses answer_prefill from context"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-001',
            'question': 'What is 2 + 2?',
            'answer': '4',
            'result': {'traj': 'Calculate: 2 + 2 = 4'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow)

        # Task metadata should contain answer_prefill
        assert 'answer_prefill' in task.metadata
        assert task.metadata['answer_prefill'] == "Thus, the answer is"

        # CompletionRequest should use prefilled answer
        assert task.request.answer_prefix == "Thus, the answer is"
        assert task.request.question == 'What is 2 + 2?'
        assert 'Calculate' in task.request.reasoning

    def test_prepare_task_without_answer_processor(self):
        """Test that prepare_task works normally without answer processor"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-002',
            'question': 'What is 3 × 3?',
            'answer': '9',
            'result': {'traj': 'Multiply: 3 × 3 = 9'}
        }

        flow = "mask('number')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow)

        # Task metadata should NOT contain answer_prefill
        assert 'answer_prefill' not in task.metadata

        # CompletionRequest should use empty answer_prefix (no prefill)
        assert task.request.answer_prefix == ""
        assert task.request.question == 'What is 3 × 3?'
        assert 'Multiply' in task.request.reasoning

    def test_prepare_task_max_tokens_with_retrieval(self):
        """Test that prepare_task reduces max_tokens to 50 when using answer('retrieval')"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-003',
            'question': 'What is 5 + 5?',
            'answer': '10',
            'result': {'traj': 'Add: 5 + 5 = 10'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow)

        # When using retrieval mode, max_tokens should be reduced to 50
        assert task.request.max_tokens == 50
        assert task.request.answer_prefix == "Thus, the answer is"

    def test_prepare_task_max_tokens_without_retrieval(self):
        """Test that prepare_task keeps max_tokens at 5000 without answer processor"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-004',
            'question': 'What is 7 - 3?',
            'answer': '4',
            'result': {'traj': 'Subtract: 7 - 3 = 4'}
        }

        flow = "mask('number')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow)

        # Without retrieval mode, max_tokens should remain 5000
        assert task.request.max_tokens == 5000
        assert task.request.answer_prefix == ""

    def test_prepare_task_max_tokens_with_custom_prefill(self):
        """Test that custom prefill parameter is ignored (now auto-determined by dataset_type)"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-005',
            'question': 'What is 12 / 3?',
            'answer': '4',
            'result': {'traj': 'Divide: 12 / 3 = 4'}
        }

        # Custom prefill_text parameter is deprecated and ignored
        # Prefill text is now auto-determined based on dataset_type
        flow = "answer('retrieval',prefill_text='The final answer is')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # Should use auto-determined prefill (not the custom text) and max_tokens should be 50
        assert task.request.max_tokens == 50
        assert task.request.answer_prefix == "Thus, the answer is"  # Auto-determined, not custom
