"""
Tests for the answer('retrieval') refactoring

Tests the simplified answer processor mechanism including:
1. _build_generated_answer helper function
2. AnswerProcessor max_tokens configuration
3. Integration with prepare_task
"""

import pytest
from processors import AnswerProcessor
from pipeline import parse_flow, Pipeline
from run_experiment import _build_generated_answer, prepare_task


class TestBuildGeneratedAnswer:
    """Tests for _build_generated_answer helper function"""

    def test_build_generated_answer_with_prefix(self):
        """Test _build_generated_answer prepends answer_prefix when present"""
        metadata = {'answer_prefill': 'Thus, the answer is'}
        response_content = '42'
        result = _build_generated_answer(response_content, metadata)
        assert result == 'Thus, the answer is 42'

    def test_build_generated_answer_without_prefix(self):
        """Test _build_generated_answer returns content unchanged when no prefix"""
        metadata = {}
        response_content = '42'
        result = _build_generated_answer(response_content, metadata)
        assert result == '42'

    def test_build_generated_answer_with_leading_spaces(self):
        """Test _build_generated_answer lstrips response content"""
        metadata = {'answer_prefill': 'Answer:'}
        response_content = '   A'
        result = _build_generated_answer(response_content, metadata)
        assert result == 'Answer: A'

    def test_build_generated_answer_with_multiple_leading_spaces(self):
        """Test _build_generated_answer lstrips multiple leading spaces"""
        metadata = {'answer_prefill': 'Thus, the answer is'}
        response_content = '\n\n\n42\n'
        result = _build_generated_answer(response_content, metadata)
        assert result == 'Thus, the answer is 42\n'

    def test_build_generated_answer_with_newline_prefix(self):
        """Test _build_generated_answer with newline in prefix"""
        metadata = {'answer_prefill': 'Thus, the code is\n```cpp\n'}
        response_content = 'int x = 5;\n```'
        result = _build_generated_answer(response_content, metadata)
        assert result == 'Thus, the code is\n```cpp\n int x = 5;\n```'

    def test_build_generated_answer_empty_response(self):
        """Test _build_generated_answer with empty response"""
        metadata = {'answer_prefill': 'Thus, the answer is'}
        response_content = ''
        result = _build_generated_answer(response_content, metadata)
        assert result == 'Thus, the answer is '

    def test_build_generated_answer_with_none_metadata(self):
        """Test _build_generated_answer handles missing answer_prefill gracefully"""
        metadata = {'other_key': 'value'}
        response_content = '123'
        result = _build_generated_answer(response_content, metadata)
        assert result == '123'


class TestAnswerProcessorMaxTokens:
    """Tests for AnswerProcessor max_tokens configuration"""

    def test_answer_processor_sets_max_tokens_math(self, sample_context):
        """Test AnswerProcessor sets max_tokens=50 for math dataset"""
        processor = AnswerProcessor(mode='retrieval')
        context = sample_context.copy()
        context['dataset_type'] = 'math'

        processor.process('reasoning text', context)

        assert context['max_tokens'] == 50
        assert processor.max_tokens == 50
        assert context['answer_prefill'] == 'Thus, the answer is'

    def test_answer_processor_sets_max_tokens_code(self, sample_context):
        """Test AnswerProcessor sets max_tokens=5000 for code dataset"""
        processor = AnswerProcessor(mode='retrieval')
        context = sample_context.copy()
        context['dataset_type'] = 'code'

        processor.process('reasoning text', context)

        assert context['max_tokens'] == 5000
        assert processor.max_tokens == 5000
        assert context['answer_prefill'] == 'Thus, the code is\n```cpp\n'

    def test_answer_processor_sets_max_tokens_science(self, sample_context):
        """Test AnswerProcessor sets max_tokens=50 for science dataset"""
        processor = AnswerProcessor(mode='retrieval')
        context = sample_context.copy()
        context['dataset_type'] = 'science'

        processor.process('reasoning text', context)

        assert context['max_tokens'] == 50
        assert processor.max_tokens == 50
        assert context['answer_prefill'] == 'Answer:'

    def test_answer_processor_default_math_dataset(self, sample_context):
        """Test AnswerProcessor defaults to math dataset when not specified"""
        processor = AnswerProcessor(mode='retrieval')
        context = sample_context.copy()
        # Don't set dataset_type

        processor.process('reasoning text', context)

        assert context['max_tokens'] == 50
        assert processor.max_tokens == 50
        assert context['answer_prefill'] == 'Thus, the answer is'

    def test_answer_processor_metadata_includes_max_tokens(self, sample_context):
        """Test AnswerProcessor.get_metadata includes max_tokens"""
        processor = AnswerProcessor(mode='retrieval')
        context = sample_context.copy()
        context['dataset_type'] = 'math'

        processor.process('reasoning text', context)
        metadata = processor.get_metadata()

        assert 'max_tokens' in metadata
        assert metadata['max_tokens'] == 50
        assert metadata['processor'] == 'answer'
        assert metadata['mode'] == 'retrieval'
        assert metadata['prefill_text'] == 'Thus, the answer is'

    def test_answer_processor_in_pipeline_sets_context_max_tokens(self, sample_context):
        """Test AnswerProcessor in pipeline sets max_tokens in context"""
        flow_str = "answer('retrieval')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        context = sample_context.copy()
        context['dataset_type'] = 'math'
        result, metadata_list = pipeline.execute('reasoning text', context)

        assert context['max_tokens'] == 50
        assert context['answer_prefill'] == 'Thus, the answer is'
        assert metadata_list[0]['max_tokens'] == 50


class TestPrepareTaskMaxTokensSimplification:
    """Tests for simplified max_tokens logic in prepare_task"""

    def test_prepare_task_gets_max_tokens_from_context(self):
        """Test prepare_task gets max_tokens from context set by AnswerProcessor"""
        item = {
            'unique_id': 'test-001',
            'question': 'What is 2 + 2?',
            'answer': '4',
            'result': {'traj': 'Calculate: 2 + 2 = 4'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # max_tokens should be 50 (set by AnswerProcessor)
        assert task.request.max_tokens == 50

    def test_prepare_task_max_tokens_code_dataset(self):
        """Test prepare_task gets max_tokens=5000 for code dataset"""
        item = {
            'unique_id': 'code-001',
            'question': 'Write a function',
            'answer': 'code',
            'result': {'traj': 'Let me write the code'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='code')

        # max_tokens should be 5000 (set by AnswerProcessor for code)
        assert task.request.max_tokens == 5000

    def test_prepare_task_max_tokens_default_without_answer_processor(self):
        """Test prepare_task defaults to 5000 when answer processor not used"""
        item = {
            'unique_id': 'test-002',
            'question': 'What is 3 + 3?',
            'answer': '6',
            'result': {'traj': 'Calculate: 3 + 3 = 6'}
        }

        flow = "mask('number')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # max_tokens should default to 5000 (no AnswerProcessor)
        assert task.request.max_tokens == 5000

    def test_prepare_task_metadata_includes_max_tokens(self):
        """Test prepare_task metadata includes max_tokens from AnswerProcessor"""
        item = {
            'unique_id': 'test-003',
            'question': 'Solve the equation',
            'answer': '10',
            'result': {'traj': 'Step 1... Answer: 10'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # Metadata should include processing_metadata with AnswerProcessor info
        processing_metadata = task.metadata.get('processing_metadata', [])
        assert len(processing_metadata) > 0
        answer_processor_meta = next(
            (m for m in processing_metadata if m.get('processor') == 'answer'),
            None
        )
        assert answer_processor_meta is not None
        assert answer_processor_meta['max_tokens'] == 50


class TestAnswerProcessorRefactorIntegration:
    """Integration tests for the refactored answer processor"""

    def test_full_flow_answer_prefix_concatenation(self):
        """Test full flow: AnswerProcessor -> prepare_task -> answer concatenation"""
        item = {
            'unique_id': 'integration-001',
            'question': 'What is 5 + 5?',
            'answer': '10',
            'result': {'traj': 'Adding 5 + 5 = 10'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # Verify configuration
        assert task.request.answer_prefix == 'Thus, the answer is'
        assert task.request.max_tokens == 50
        assert task.metadata['answer_prefill'] == 'Thus, the answer is'

        # Simulate response content
        response_content = '10'
        generated_answer = _build_generated_answer(response_content, task.metadata)
        assert generated_answer == 'Thus, the answer is 10'

    def test_combined_processors_answer_and_mask(self):
        """Test answer processor combined with mask"""
        item = {
            'unique_id': 'combined-001',
            'question': 'Calculate 2 + 2',
            'answer': '4',
            'result': {'traj': 'Step 1: 2 + 2 = 4'}
        }

        flow = "mask('number'),answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # Both processors should be applied
        assert '█' in task.request.reasoning  # From mask processor
        assert task.request.answer_prefix == 'Thus, the answer is'  # From answer processor
        assert task.request.max_tokens == 50  # From answer processor

    def test_science_dataset_configuration(self):
        """Test science dataset type configuration"""
        item = {
            'unique_id': 'science-001',
            'question': 'Which is correct?',
            'answer': 'A',
            'result': {'traj': 'Based on analysis, the answer is A'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='science')

        assert task.request.answer_prefix == 'Answer:'
        assert task.request.max_tokens == 50
        assert task.metadata['answer_prefill'] == 'Answer:'

    def test_backward_compatibility_no_flow(self):
        """Test backward compatibility when no flow is specified"""
        item = {
            'unique_id': 'no-flow-001',
            'question': 'What is 7 × 3?',
            'answer': '21',
            'result': {'traj': 'Multiply 7 × 3 = 21'}
        }

        # No flow specified
        task = prepare_task(item, model_type='gpt-oss', flow=None, dataset_type='math')

        assert task.request.answer_prefix == ''
        assert task.request.max_tokens == 5000  # Default
        assert 'answer_prefill' not in task.metadata


class TestAnswerProcessorInvalidInput:
    """Tests for error handling in refactored answer processor"""

    def test_answer_processor_invalid_dataset_type(self, sample_context):
        """Test AnswerProcessor raises error for invalid dataset_type"""
        processor = AnswerProcessor(mode='retrieval')
        context = sample_context.copy()
        context['dataset_type'] = 'invalid_type'

        with pytest.raises(Exception):
            processor.process('reasoning text', context)

    def test_build_generated_answer_with_empty_metadata(self):
        """Test _build_generated_answer handles empty metadata dict"""
        metadata = {}
        response_content = 'test response'
        result = _build_generated_answer(response_content, metadata)
        assert result == 'test response'
