"""
Tests for dataset type detection and code-specific answer prefix handling

Tests the functionality that detects whether a dataset is 'math' or 'code',
and automatically adjusts answer prefix and max_tokens for code datasets.
"""

import pytest
from run_experiment import detect_dataset_type, prepare_task


class TestDatasetTypeDetection:
    """Tests for detect_dataset_type function"""

    def test_detect_math_dataset_from_aime_id(self):
        """Test detection of math dataset from AIME unique_id"""
        results = [
            {'unique_id': 'aime2025-I-0-1', 'question': 'Math problem'},
            {'unique_id': 'aime2025-I-0-2', 'question': 'Math problem'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'

    def test_detect_code_dataset_from_codeelo_id(self):
        """Test detection of code dataset from codeelo unique_id"""
        results = [
            {'unique_id': 'codeelo-1234-A-0', 'question': 'Code problem'},
            {'unique_id': 'codeelo-1234-B-0', 'question': 'Code problem'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'code'

    def test_detect_math_dataset_from_math500_id(self):
        """Test detection of math dataset from MATH500 unique_id"""
        results = [
            {'unique_id': 'math500-algebra-001', 'question': 'Math problem'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'

    def test_detect_empty_dataset_defaults_to_math(self):
        """Test that empty dataset defaults to math"""
        results = []

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'

    def test_detect_unknown_dataset_defaults_to_math(self):
        """Test that unknown dataset type defaults to math"""
        results = [
            {'unique_id': 'unknown-dataset-001', 'question': 'Problem'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'


class TestPrepareTaskWithDatasetType:
    """Tests for prepare_task function with dataset_type parameter"""

    def test_prepare_task_math_dataset_with_retrieval(self):
        """Test prepare_task for math dataset with answer('retrieval')"""
        item = {
            'unique_id': 'aime2025-I-0-1',
            'question': 'What is 2 + 2?',
            'answer': '4',
            'result': {'traj': 'Calculate: 2 + 2 = 4'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # Math dataset should use default prefix
        assert task.request.answer_prefix == "Thus, the answer is"

        # Math dataset should use reduced max_tokens (50)
        assert task.request.max_tokens == 50

        # Metadata should contain answer_prefill
        assert 'answer_prefill' in task.metadata
        assert task.metadata['answer_prefill'] == "Thus, the answer is"

    def test_prepare_task_code_dataset_with_retrieval(self):
        """Test prepare_task for code dataset with answer('retrieval')"""
        item = {
            'unique_id': 'codeelo-1234-A-0',
            'question': 'Write a function to add two numbers',
            'test_cases': [(['2', '3'], '5'), (['10', '20'], '30')],
            'result': {'traj': 'Let me think about the solution...'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='code')

        # Code dataset should use code-specific prefix
        assert task.request.answer_prefix == "Thus, the code is\n```cpp\n"

        # Code dataset should use full max_tokens (5000)
        assert task.request.max_tokens == 5000

        # Metadata should contain test_cases
        assert 'test_cases' in task.metadata
        assert len(task.metadata['test_cases']) == 2

    def test_prepare_task_code_dataset_with_custom_prefill(self):
        """Test that code dataset ignores custom prefill_text and uses code prefix"""
        item = {
            'unique_id': 'codeelo-1234-B-0',
            'question': 'Write a function',
            'test_cases': [(['1'], '1')],
            'result': {'traj': 'Solution approach...'}
        }

        flow = "answer('retrieval',prefill_text='Custom prefix')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='code')

        # Code dataset should override custom prefix with code prefix
        assert task.request.answer_prefix == "Thus, the code is\n```cpp\n"

        # max_tokens should still be 5000 for code
        assert task.request.max_tokens == 5000

    def test_prepare_task_math_dataset_with_custom_prefill(self):
        """Test that custom prefill_text is ignored (auto-determined by dataset_type)"""
        item = {
            'unique_id': 'aime2025-I-0-2',
            'question': 'What is 3 × 4?',
            'answer': '12',
            'result': {'traj': 'Multiply: 3 × 4 = 12'}
        }

        # Custom prefill_text parameter is deprecated and ignored
        custom_prefill = "Therefore, the final answer is"
        flow = f"answer('retrieval',prefill_text='{custom_prefill}')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # Should use auto-determined prefix for math, not custom
        assert task.request.answer_prefix == "Thus, the answer is"

        # max_tokens should still be 50 for math with retrieval
        assert task.request.max_tokens == 50

    def test_prepare_task_code_dataset_without_retrieval(self):
        """Test prepare_task for code dataset without answer('retrieval')"""
        item = {
            'unique_id': 'codeelo-1234-C-0',
            'question': 'Implement a sorting function',
            'test_cases': [(['3', '1', '2'], '1 2 3')],
            'result': {'traj': 'Use merge sort...'}
        }

        flow = "mask('number')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='code')

        # Without retrieval, answer_prefix should be empty
        assert task.request.answer_prefix == ""

        # max_tokens should be 5000 (default)
        assert task.request.max_tokens == 5000

        # Metadata should NOT contain answer_prefill
        assert 'answer_prefill' not in task.metadata

    def test_prepare_task_math_dataset_without_retrieval(self):
        """Test prepare_task for math dataset without answer('retrieval')"""
        item = {
            'unique_id': 'aime2025-I-0-3',
            'question': 'Calculate 10 / 2',
            'answer': '5',
            'result': {'traj': 'Divide: 10 / 2 = 5'}
        }

        flow = "mask('number')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='math')

        # Without retrieval, answer_prefix should be empty
        assert task.request.answer_prefix == ""

        # max_tokens should be 5000 (default)
        assert task.request.max_tokens == 5000

    def test_prepare_task_default_dataset_type(self):
        """Test that prepare_task defaults to math when dataset_type not specified"""
        item = {
            'unique_id': 'test-001',
            'question': 'What is 5 + 5?',
            'answer': '10',
            'result': {'traj': 'Add: 5 + 5 = 10'}
        }

        flow = "answer('retrieval')"
        # Not specifying dataset_type, should default to 'math'
        task = prepare_task(item, model_type='gpt-oss', flow=flow)

        # Should behave like math dataset
        assert task.request.answer_prefix == "Thus, the answer is"
        assert task.request.max_tokens == 50


class TestAnswerPrefixIntegration:
    """Integration tests for answer prefix with different dataset types"""

    def test_pipeline_preserves_code_prefix(self):
        """Test that pipeline execution preserves code prefix through processing"""
        from pipeline import parse_flow, Pipeline

        item = {
            'unique_id': 'codeelo-1234-D-0',
            'question': 'Write a function',
            'test_cases': [(['1', '2'], '3')],
            'result': {'traj': 'Implementation strategy...\nStep 1: Initialize variables\nStep 2: Process input'}
        }

        flow = "mask('number'),answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='code')

        # Verify code prefix is applied
        assert task.request.answer_prefix == "Thus, the code is\n```cpp\n"
        assert task.request.max_tokens == 5000

        # Verify processing metadata
        assert 'processing_metadata' in task.metadata
        assert len(task.metadata['processing_metadata']) == 2
        assert task.metadata['processing_metadata'][0]['processor'] == 'mask'
        assert task.metadata['processing_metadata'][1]['processor'] == 'answer'

    def test_multiple_processors_with_code_retrieval(self):
        """Test complex pipeline with code dataset and retrieval"""
        item = {
            'unique_id': 'codeelo-5678-E-0',
            'question': 'Implement binary search',
            'test_cases': [(['1', '2', '3', '4', '5'], '3')],
            'result': {'traj': 'Binary search implementation:\nStep 1: Initialize pointers\nStep 2: Compare middle element'}
        }

        flow = "truncate('last_ratio',ratio=0.3),mask('number'),shuffle('line'),answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='code')

        # Code prefix should still be applied after all transformations
        assert task.request.answer_prefix == "Thus, the code is\n```cpp\n"
        assert task.request.max_tokens == 5000

        # All processors should be recorded
        assert len(task.metadata['processing_metadata']) == 4

    def test_mixed_dataset_detection_in_batch(self):
        """Test that dataset type detection works correctly in batch processing"""
        # Math dataset items
        math_items = [
            {'unique_id': 'aime2025-I-0-1', 'question': 'Math Q1'},
            {'unique_id': 'aime2025-I-0-2', 'question': 'Math Q2'}
        ]

        # Code dataset items
        code_items = [
            {'unique_id': 'codeelo-1234-A-0', 'question': 'Code Q1'},
            {'unique_id': 'codeelo-1234-B-0', 'question': 'Code Q2'}
        ]

        # Test math detection
        math_type = detect_dataset_type(math_items)
        assert math_type == 'math'

        # Test code detection
        code_type = detect_dataset_type(code_items)
        assert code_type == 'code'


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_prepare_task_with_empty_flow(self):
        """Test prepare_task with empty flow string"""
        item = {
            'unique_id': 'test-001',
            'question': 'Test question',
            'answer': '42',
            'result': {'traj': 'Test reasoning'}
        }

        task = prepare_task(item, model_type='gpt-oss', flow="", dataset_type='math')

        # With empty flow, no processing should occur
        assert task.request.answer_prefix == ""
        assert task.request.max_tokens == 5000
        assert task.metadata['processing_metadata'] == []

    def test_prepare_task_code_dataset_with_none_flow(self):
        """Test prepare_task for code dataset with None flow"""
        item = {
            'unique_id': 'codeelo-9999-Z-0',
            'question': 'Code problem',
            'test_cases': [(['input'], 'output')],
            'result': {'traj': 'Solution...'}
        }

        task = prepare_task(item, model_type='gpt-oss', flow=None, dataset_type='code')

        # With None flow, no processing
        assert task.request.answer_prefix == ""
        assert task.request.max_tokens == 5000
        assert 'test_cases' in task.metadata

    def test_dataset_detection_with_missing_unique_id(self):
        """Test dataset type detection when unique_id is missing"""
        results = [
            {'question': 'Problem without unique_id'}
        ]

        # Should default to 'math' when unique_id is missing
        dataset_type = detect_dataset_type(results)
        assert dataset_type == 'math'

    def test_prepare_task_preserves_test_cases(self):
        """Test that prepare_task preserves test_cases for code datasets"""
        test_cases = [
            (['1', '2'], '3'),
            (['10', '20'], '30'),
            (['100', '200'], '300')
        ]

        item = {
            'unique_id': 'codeelo-1111-A-0',
            'question': 'Sum two numbers',
            'test_cases': test_cases,
            'result': {'traj': 'Implementation...'}
        }

        flow = "answer('retrieval')"
        task = prepare_task(item, model_type='gpt-oss', flow=flow, dataset_type='code')

        # Test cases should be preserved in metadata
        assert 'test_cases' in task.metadata
        assert task.metadata['test_cases'] == test_cases
        assert len(task.metadata['test_cases']) == 3
