"""
Integration tests for run_experiment.py

These tests verify that the new Pipeline-based architecture works correctly
while maintaining backward compatibility with legacy parameters.
"""

import pytest


class TestLegacyParamsToFlow:
    """Tests for legacy_params_to_flow function"""

    def test_mask_only(self):
        """Test converting mask_mode to flow"""
        from run_experiment import legacy_params_to_flow

        flow = legacy_params_to_flow(mask_mode='number')
        assert flow == "mask('number')"

    def test_mask_with_custom_char(self):
        """Test mask with custom character"""
        from run_experiment import legacy_params_to_flow

        flow = legacy_params_to_flow(mask_mode='number', mask_char='*')
        assert flow == "mask('number',mask_char='*')"

    def test_mask_and_shuffle(self):
        """Test mask + shuffle"""
        from run_experiment import legacy_params_to_flow

        flow = legacy_params_to_flow(mask_mode='number', shuffle=True)
        assert flow == "mask('number'),shuffle('line')"

    def test_remove_answer_and_mask(self):
        """Test remove answer + mask"""
        from run_experiment import legacy_params_to_flow

        flow = legacy_params_to_flow(remove_answer_after=True, mask_mode='number')
        assert flow == "truncate('answer_and_after'),mask('number')"

    def test_full_pipeline(self):
        """Test complete pipeline"""
        from run_experiment import legacy_params_to_flow

        flow = legacy_params_to_flow(
            remove_answer_after=True,
            mask_mode='number',
            shuffle=True
        )
        assert flow == "truncate('answer_and_after'),mask('number'),shuffle('line')"

    def test_nlines_mode(self):
        """Test n-lines mode with num_prev_lines"""
        from run_experiment import legacy_params_to_flow

        flow = legacy_params_to_flow(mask_mode='n-lines', num_prev_lines=2)
        assert flow == "mask('n-lines',num_prev_lines=2)"

    def test_empty_flow(self):
        """Test with no operations"""
        from run_experiment import legacy_params_to_flow

        flow = legacy_params_to_flow()
        assert flow == ""


class TestPrepareTask:
    """Tests for prepare_task function"""

    def test_with_flow_parameter(self):
        """Test using flow parameter"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-1',
            'question': 'What is 2+2?',
            'answer': '4',
            'result': {
                'traj': 'Let me calculate: 2 + 2 = 4. The answer is 4.'
            }
        }

        task = prepare_task(
            item,
            model_type='gpt-oss',
            flow="mask('number')"
        )

        assert task.metadata['unique_id'] == 'test-1'
        assert task.metadata['flow'] == "mask('number')"
        assert '2' not in task.metadata['processed_reasoning']
        assert '4' not in task.metadata['processed_reasoning']
        assert '█' in task.metadata['processed_reasoning']

    def test_with_legacy_parameters(self):
        """Test using legacy parameters"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-1',
            'question': 'What is 2+2?',
            'answer': '4',
            'result': {
                'traj': 'Let me calculate: 2 + 2 = 4. The answer is 4.'
            }
        }

        task = prepare_task(
            item,
            model_type='gpt-oss',
            mask_mode='number'
        )

        assert task.metadata['flow'] == "mask('number')"
        assert '2' not in task.metadata['processed_reasoning']
        assert '4' not in task.metadata['processed_reasoning']

    def test_flow_takes_precedence(self):
        """Test that flow parameter takes precedence over legacy params"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-1',
            'question': 'What is 2+2?',
            'answer': '4',
            'result': {
                'traj': 'Let me calculate: 2 + 2 = 4. The answer is 4.'
            }
        }

        task = prepare_task(
            item,
            model_type='gpt-oss',
            flow="mask('answer')",  # Flow says mask only answer
            mask_mode='number'  # Legacy says mask all numbers
        )

        # Flow should take precedence - only answer (4) should be masked
        assert task.metadata['flow'] == "mask('answer')"
        assert '2' in task.metadata['processed_reasoning']  # 2 should NOT be masked
        assert '4' not in task.metadata['processed_reasoning']  # 4 SHOULD be masked

    def test_processing_metadata_recorded(self):
        """Test that processing metadata is recorded"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-1',
            'question': 'What is 2+2?',
            'answer': '4',
            'result': {
                'traj': 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5'
            }
        }

        task = prepare_task(
            item,
            model_type='gpt-oss',
            flow="truncate('last_n_lines',n=2),mask('number')"
        )

        metadata = task.metadata['processing_metadata']
        assert len(metadata) == 2
        assert metadata[0]['processor'] == 'truncate'
        assert metadata[0]['removed_lines'] == 2
        assert metadata[1]['processor'] == 'mask'

    def test_multiline_reasoning_with_shuffle(self):
        """Test processing multiline reasoning with shuffle"""
        from run_experiment import prepare_task

        item = {
            'unique_id': 'test-1',
            'question': 'Question?',
            'answer': '100',
            'result': {
                'traj': 'Step 1: Calculate 10\nStep 2: Calculate 20\nStep 3: Calculate 30'
            }
        }

        task = prepare_task(
            item,
            model_type='gpt-oss',
            flow="mask('number'),shuffle('line',seed=42)"
        )

        # Numbers should be masked
        assert '10' not in task.metadata['processed_reasoning']
        assert '20' not in task.metadata['processed_reasoning']
        assert '30' not in task.metadata['processed_reasoning']

        # Should have 3 lines (shuffled)
        lines = [l for l in task.metadata['processed_reasoning'].split('\n') if l.strip()]
        assert len(lines) == 3
