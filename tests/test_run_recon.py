#!/usr/bin/env python3
"""
Tests for run_recon.py - Reconstruction experiment
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_recon import (
    extract_reconstructed_reasoning,
    build_reconstruction_prompt,
    append_recon_to_flow,
    load_input_results,
)


class TestExtractReconstructedReasoning:
    """Test extraction of reconstructed reasoning from model response"""

    def test_extract_from_txt_code_block(self):
        """Should extract content from ```txt code block"""
        response = """Here is the reconstruction:
```txt
This is the reconstructed reasoning.
It has multiple lines.
```
Some extra text after.
"""
        result = extract_reconstructed_reasoning(response)
        expected = "This is the reconstructed reasoning.\nIt has multiple lines."
        assert result == expected

    def test_extract_from_generic_code_block(self):
        """Should extract from generic code block if no txt block found"""
        response = """Here is the reconstruction:
```
This is the reconstructed reasoning.
Without txt marker.
```
"""
        result = extract_reconstructed_reasoning(response)
        expected = "This is the reconstructed reasoning.\nWithout txt marker."
        assert result == expected

    def test_fallback_to_full_content(self):
        """Should use full content if no code block found"""
        response = "This is the reconstructed reasoning without code blocks."
        result = extract_reconstructed_reasoning(response)
        assert result == response.strip()

    def test_empty_response(self):
        """Should handle empty response"""
        result = extract_reconstructed_reasoning("")
        assert result == ""

    def test_whitespace_handling(self):
        """Should strip leading/trailing whitespace"""
        response = """
```txt
  Content with whitespace
```
        """
        result = extract_reconstructed_reasoning(response)
        assert result == "Content with whitespace"


class TestBuildReconstructionPrompt:
    """Test reconstruction prompt building"""

    def test_basic_prompt_structure(self):
        """Should build prompt with question and broken info"""
        question = "What is 2+2?"
        broken_info = "2 + 2 = X"

        prompt = build_reconstruction_prompt(question, broken_info)

        assert question in prompt
        assert broken_info in prompt
        assert "RECONSTRUCT" in prompt.upper()
        assert "```txt" in prompt

    def test_multiline_broken_info(self):
        """Should handle multiline broken info"""
        question = "Solve the equation"
        broken_info = "Line 1\nLine 2\nLine 3"

        prompt = build_reconstruction_prompt(question, broken_info)

        assert broken_info in prompt


class TestAppendReconToFlow:
    """Test flow string manipulation"""

    def test_append_to_existing_flow(self):
        """Should append recon() to existing flow"""
        input_flow = "mask('number'),shuffle('line')"
        result = append_recon_to_flow(input_flow)
        assert result == "mask('number'),shuffle('line'),recon()"

    def test_append_to_empty_flow(self):
        """Should handle empty flow string"""
        result = append_recon_to_flow("")
        assert result == "recon()"

    def test_append_to_none_flow(self):
        """Should handle None flow"""
        result = append_recon_to_flow(None)
        assert result == "recon()"

    def test_append_flow_config(self):
        """Should append recon step to flow_config"""
        from run_recon import append_recon_to_flow_config

        input_config = [
            {'step': 1, 'processor': 'mask', 'params': {'mode': 'number'}},
            {'step': 2, 'processor': 'shuffle', 'params': {'mode': 'line'}}
        ]

        result = append_recon_to_flow_config(input_config)

        assert len(result) == 3
        assert result[2]['step'] == 3
        assert result[2]['processor'] == 'recon'
        assert result[2]['params'] == {}

    def test_append_flow_config_empty(self):
        """Should handle empty flow_config"""
        from run_recon import append_recon_to_flow_config

        result = append_recon_to_flow_config([])

        assert len(result) == 1
        assert result[0]['step'] == 1
        assert result[0]['processor'] == 'recon'


class TestLoadInputResults:
    """Test loading input results.json"""

    def test_load_valid_results(self, tmp_path):
        """Should load valid results.json"""
        # Create mock results.json
        results_data = {
            'experiment_metadata': {
                'experiment_name': 'test_exp',
                'experiment_date': '2025-01-01',
                'dataset': 'AIME2025',
                'model_type': 'gpt-oss',
                'flow': "mask('number')",
                'flow_config': [
                    {'step': 1, 'processor': 'mask', 'params': {'mode': 'number'}}
                ]
            },
            'summary': {
                'total_questions': 2,
                'correct': 1,
                'accuracy': 0.5
            },
            'results': [
                {
                    'unique_id': 'test-0',
                    'question_id': 0,
                    'question': 'What is 2+2?',
                    'ground_truth': '4',
                    'original_reasoning': 'Full reasoning here',
                    'processed_reasoning': 'X + X = X',
                    'generated_answer': '4',
                    'is_correct': True,
                    'success': True
                },
                {
                    'unique_id': 'test-1',
                    'question_id': 1,
                    'question': 'What is 3+3?',
                    'ground_truth': '6',
                    'original_reasoning': 'Another reasoning',
                    'processed_reasoning': 'Y + Y = Y',
                    'generated_answer': '7',
                    'is_correct': False,
                    'success': True
                }
            ]
        }

        results_file = tmp_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f)

        # Load results
        metadata, results = load_input_results(str(results_file))

        assert metadata['experiment_name'] == 'test_exp'
        assert metadata['flow'] == "mask('number')"
        assert len(results) == 2
        assert results[0]['unique_id'] == 'test-0'
        assert results[0]['processed_reasoning'] == 'X + X = X'

    def test_load_nonexistent_file(self):
        """Should raise error for nonexistent file"""
        with pytest.raises(FileNotFoundError):
            load_input_results('/nonexistent/path/results.json')

    def test_load_invalid_json(self, tmp_path):
        """Should raise error for invalid JSON"""
        results_file = tmp_path / "invalid.json"
        with open(results_file, 'w') as f:
            f.write("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_input_results(str(results_file))


class TestReconstructionTask:
    """Test reconstruction task preparation"""

    def test_prepare_reconstruction_task(self):
        """Should prepare task with reconstruction prompt"""
        from run_recon import prepare_reconstruction_task

        item = {
            'unique_id': 'test-0',
            'question_id': 0,
            'question': 'What is 2+2?',
            'ground_truth': '4',
            'original_reasoning': 'Full reasoning',
            'processed_reasoning': 'X + X = X'
        }

        task = prepare_reconstruction_task(item, model_type='gpt-oss')

        assert task.index == 0
        assert task.metadata['unique_id'] == 'test-0'
        assert task.metadata['question'] == 'What is 2+2?'
        assert task.metadata['processed_reasoning'] == 'X + X = X'
        assert 'X + X = X' in task.request.queries[0]  # Broken info in prompt


class TestAnswerGenerationTask:
    """Test answer generation task preparation"""

    def test_prepare_answer_generation_task(self):
        """Should prepare task with reconstructed reasoning"""
        from run_recon import prepare_answer_generation_task

        result = {
            'unique_id': 'test-0',
            'question_id': 0,
            'question': 'What is 2+2?',
            'ground_truth': '4',
            'reconstructed_reasoning': 'We need to add 2 + 2 = 4'
        }

        task = prepare_answer_generation_task(result, model_type='gpt-oss')

        assert task.index == 0
        assert task.metadata['unique_id'] == 'test-0'
        # Should use build_gpt_oss_prompt_with_reasoning
        assert 'reconstructed_reasoning' in task.metadata


class TestOutputStructure:
    """Test output results.json structure"""

    def test_output_has_required_fields(self):
        """Output should have experiment_metadata, summary, and results"""
        from run_recon import build_output_structure

        input_metadata = {
            'experiment_name': 'test',
            'dataset': 'AIME2025',
            'model_type': 'gpt-oss',
            'flow': "mask('number')",
            'flow_config': []
        }

        results = [
            {
                'unique_id': 'test-0',
                'question_id': 0,
                'is_correct': True,
                'success': True,
                'reconstruction_success': True
            }
        ]

        output = build_output_structure(input_metadata, results)

        assert 'experiment_metadata' in output
        assert 'summary' in output
        assert 'results' in output
        assert output['experiment_metadata']['flow'] == "mask('number'),recon()"
        assert len(output['results']) == 1

    def test_summary_statistics(self):
        """Summary should calculate correct statistics"""
        from run_recon import build_output_structure

        input_metadata = {
            'experiment_name': 'test',
            'dataset': 'AIME2025',
            'model_type': 'gpt-oss',
            'flow': '',
            'flow_config': []
        }

        results = [
            {'unique_id': 'test-0', 'generation_success': True, 'success': True, 'is_correct': True},
            {'unique_id': 'test-1', 'generation_success': True, 'success': True, 'is_correct': False},
            {'unique_id': 'test-2', 'generation_success': False, 'success': False, 'is_correct': False},
        ]

        output = build_output_structure(input_metadata, results)

        summary = output['summary']
        assert summary['total_questions'] == 3
        assert summary['generation_successful'] == 2
        assert summary['generation_failed'] == 1
        assert summary['grading_successful'] == 2
        assert summary['correct'] == 1
        assert summary['accuracy'] == 0.5


class TestAnswerRetrieval:
    """Test answer retrieval functionality"""

    def test_append_answer_retrieval_to_flow(self):
        """Should append answer('retrieval') to flow when enabled"""
        from run_recon import append_answer_retrieval_to_flow

        # Test with existing flow
        input_flow = "mask('number'),recon()"
        result = append_answer_retrieval_to_flow(input_flow)
        assert result == "mask('number'),recon(),answer('retrieval')"

        # Test with empty flow
        result = append_answer_retrieval_to_flow("recon()")
        assert result == "recon(),answer('retrieval')"

    def test_append_answer_retrieval_to_flow_config(self):
        """Should append answer step to flow_config"""
        from run_recon import append_answer_retrieval_to_flow_config

        input_config = [
            {'step': 1, 'processor': 'mask', 'params': {'mode': 'number'}},
            {'step': 2, 'processor': 'recon', 'params': {}}
        ]

        result = append_answer_retrieval_to_flow_config(
            input_config,
            prefill_text="Thus, the answer is"
        )

        assert len(result) == 3
        assert result[2]['step'] == 3
        assert result[2]['processor'] == 'answer'
        assert result[2]['params'] == {
            'mode': 'retrieval',
            'prefill_text': 'Thus, the answer is'
        }

    def test_prepare_answer_generation_task_with_prefill(self):
        """Should use prefilled prompt when answer_retrieval is enabled"""
        from run_recon import prepare_answer_generation_task

        result = {
            'unique_id': 'test-0',
            'question_id': 0,
            'question': 'What is 2+2?',
            'ground_truth': '4',
            'reconstructed_reasoning': 'We need to add 2 + 2 = 4',
            'answer_prefill': 'Thus, the answer is'  # Indicates prefill should be used
        }

        task = prepare_answer_generation_task(result, model_type='gpt-oss')

        # Should use CompletionRequest with answer_prefix
        assert task.request.answer_prefix == 'Thus, the answer is'
        assert task.request.question == 'What is 2+2?'
        assert task.request.reasoning == 'We need to add 2 + 2 = 4'
        # Metadata should track answer_prefill
        assert 'answer_prefill' in task.metadata
        assert task.metadata['answer_prefill'] == 'Thus, the answer is'

    def test_prepare_answer_generation_task_without_prefill(self):
        """Should use normal prompt when answer_retrieval is disabled"""
        from run_recon import prepare_answer_generation_task

        result = {
            'unique_id': 'test-0',
            'question_id': 0,
            'question': 'What is 2+2?',
            'ground_truth': '4',
            'reconstructed_reasoning': 'We need to add 2 + 2 = 4'
            # No answer_prefill key
        }

        task = prepare_answer_generation_task(result, model_type='gpt-oss')

        # Should use CompletionRequest with empty answer_prefix
        assert task.request.answer_prefix == ''
        assert task.request.question == 'What is 2+2?'
        assert task.request.reasoning == 'We need to add 2 + 2 = 4'
        # Metadata should not have answer_prefill
        assert 'answer_prefill' not in task.metadata

    def test_build_output_structure_with_answer_retrieval(self):
        """Output structure should include answer('retrieval') in flow"""
        from run_recon import build_output_structure

        input_metadata = {
            'experiment_name': 'test',
            'dataset': 'AIME2025',
            'model_type': 'gpt-oss',
            'flow': "mask('number')",
            'flow_config': [
                {'step': 1, 'processor': 'mask', 'params': {'mode': 'number'}}
            ]
        }

        results = [
            {
                'unique_id': 'test-0',
                'question_id': 0,
                'is_correct': True,
                'success': True,
                'answer_prefill': 'Thus, the answer is'  # Indicates answer retrieval was used
            }
        ]

        output = build_output_structure(
            input_metadata,
            results,
            use_answer_retrieval=True,
            answer_prefill_text='Thus, the answer is'
        )

        # Flow should include answer('retrieval')
        expected_flow = "mask('number'),recon(),answer('retrieval')"
        assert output['experiment_metadata']['flow'] == expected_flow

        # Flow config should include answer step
        flow_config = output['experiment_metadata']['flow_config']
        answer_step = [step for step in flow_config if step['processor'] == 'answer']
        assert len(answer_step) == 1
        assert answer_step[0]['params']['mode'] == 'retrieval'
        assert answer_step[0]['params']['prefill_text'] == 'Thus, the answer is'

    def test_build_output_structure_without_answer_retrieval(self):
        """Output structure should NOT include answer('retrieval') when disabled"""
        from run_recon import build_output_structure

        input_metadata = {
            'experiment_name': 'test',
            'dataset': 'AIME2025',
            'model_type': 'gpt-oss',
            'flow': "mask('number')",
            'flow_config': [
                {'step': 1, 'processor': 'mask', 'params': {'mode': 'number'}}
            ]
        }

        results = [
            {
                'unique_id': 'test-0',
                'question_id': 0,
                'is_correct': True,
                'success': True
            }
        ]

        output = build_output_structure(
            input_metadata,
            results,
            use_answer_retrieval=False
        )

        # Flow should NOT include answer('retrieval')
        expected_flow = "mask('number'),recon()"
        assert output['experiment_metadata']['flow'] == expected_flow

        # Flow config should NOT include answer step
        flow_config = output['experiment_metadata']['flow_config']
        answer_step = [step for step in flow_config if step['processor'] == 'answer']
        assert len(answer_step) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
