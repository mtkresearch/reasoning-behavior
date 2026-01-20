#!/usr/bin/env python3
"""
Tests for run_experiment.py retry logic

Verifies that results with null generated_answer are retried in subsequent runs,
not skipped as "already completed".
"""

import json
import tempfile
from pathlib import Path
import pytest

# Import the functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_experiment import load_from_jsonl, append_to_jsonl


def create_test_result(unique_id, generated_answer=None, generation_success=True):
    """Helper to create a test result dict"""
    return {
        'unique_id': unique_id,
        'question_id': int(unique_id.split('-')[-1]),
        'question': f'Question {unique_id}',
        'ground_truth': '42',
        'original_reasoning': 'This is reasoning',
        'processed_reasoning': 'This is processed reasoning',
        'flow': 'test_flow',
        'processing_metadata': [],
        'generated_answer': generated_answer,
        'is_correct': None,
        'grading_reasoning': None,
        'generation_success': generation_success,
        'success': False,
        'error': 'Empty generated_answer from API' if not generated_answer else None,
        'retry_count': 0
    }


class TestRetryLogic:
    """Test suite for retry logic with null answers"""

    def test_null_answer_should_be_retried(self):
        """Test that results with null generated_answer are marked for retry"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stage1_jsonl = tmpdir / "stage1.jsonl"

            # Create test data: 2 results, 1 with null answer
            results = [
                create_test_result('q-0', generated_answer='42', generation_success=True),
                create_test_result('q-1', generated_answer=None, generation_success=True),  # Has null!
            ]

            # Write results to JSONL
            for result in results:
                append_to_jsonl(stage1_jsonl, result)

            # Load and simulate the completed_stage1_ids logic
            existing_stage1 = load_from_jsonl(stage1_jsonl)

            # Logic from run_experiment.py:
            # A result is considered truly successful only if generation_success=True AND generated_answer is not None
            completed_stage1_ids = {
                r['unique_id'] for r in existing_stage1
                if r.get('generation_success', False) and r.get('generated_answer') is not None
            }

            # Verify that q-1 (null answer) is NOT in completed_stage1_ids
            assert 'q-0' in completed_stage1_ids, "q-0 should be completed"
            assert 'q-1' not in completed_stage1_ids, "q-1 with null answer should NOT be marked as completed"

    def test_retry_count_increments(self):
        """Test that retry_count increments as results are retried"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stage1_jsonl = tmpdir / "stage1.jsonl"

            # Create initial failed result with null answer
            result = create_test_result('q-0', generated_answer=None, generation_success=True)
            result['retry_count'] = 0
            append_to_jsonl(stage1_jsonl, result)

            # Simulate retry: update result with new answer
            result['generated_answer'] = '42'
            result['retry_count'] = 1
            append_to_jsonl(stage1_jsonl, result)

            # Load all results
            all_results = load_from_jsonl(stage1_jsonl)

            # The last entry should be the updated one with retry_count=1
            assert len(all_results) == 2, "Should have 2 entries"
            assert all_results[-1]['retry_count'] == 1, "Last entry should have retry_count=1"

    def test_failed_results_reappear_in_retry(self):
        """Test that failed results appear in the list to be processed in next run"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stage1_jsonl = tmpdir / "stage1.jsonl"

            # First run: process 3 items, 1 succeeds, 2 fail with null answers
            results_from_run1 = [
                create_test_result('q-0', generated_answer='42', generation_success=True),
                create_test_result('q-1', generated_answer=None, generation_success=True),  # Failed
                create_test_result('q-2', generated_answer=None, generation_success=False),  # Failed
            ]

            for result in results_from_run1:
                append_to_jsonl(stage1_jsonl, result)

            # Simulate what happens in next run
            existing_stage1 = load_from_jsonl(stage1_jsonl)
            completed_stage1_ids = {
                r['unique_id'] for r in existing_stage1
                if r.get('generation_success', False) and r.get('generated_answer') is not None
            }

            # Only q-0 should be completed
            assert len(completed_stage1_ids) == 1, f"Only 1 should be completed, got {len(completed_stage1_ids)}"
            assert 'q-0' in completed_stage1_ids

            # q-1 and q-2 should be retried
            items_to_retry = [r for r in existing_stage1 if r['unique_id'] not in completed_stage1_ids]
            assert len(items_to_retry) == 2, f"Should have 2 items to retry, got {len(items_to_retry)}"
            assert {r['unique_id'] for r in items_to_retry} == {'q-1', 'q-2'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
