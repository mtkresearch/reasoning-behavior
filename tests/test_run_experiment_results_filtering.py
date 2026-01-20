#!/usr/bin/env python3
"""
Tests for run_experiment.py results filtering

Verifies that results with null generated_answer are properly filtered
from the final results.json output.
"""

import json
import tempfile
from pathlib import Path
import pytest

# Import the function to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_experiment import rebuild_json_from_jsonl, append_to_jsonl, load_from_jsonl


def create_test_result(unique_id, generated_answer=None, generation_success=True, is_correct=None):
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
        'is_correct': is_correct,
        'grading_reasoning': 'Grading result',
        'generation_success': generation_success,
        'success': generation_success and generated_answer is not None and is_correct is not None,
        'error': None if generated_answer else 'Generation failed',
        'retry_count': 0
    }


class TestResultsFiltering:
    """Test suite for null generated_answer filtering"""

    def test_rebuild_json_filters_null_answers(self):
        """Test that results with null generated_answer are filtered out"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stage2_jsonl = tmpdir / "stage2.jsonl"
            output_json = tmpdir / "results.json"

            # Create test data: 3 results, 1 with null generated_answer
            results = [
                create_test_result('q-0', generated_answer='42', generation_success=True, is_correct=True),
                create_test_result('q-1', generated_answer=None, generation_success=False, is_correct=None),  # Should be filtered
                create_test_result('q-2', generated_answer='43', generation_success=True, is_correct=False),
            ]

            # Write results to JSONL
            for result in results:
                append_to_jsonl(stage2_jsonl, result)

            # Rebuild JSON
            metadata = {'experiment_name': 'test'}
            rebuild_json_from_jsonl(stage2_jsonl, output_json, metadata)

            # Load and verify
            with open(output_json, 'r') as f:
                output = json.load(f)

            # Check that null answer was filtered
            output_results = output['results']
            assert len(output_results) == 2, f"Expected 2 results after filtering, got {len(output_results)}"

            # Verify remaining results have non-null answers
            for result in output_results:
                assert result['generated_answer'] is not None, \
                    f"Found null generated_answer in result {result['unique_id']}"

            # Check summary
            summary = output['summary']
            assert summary['total_questions'] == 3, "total_questions should include all original items"
            assert summary['note'] == 'Filtered out 1 results with null generated_answer'

    def test_rebuild_json_all_valid_answers(self):
        """Test that all results are included when all have valid answers"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stage2_jsonl = tmpdir / "stage2.jsonl"
            output_json = tmpdir / "results.json"

            # Create test data: 2 results, all with valid answers
            results = [
                create_test_result('q-0', generated_answer='42', generation_success=True, is_correct=True),
                create_test_result('q-1', generated_answer='43', generation_success=True, is_correct=False),
            ]

            # Write results to JSONL
            for result in results:
                append_to_jsonl(stage2_jsonl, result)

            # Rebuild JSON
            metadata = {'experiment_name': 'test'}
            rebuild_json_from_jsonl(stage2_jsonl, output_json, metadata)

            # Load and verify
            with open(output_json, 'r') as f:
                output = json.load(f)

            output_results = output['results']
            assert len(output_results) == 2, f"Expected 2 results, got {len(output_results)}"

            # Check summary
            summary = output['summary']
            assert summary['note'] == '', "note should be empty when nothing is filtered"
            assert summary['correct'] == 1, "Should have 1 correct answer"

    def test_rebuild_json_all_null_answers(self):
        """Test that all results are filtered when all have null answers"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stage2_jsonl = tmpdir / "stage2.jsonl"
            output_json = tmpdir / "results.json"

            # Create test data: 2 results, all with null answers
            results = [
                create_test_result('q-0', generated_answer=None, generation_success=False, is_correct=None),
                create_test_result('q-1', generated_answer=None, generation_success=False, is_correct=None),
            ]

            # Write results to JSONL
            for result in results:
                append_to_jsonl(stage2_jsonl, result)

            # Rebuild JSON
            metadata = {'experiment_name': 'test'}
            rebuild_json_from_jsonl(stage2_jsonl, output_json, metadata)

            # Load and verify
            with open(output_json, 'r') as f:
                output = json.load(f)

            output_results = output['results']
            assert len(output_results) == 0, f"Expected 0 results after filtering, got {len(output_results)}"

            # Check summary
            summary = output['summary']
            assert summary['total_questions'] == 2
            assert summary['generation_successful'] == 0
            assert summary['generation_failed'] == 2
            assert summary['grading_successful'] == 0
            assert summary['grading_failed'] == 0
            assert summary['note'] == 'Filtered out 2 results with null generated_answer'

    def test_empty_string_answer_is_not_filtered(self):
        """Test that empty string answers are NOT filtered (only null is filtered)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stage2_jsonl = tmpdir / "stage2.jsonl"
            output_json = tmpdir / "results.json"

            # Create test data with empty string (not None)
            result = create_test_result('q-0', generated_answer='', generation_success=True, is_correct=False)
            append_to_jsonl(stage2_jsonl, result)

            # Rebuild JSON
            metadata = {'experiment_name': 'test'}
            rebuild_json_from_jsonl(stage2_jsonl, output_json, metadata)

            # Load and verify
            with open(output_json, 'r') as f:
                output = json.load(f)

            output_results = output['results']
            assert len(output_results) == 1, "Empty string answer should NOT be filtered out"
            assert output_results[0]['generated_answer'] == ''

    def test_statistics_calculation_with_mixed_results(self):
        """Test that statistics are calculated correctly with mixed success/failure results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            stage2_jsonl = tmpdir / "stage2.jsonl"
            output_json = tmpdir / "results.json"

            # Create test data:
            # - 2 successful generation + grading (1 correct, 1 incorrect)
            # - 1 successful generation but failed grading
            # - 1 failed generation (null answer)
            # Total: 4 questions
            results = [
                create_test_result('q-0', generated_answer='42', generation_success=True, is_correct=True),   # Pass all
                create_test_result('q-1', generated_answer='43', generation_success=True, is_correct=False),  # Gen OK, grade fail
                create_test_result('q-2', generated_answer='44', generation_success=True, is_correct=None),   # Gen OK, not graded
                create_test_result('q-3', generated_answer=None, generation_success=False, is_correct=None),  # Gen fail
            ]

            # Write results to JSONL
            for result in results:
                append_to_jsonl(stage2_jsonl, result)

            # Rebuild JSON
            metadata = {'experiment_name': 'test'}
            rebuild_json_from_jsonl(stage2_jsonl, output_json, metadata)

            # Load and verify
            with open(output_json, 'r') as f:
                output = json.load(f)

            summary = output['summary']

            # total_questions should include all original items (4)
            assert summary['total_questions'] == 4
            # generation_successful: q-0, q-1, q-2 (3 with non-null answers)
            assert summary['generation_successful'] == 3
            # generation_failed: q-3 (1 with null answer)
            assert summary['generation_failed'] == 1
            # grading_successful: q-0, q-1 (2 have success=True)
            assert summary['grading_successful'] == 2
            # grading_failed: q-2 (1 successfully generated but not graded)
            assert summary['grading_failed'] == 1
            # correct: q-0 (1 is_correct=True)
            assert summary['correct'] == 1
            # accuracy: 1 / 2 = 0.5
            assert summary['accuracy'] == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
