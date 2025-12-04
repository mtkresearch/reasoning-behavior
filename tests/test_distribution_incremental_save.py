#!/usr/bin/env python3
"""
Integration test for run_distribution.py incremental save functionality

Tests that JSONL files are written incrementally as questions are completed,
not just at the end of all processing.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from run_distribution import execute_and_group_tasks, calculate_distribution
from llm_client import Task, CompletionRequest, Response
from core import load_from_jsonl


class TestIncrementalSave:
    """Tests for incremental JSONL writing during task execution"""

    def test_execute_and_group_tasks_writes_immediately(self):
        """Should write to JSONL as soon as all samples for a question are completed"""

        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_path = Path(f.name)

        try:
            # Prepare mock data
            question_metadata = {
                'q1': {
                    'unique_id': 'q1',
                    'question_id': 1,
                    'question': 'What is 2+2?',
                    'ground_truth': '4',
                    'processed_reasoning': 'Let me think...'
                },
                'q2': {
                    'unique_id': 'q2',
                    'question_id': 2,
                    'question': 'What is 3+3?',
                    'ground_truth': '6',
                    'processed_reasoning': 'Let me think...'
                }
            }

            # Create mock tasks (2 questions × 3 samples each)
            all_tasks = []
            for unique_id in ['q1', 'q2']:
                for sample_id in range(3):
                    task = Task(
                        index=len(all_tasks),
                        request=Mock(spec=CompletionRequest),
                        metadata={'unique_id': unique_id, 'sample_id': sample_id}
                    )
                    # Mock successful response
                    task.response = Response(
                        content=f"{question_metadata[unique_id]['ground_truth']}",
                        history="",
                        elapsed_seconds=0.1,
                        success=True
                    )
                    all_tasks.append(task)

            # Mock args
            args = Mock()
            args.n = 3
            args.answer_free_gen = False
            args.max_workers = 2

            # Mock client that returns tasks in order
            mock_client = Mock()
            mock_client.complete_concurrent.return_value = iter(all_tasks)

            # Execute function
            execute_and_group_tasks(
                all_tasks=all_tasks,
                client=mock_client,
                args=args,
                question_metadata=question_metadata,
                sampling_jsonl=jsonl_path
            )

            # Verify JSONL was written
            results = load_from_jsonl(jsonl_path)

            # Should have 2 results (one for each question)
            assert len(results) == 2

            # Verify structure of results
            for result in results:
                assert 'unique_id' in result
                assert 'samples' in result
                assert 'distribution' in result
                assert len(result['samples']) == 3

            # Verify question data
            q1_result = next(r for r in results if r['unique_id'] == 'q1')
            assert q1_result['question_id'] == 1
            assert q1_result['ground_truth'] == '4'

            q2_result = next(r for r in results if r['unique_id'] == 'q2')
            assert q2_result['question_id'] == 2
            assert q2_result['ground_truth'] == '6'

        finally:
            # Cleanup
            if jsonl_path.exists():
                jsonl_path.unlink()

    def test_partial_completion_does_not_write(self):
        """Should not write to JSONL until all N samples are completed"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_path = Path(f.name)

        try:
            question_metadata = {
                'q1': {
                    'unique_id': 'q1',
                    'question_id': 1,
                    'question': 'Test question',
                    'ground_truth': '42',
                    'processed_reasoning': 'Reasoning...'
                }
            }

            # Create only 2 tasks (but n=3, so incomplete)
            all_tasks = []
            for sample_id in range(2):
                task = Task(
                    index=sample_id,
                    request=Mock(spec=CompletionRequest),
                    metadata={'unique_id': 'q1', 'sample_id': sample_id}
                )
                task.response = Response(
                    content="42",
                    history="",
                    elapsed_seconds=0.1,
                    success=True
                )
                all_tasks.append(task)

            args = Mock()
            args.n = 3  # Need 3 samples
            args.answer_free_gen = False
            args.max_workers = 2

            mock_client = Mock()
            mock_client.complete_concurrent.return_value = iter(all_tasks)

            # Execute
            execute_and_group_tasks(
                all_tasks=all_tasks,
                client=mock_client,
                args=args,
                question_metadata=question_metadata,
                sampling_jsonl=jsonl_path
            )

            # Should have written nothing (incomplete)
            results = load_from_jsonl(jsonl_path)
            assert len(results) == 0

        finally:
            if jsonl_path.exists():
                jsonl_path.unlink()

    def test_thread_safety_with_concurrent_completion(self):
        """Should handle thread-safe writing when multiple questions complete concurrently"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_path = Path(f.name)

        try:
            # Create 5 questions with 2 samples each
            question_metadata = {}
            all_tasks = []

            for q_idx in range(5):
                unique_id = f'q{q_idx}'
                question_metadata[unique_id] = {
                    'unique_id': unique_id,
                    'question_id': q_idx,
                    'question': f'Question {q_idx}',
                    'ground_truth': str(q_idx * 10),
                    'processed_reasoning': 'Reasoning...'
                }

                for sample_id in range(2):
                    task = Task(
                        index=len(all_tasks),
                        request=Mock(spec=CompletionRequest),
                        metadata={'unique_id': unique_id, 'sample_id': sample_id}
                    )
                    task.response = Response(
                        content=str(q_idx * 10),
                        history="",
                        elapsed_seconds=0.1,
                        success=True
                    )
                    all_tasks.append(task)

            args = Mock()
            args.n = 2
            args.answer_free_gen = False
            args.max_workers = 4  # Multiple workers for concurrency

            mock_client = Mock()
            # Shuffle task order to simulate concurrent completion
            import random
            shuffled = all_tasks.copy()
            random.shuffle(shuffled)
            mock_client.complete_concurrent.return_value = iter(shuffled)

            # Execute
            execute_and_group_tasks(
                all_tasks=all_tasks,
                client=mock_client,
                args=args,
                question_metadata=question_metadata,
                sampling_jsonl=jsonl_path
            )

            # Verify all 5 questions were written
            results = load_from_jsonl(jsonl_path)
            assert len(results) == 5

            # Verify each has correct samples
            for result in results:
                assert len(result['samples']) == 2
                assert result['total_samples'] == 2

        finally:
            if jsonl_path.exists():
                jsonl_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
