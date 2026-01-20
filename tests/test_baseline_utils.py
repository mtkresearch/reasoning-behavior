#!/usr/bin/env python3
"""
Unit tests for baseline_utils.py

Tests cover:
- JSONL loading and appending
- Result saving
- Core generation workflow with mocks
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from baseline_utils import (
    load_existing_jsonl_results,
    append_result_to_jsonl,
    save_results,
    generate_baseline_core
)
from llm_client import Response


class TestLoadExistingJsonlResults:
    """Test JSONL loading functionality."""

    def test_load_empty_file(self):
        """Test loading from non-existent JSONL file returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"
            results = load_existing_jsonl_results(jsonl_path)
            assert results == {}

    def test_load_single_result(self):
        """Test loading a single result from JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            # Write one result
            result = {
                'unique_id': 'test-1',
                'question': 'What is 2+2?',
                'answer': '4'
            }
            with open(jsonl_path, 'w') as f:
                f.write(json.dumps(result) + '\n')

            # Load and verify
            loaded = load_existing_jsonl_results(jsonl_path)
            assert len(loaded) == 1
            assert 'test-1' in loaded
            assert loaded['test-1'] == result

    def test_load_multiple_results(self):
        """Test loading multiple results from JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            # Write multiple results
            results = [
                {'unique_id': f'problem-{i}', 'value': i}
                for i in range(5)
            ]
            with open(jsonl_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')

            # Load and verify
            loaded = load_existing_jsonl_results(jsonl_path)
            assert len(loaded) == 5
            for i in range(5):
                assert f'problem-{i}' in loaded

    def test_load_ignores_empty_lines(self):
        """Test that empty lines in JSONL are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            # Write results with empty lines
            with open(jsonl_path, 'w') as f:
                f.write(json.dumps({'unique_id': 'problem-1', 'value': 1}) + '\n')
                f.write('\n')
                f.write('   \n')
                f.write(json.dumps({'unique_id': 'problem-2', 'value': 2}) + '\n')

            loaded = load_existing_jsonl_results(jsonl_path)
            assert len(loaded) == 2


class TestAppendResultToJsonl:
    """Test JSONL appending functionality."""

    def test_append_to_new_file(self):
        """Test appending to a non-existent file creates it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            result = {'unique_id': 'test-1', 'value': 'data'}
            append_result_to_jsonl(result, jsonl_path)

            # Verify file was created and contains result
            assert jsonl_path.exists()
            with open(jsonl_path) as f:
                line = f.readline()
                loaded = json.loads(line)
                assert loaded == result

    def test_append_multiple_results(self):
        """Test appending multiple results sequentially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            results = [
                {'unique_id': f'problem-{i}', 'value': i}
                for i in range(3)
            ]

            for result in results:
                append_result_to_jsonl(result, jsonl_path)

            # Verify all results are in file
            with open(jsonl_path) as f:
                lines = f.readlines()
                assert len(lines) == 3
                for i, line in enumerate(lines):
                    loaded = json.loads(line)
                    assert loaded['unique_id'] == f'problem-{i}'

    def test_append_preserves_chinese_characters(self):
        """Test that Chinese characters are preserved in JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            result = {'unique_id': 'test-1', 'text': '這是中文', 'question': '什麼是AI?'}
            append_result_to_jsonl(result, jsonl_path)

            # Reload and verify
            loaded = load_existing_jsonl_results(jsonl_path)
            assert loaded['test-1']['text'] == '這是中文'
            assert loaded['test-1']['question'] == '什麼是AI?'

    def test_append_creates_parent_directory(self):
        """Test that parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "deep" / "nested" / "path" / "results.jsonl"

            result = {'unique_id': 'test-1'}
            append_result_to_jsonl(result, jsonl_path)

            assert jsonl_path.exists()


class TestSaveResults:
    """Test JSON results saving functionality."""

    def test_save_empty_list(self):
        """Test saving empty results list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            save_results([], str(output_path))

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
                assert data == []

    def test_save_multiple_results(self):
        """Test saving multiple results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            results = [
                {'unique_id': 'test-1', 'value': 1},
                {'unique_id': 'test-2', 'value': 2},
                {'unique_id': 'test-3', 'value': 3}
            ]

            save_results(results, str(output_path))

            # Verify file contents
            with open(output_path) as f:
                loaded = json.load(f)
                assert len(loaded) == 3
                assert loaded[0]['unique_id'] == 'test-1'

    def test_save_preserves_order(self):
        """Test that result order is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            results = [
                {'id': i, 'data': f'item-{i}'}
                for i in range(10)
            ]

            save_results(results, str(output_path))

            with open(output_path) as f:
                loaded = json.load(f)
                for i, result in enumerate(loaded):
                    assert result['id'] == i

    def test_save_creates_parent_directory(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "deep" / "nested" / "results.json"

            save_results([{'id': 1}], str(output_path))

            assert output_path.exists()


class TestGenerateBaselineCore:
    """Test core generation workflow."""

    def test_generate_with_mock_response(self):
        """Test basic generation workflow with mocked LLM client."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            # Create mock problems
            problems = [
                {'unique_id': f'problem-{i}', 'question': f'Question {i}'}
                for i in range(2)
            ]

            # Mock functions
            def build_prompt(problem):
                return f"Prompt for {problem['unique_id']}"

            def format_result(problem, response, system_prompt):
                return {
                    'unique_id': problem['unique_id'],
                    'question': problem['question'],
                    'result': {
                        'answer': response.content,
                        'traj': response.reasoning_content,
                        'sys_prompt': system_prompt
                    }
                }

            # Mock response
            mock_response = Mock(spec=Response)
            mock_response.success = True
            mock_response.content = 'Test answer'
            mock_response.reasoning_content = 'Test reasoning'
            mock_response.elapsed_seconds = 1.0
            mock_response.err_message = None

            # Mock LLMClient
            with patch('baseline_utils.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                # Mock task generator
                mock_tasks = []
                def mock_generate(tasks, max_workers, use_complete_api):
                    # Create completed tasks
                    for task in tasks:
                        task.response = mock_response
                        mock_tasks.append(task)
                    return mock_tasks

                mock_client.generate_concurrent.side_effect = mock_generate

                # Run generation
                results = generate_baseline_core(
                    problems=problems,
                    output_path=str(output_path),
                    build_prompt_fn=build_prompt,
                    format_result_fn=format_result,
                    system_prompt='Test system prompt',
                    id_field='unique_id',
                    limit=None,
                    max_workers=2
                )

                # Verify results (order may vary due to shuffling)
                assert len(results) == 2
                result_ids = {r['unique_id'] for r in results}
                assert result_ids == {'problem-0', 'problem-1'}
                # Verify all results have expected structure
                for result in results:
                    assert 'unique_id' in result
                    assert 'question' in result
                    assert 'result' in result
                    assert result['result']['answer'] == 'Test answer'
                    assert result['result']['traj'] == 'Test reasoning'

    def test_generate_with_existing_results(self):
        """Test that generation skips existing results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            jsonl_path = Path(tmpdir) / "results.jsonl"

            # Create existing JSONL result
            existing_result = {
                'unique_id': 'problem-0',
                'result': {'answer': 'existing answer'}
            }
            with open(jsonl_path, 'w') as f:
                f.write(json.dumps(existing_result) + '\n')

            # Create problems (including one that exists)
            problems = [
                {'unique_id': 'problem-0', 'question': 'Q1'},
                {'unique_id': 'problem-1', 'question': 'Q2'}
            ]

            # Mock functions
            def build_prompt(p):
                return p['question']

            def format_result(p, r, s):
                return {'unique_id': p['unique_id'], 'result': {'answer': r.content}}

            mock_response = Mock(spec=Response)
            mock_response.success = True
            mock_response.content = 'New answer'
            mock_response.reasoning_content = 'Reasoning'
            mock_response.elapsed_seconds = 1.0

            with patch('baseline_utils.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                def mock_generate(tasks, max_workers, use_complete_api):
                    for task in tasks:
                        task.response = mock_response
                    return tasks

                mock_client.generate_concurrent.side_effect = mock_generate

                results = generate_baseline_core(
                    problems=problems,
                    output_path=str(output_path),
                    build_prompt_fn=build_prompt,
                    format_result_fn=format_result,
                    system_prompt='System',
                    id_field='unique_id',
                    limit=None,
                    max_workers=1
                )

                # Verify only one new result was generated
                assert len(results) == 2
                # Existing result should be first
                assert results[0]['result']['answer'] == 'existing answer'

    def test_generate_with_limit(self):
        """Test that limit parameter works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            # Create 10 problems but limit to 2
            problems = [
                {'unique_id': f'problem-{i}', 'question': f'Q{i}'}
                for i in range(10)
            ]

            def build_prompt(p):
                return p['question']

            def format_result(p, r, s):
                return {'unique_id': p['unique_id']}

            mock_response = Mock(spec=Response)
            mock_response.success = True
            mock_response.content = 'Answer'
            mock_response.reasoning_content = 'Reasoning'
            mock_response.elapsed_seconds = 1.0

            with patch('baseline_utils.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                def mock_generate(tasks, max_workers, use_complete_api):
                    for task in tasks:
                        task.response = mock_response
                    return tasks

                mock_client.generate_concurrent.side_effect = mock_generate

                results = generate_baseline_core(
                    problems=problems,
                    output_path=str(output_path),
                    build_prompt_fn=build_prompt,
                    format_result_fn=format_result,
                    system_prompt='System',
                    id_field='unique_id',
                    limit=2,
                    max_workers=1
                )

                # Only 2 should be generated
                assert len(results) == 2

    def test_generate_skips_empty_response(self):
        """Test that empty responses are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            problems = [{'unique_id': 'problem-1'}]

            def build_prompt(p):
                return 'Q'

            def format_result(p, r, s):
                return {'unique_id': p['unique_id']}

            # Mock response with empty content
            mock_response = Mock(spec=Response)
            mock_response.success = True
            mock_response.content = ''  # Empty!
            mock_response.reasoning_content = 'Reasoning'
            mock_response.elapsed_seconds = 1.0

            with patch('baseline_utils.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                def mock_generate(tasks, max_workers, use_complete_api):
                    for task in tasks:
                        task.response = mock_response
                    return tasks

                mock_client.generate_concurrent.side_effect = mock_generate

                results = generate_baseline_core(
                    problems=problems,
                    output_path=str(output_path),
                    build_prompt_fn=build_prompt,
                    format_result_fn=format_result,
                    system_prompt='System',
                    id_field='unique_id',
                    limit=None,
                    max_workers=1
                )

                # No results should be saved due to empty content
                assert len(results) == 0
