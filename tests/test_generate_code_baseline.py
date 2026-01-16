#!/usr/bin/env python3
"""
Unit tests for generate_code_baseline.py
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_code_baseline import (
    load_existing_jsonl_results,
    append_result_to_jsonl,
    make_html_problem,
    build_prompt
)


class TestJSONLCache:
    """Test JSONL caching functionality"""

    def test_load_existing_jsonl_results_empty(self):
        """Test loading from non-existent JSONL file"""
        with TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"
            results = load_existing_jsonl_results(jsonl_path)
            assert results == {}

    def test_append_result_to_jsonl(self):
        """Test appending a single result to JSONL"""
        with TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            result = {
                'unique_id': 'codeforces-1234A-0',
                'question': 'Test question',
                'test_cases': [['1', '2']],
                'result': {
                    'traj': 'Test reasoning',
                    'answer': 'Test answer',
                    'sys_prompt': 'Test prompt',
                    'elapsed_seconds': 1.5
                }
            }

            # Append result
            append_result_to_jsonl(result, jsonl_path)

            # Verify file exists and contains correct data
            assert jsonl_path.exists()
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 1
                loaded_result = json.loads(lines[0])
                assert loaded_result == result

    def test_load_existing_jsonl_results_with_data(self):
        """Test loading existing results from JSONL file"""
        with TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            # Create test data
            results = [
                {
                    'unique_id': 'codeforces-1234A-0',
                    'question': 'Question 1',
                    'test_cases': [['1', '2']],
                    'result': {'traj': 'Reasoning 1', 'answer': 'Answer 1', 'sys_prompt': 'Prompt', 'elapsed_seconds': 1.0}
                },
                {
                    'unique_id': 'codeforces-5678B-0',
                    'question': 'Question 2',
                    'test_cases': [['3', '4']],
                    'result': {'traj': 'Reasoning 2', 'answer': 'Answer 2', 'sys_prompt': 'Prompt', 'elapsed_seconds': 2.0}
                }
            ]

            # Write test data
            for result in results:
                append_result_to_jsonl(result, jsonl_path)

            # Load and verify
            loaded_results = load_existing_jsonl_results(jsonl_path)
            assert len(loaded_results) == 2
            assert '1234A' in loaded_results
            assert '5678B' in loaded_results
            assert loaded_results['1234A']['question'] == 'Question 1'
            assert loaded_results['5678B']['question'] == 'Question 2'

    def test_load_existing_jsonl_results_skip_invalid_lines(self):
        """Test that invalid lines are skipped when loading"""
        with TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            # Write valid and invalid lines
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'unique_id': 'codeforces-1234A-0', 'question': 'Q1'}) + '\n')
                f.write('\n')  # Empty line
                f.write(json.dumps({'unique_id': 'codeforces-5678B-0', 'question': 'Q2'}) + '\n')

            # Load and verify
            loaded_results = load_existing_jsonl_results(jsonl_path)
            assert len(loaded_results) == 2


class TestProblemFormatting:
    """Test problem formatting functions"""

    def test_make_html_problem_basic(self):
        """Test basic HTML problem generation"""
        problem = {
            'title': 'Test Problem',
            'time_limit_ms': 1000,
            'memory_limit_mb': 256,
            'rating': 1200,
            'description': 'Test description',
            'input': 'Test input format',
            'output': 'Test output format',
            'examples': [['1 2', '3'], ['4 5', '9']],
            'note': 'Test note'
        }

        html = make_html_problem(problem)

        # Verify key elements are present
        assert '<html><body>' in html
        assert '<h1>Test Problem</h1>' in html
        assert '<b>Time limit:</b> 1000 ms' in html
        assert '<b>Memory limit:</b> 256 MB' in html
        assert '<b>Rating:</b> 1200' in html
        assert 'Test description' in html
        assert 'Test input format' in html
        assert 'Test output format' in html
        assert 'Example 1' in html
        assert 'Example 2' in html
        assert 'Test note' in html
        assert '</body></html>' in html

    def test_make_html_problem_without_note(self):
        """Test HTML problem generation without note"""
        problem = {
            'title': 'Test Problem',
            'time_limit_ms': 1000,
            'memory_limit_mb': 256,
            'rating': 1200,
            'description': 'Test description',
            'input': 'Test input format',
            'output': 'Test output format',
            'examples': []
        }

        html = make_html_problem(problem)

        # Verify note section is not present
        assert '<h2>Note</h2>' not in html

    def test_build_prompt(self):
        """Test prompt building"""
        problem = {
            'title': 'Test Problem',
            'time_limit_ms': 1000,
            'memory_limit_mb': 256,
            'rating': 1200,
            'description': 'Test description',
            'input': 'Test input format',
            'output': 'Test output format',
            'examples': []
        }

        prompt = build_prompt(problem)

        # Verify instruction is included
        assert 'You are a coding expert' in prompt
        assert 'C++ program' in prompt

        # Verify problem HTML is included
        assert '<html><body>' in prompt
        assert 'Test Problem' in prompt


class TestResumeCapability:
    """Test resume capability with JSONL cache"""

    def test_resume_from_partial_results(self):
        """Test that only unprocessed problems are handled"""
        with TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"

            # Simulate existing results for problem_id '1234A'
            existing_result = {
                'unique_id': 'codeforces-1234A-0',
                'question': 'Question 1',
                'test_cases': [['1', '2']],
                'result': {'traj': 'Reasoning 1', 'answer': 'Answer 1', 'sys_prompt': 'Prompt', 'elapsed_seconds': 1.0}
            }
            append_result_to_jsonl(existing_result, jsonl_path)

            # Load existing results
            loaded_results = load_existing_jsonl_results(jsonl_path)

            # Verify that problem '1234A' is loaded
            assert '1234A' in loaded_results

            # Simulate filtering problems
            all_problems = [
                {'problem_id': '1234A', 'title': 'Problem A'},
                {'problem_id': '5678B', 'title': 'Problem B'}
            ]

            problems_to_process = [
                p for p in all_problems
                if p['problem_id'] not in loaded_results
            ]

            # Verify only '5678B' needs processing
            assert len(problems_to_process) == 1
            assert problems_to_process[0]['problem_id'] == '5678B'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
