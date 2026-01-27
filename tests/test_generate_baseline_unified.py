#!/usr/bin/env python3
"""
Integration tests for generate_baseline.py (unified script)

Tests the new unified baseline generation script that supports
code, math, and science tasks.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_baseline import (
    load_problems_code,
    load_problems_math,
    load_problems_science,
    build_prompt_code,
    build_prompt_math,
    build_prompt_science,
    format_result_code,
    format_result_math,
    format_result_science,
    make_html_problem,
    TASK_CONFIGS
)


class TestCodeTaskSupport:
    """Test code task type support in unified script."""

    def test_load_problems_code(self):
        """Test loading code problems."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test code problems file
            problems = [
                {
                    'problem_id': '1234A',
                    'title': 'Test Problem',
                    'time_limit_ms': 1000,
                    'memory_limit_mb': 256,
                    'rating': 1200,
                    'description': 'Test',
                    'input': 'Input format',
                    'output': 'Output format',
                    'examples': [['1', '2']],
                }
            ]
            problems_path = Path(tmpdir) / "problems.json"
            with open(problems_path, 'w') as f:
                json.dump(problems, f)

            # Load
            loaded = load_problems_code(str(problems_path))
            assert len(loaded) == 1
            assert loaded[0]['problem_id'] == '1234A'

    def test_build_prompt_code(self):
        """Test code prompt building."""
        problem = {
            'problem_id': '1234A',
            'title': 'Test Problem',
            'time_limit_ms': 1000,
            'memory_limit_mb': 256,
            'rating': 1200,
            'description': 'Test',
            'input': 'Input format',
            'output': 'Output format',
            'examples': [['1', '2']],
        }

        prompt = build_prompt_code(problem)

        assert 'You are a coding expert' in prompt
        assert 'C++ program' in prompt
        assert '<html>' in prompt

    def test_format_result_code(self):
        """Test code result formatting."""
        problem = {
            'problem_id': '1234A',
            'title': 'Test',
            'time_limit_ms': 1000,
            'memory_limit_mb': 256,
            'rating': 1200,
            'description': 'Test',
            'input': 'Input',
            'output': 'Output',
            'examples': [['1', '2']],
        }

        mock_response = Mock()
        mock_response.reasoning_content = 'Test reasoning'
        mock_response.content = 'Test code'
        mock_response.elapsed_seconds = 1.5

        result = format_result_code(problem, mock_response, 'system prompt')

        assert result['unique_id'] == 'codeelo-1234A-0'
        assert result['test_cases'] == [['1', '2']]
        assert result['result']['traj'] == 'Test reasoning'
        assert result['result']['answer'] == 'Test code'


class TestMathTaskSupport:
    """Test math task type support in unified script."""

    def test_load_problems_math_no_repeat(self):
        """Test loading math problems without repeat."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problems = [
                {
                    'unique_id': 'aime-1',
                    'question': 'What is 2+2?',
                    'answer': '4'
                }
            ]
            problems_path = Path(tmpdir) / "problems.json"
            with open(problems_path, 'w') as f:
                json.dump(problems, f)

            loaded = load_problems_math(str(problems_path), repeat_num=1)
            assert len(loaded) == 1
            assert loaded[0]['unique_id'] == 'aime-1'

    def test_load_problems_math_with_repeat(self):
        """Test loading math problems with repeat (R10)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problems = [
                {
                    'unique_id': 'aime-1',
                    'question': 'What is 2+2?',
                    'answer': '4'
                },
                {
                    'unique_id': 'aime-2',
                    'question': 'What is 3+3?',
                    'answer': '6'
                }
            ]
            problems_path = Path(tmpdir) / "problems.json"
            with open(problems_path, 'w') as f:
                json.dump(problems, f)

            loaded = load_problems_math(str(problems_path), repeat_num=3)

            # Should have 2 * 3 = 6 problems
            assert len(loaded) == 6

            # Check that repetitions are properly indexed
            assert loaded[0]['unique_id'] == 'aime-1-0'
            assert loaded[1]['unique_id'] == 'aime-1-1'
            assert loaded[2]['unique_id'] == 'aime-1-2'
            assert loaded[3]['unique_id'] == 'aime-2-0'
            assert loaded[4]['unique_id'] == 'aime-2-1'
            assert loaded[5]['unique_id'] == 'aime-2-2'

    def test_build_prompt_math(self):
        """Test math prompt building."""
        problem = {
            'unique_id': 'aime-1',
            'question': 'What is 2+2?',
            'answer': '4'
        }

        prompt = build_prompt_math(problem)
        assert prompt == 'What is 2+2?'

    def test_format_result_math(self):
        """Test math result formatting."""
        problem = {
            'unique_id': 'aime-1',
            'question': 'What is 2+2?',
            'answer': '4'
        }

        mock_response = Mock()
        mock_response.reasoning_content = 'Let me think...'
        mock_response.content = '4'
        mock_response.elapsed_seconds = 2.0

        result = format_result_math(problem, mock_response, 'system prompt')

        assert result['unique_id'] == 'aime-1'
        assert result['answer'] == '4'
        assert result['result']['answer'] == '4'


class TestScienceTaskSupport:
    """Test science task type support in unified script."""

    def test_load_problems_science(self):
        """Test loading science problems from parquet."""
        # This test uses the actual GPQA-Diamond dataset
        try:
            problems = load_problems_science('datasets/GPQA-Diamond/test/gpqa_diamond.parquet')

            assert len(problems) == 198
            assert all('unique_id' in p for p in problems)
            assert all('question' in p for p in problems)
            assert all('answer' in p for p in problems)

            # Check format of unique_id
            assert all(p['unique_id'].startswith('gpqa-diamond-') for p in problems)

            # Check answer is single letter
            assert all(p['answer'] in ['A', 'B', 'C', 'D'] for p in problems)
        except FileNotFoundError:
            pytest.skip("GPQA-Diamond dataset not found")

    def test_build_prompt_science(self):
        """Test science prompt building."""
        problem = {
            'unique_id': 'gpqa-diamond-0',
            'question': 'What is the capital of France?\na) London\nb) Paris\nc) Berlin\nd) Madrid',
            'answer': 'B'
        }

        prompt = build_prompt_science(problem)
        assert 'What is the capital' in prompt

    def test_format_result_science(self):
        """Test science result formatting."""
        problem = {
            'unique_id': 'gpqa-diamond-0',
            'question': 'What is the capital?',
            'answer': 'B'
        }

        mock_response = Mock()
        mock_response.reasoning_content = 'The capital is...'
        mock_response.content = 'B'
        mock_response.elapsed_seconds = 1.5

        result = format_result_science(problem, mock_response, 'system prompt')

        assert result['unique_id'] == 'gpqa-diamond-0'
        assert result['answer'] == 'B'
        assert result['result']['answer'] == 'B'


class TestTaskConfigs:
    """Test task configuration dictionary."""

    def test_code_config_exists(self):
        """Test code task config."""
        config = TASK_CONFIGS['code']
        assert config['id_field'] == 'problem_id'
        assert 'default_path' in config
        assert 'default_output' in config
        assert 'load_fn' in config
        assert 'prompt_fn' in config
        assert 'result_fn' in config

    def test_math_config_exists(self):
        """Test math task config."""
        config = TASK_CONFIGS['math']
        assert config['id_field'] == 'unique_id'
        assert config['supports_repeat'] is True
        assert 'default_path' in config

    def test_science_config_exists(self):
        """Test science task config."""
        config = TASK_CONFIGS['science']
        assert config['id_field'] == 'unique_id'
        assert config['supports_repeat'] is False
        assert config['default_path'] == 'datasets/GPQA-Diamond/test/gpqa_diamond.parquet'

    def test_all_configs_have_required_fields(self):
        """Test that all configs have required fields."""
        required_fields = {
            'load_fn', 'prompt_fn', 'result_fn', 'system_prompt',
            'id_field', 'default_path', 'default_output', 'supports_repeat'
        }

        for task_type, config in TASK_CONFIGS.items():
            missing = required_fields - set(config.keys())
            assert not missing, f"{task_type} config missing: {missing}"


class TestMakeHtmlProblem:
    """Test HTML problem formatting utility."""

    def test_make_html_problem_complete(self):
        """Test complete HTML generation."""
        problem = {
            'title': 'Test Problem',
            'time_limit_ms': 1000,
            'memory_limit_mb': 256,
            'rating': 1200,
            'description': 'Solve this problem',
            'input': 'First line contains n',
            'output': 'Output the result',
            'examples': [['5', '10']],
            'note': 'Note: be careful'
        }

        html = make_html_problem(problem)

        assert '<html><body>' in html
        assert 'Test Problem' in html
        assert '1000 ms' in html
        assert '256 MB' in html
        assert '1200' in html
        assert 'Solve this problem' in html
        assert 'First line contains n' in html
        assert 'Output the result' in html
        assert 'Example 1' in html
        assert 'be careful' in html
        assert '</body></html>' in html

    def test_make_html_problem_minimal(self):
        """Test HTML generation with minimal fields."""
        problem = {
            'title': 'Simple',
            'time_limit_ms': 1000,
            'memory_limit_mb': 256,
            'rating': 1000,
            'description': 'Desc',
            'input': 'Input',
            'output': 'Output',
            'examples': []
        }

        html = make_html_problem(problem)

        assert 'Simple' in html
        assert '<h2>Examples</h2>' not in html
        assert '<h2>Note</h2>' not in html


class TestOutputPathGeneration:
    """Test that output_path is correctly generated based on model_type."""

    def test_output_path_with_default_model(self):
        """Test default output path uses gpt-oss."""
        # Science task default
        config = TASK_CONFIGS['science']
        assert 'gpt-oss' in config['default_output']

    def test_output_path_generation_logic(self):
        """Test the logic for replacing model_type in output_path."""
        # Simulate what main() does
        default_output = 'data/GPQA-Diamond/gpt-oss/p1/results.json'
        model_type = 'deepseek'

        # This is the logic from the fixed main()
        output_path = default_output.replace('/gpt-oss/', f'/{model_type}/')

        assert output_path == 'data/GPQA-Diamond/deepseek/p1/results.json'

    def test_output_path_generation_with_different_models(self):
        """Test output path generation with various model types."""
        test_cases = [
            ('deepseek', 'data/CodeElo/deepseek/p1/results.json'),
            ('qwen3', 'data/CodeElo/qwen3/p1/results.json'),
            ('gpt-oss', 'data/CodeElo/gpt-oss/p1/results.json'),
        ]

        default_output = 'data/CodeElo/gpt-oss/p1/results.json'

        for model_type, expected_output in test_cases:
            output_path = default_output.replace('/gpt-oss/', f'/{model_type}/')
            assert output_path == expected_output, f"Failed for model_type={model_type}"
