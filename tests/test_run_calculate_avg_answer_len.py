"""
Unit tests for run_calculate_avg_answer_len.py
"""

import pytest
import json
import tempfile
from pathlib import Path

from run_calculate_avg_answer_len import (
    calculate_answer_length_stats,
    load_tokenizer,
    save_statistics,
)


class TestLoadTokenizer:
    """Tests for tokenizer loading"""

    def test_load_tokenizer_default(self):
        """Test loading default GPT-OSS tokenizer"""
        tokenizer = load_tokenizer()
        assert tokenizer is not None

    def test_load_tokenizer_cache(self):
        """Test tokenizer caching"""
        tokenizer1 = load_tokenizer("openai/gpt-oss-120b")
        tokenizer2 = load_tokenizer("openai/gpt-oss-120b")
        assert tokenizer1 is tokenizer2  # Should be the same cached instance


class TestCalculateAnswerLengthStats:
    """Tests for calculate_answer_length_stats function"""

    def test_calculate_stats_basic(self, tmp_path):
        """Test basic statistics calculation"""
        # Create test results file
        results_data = {
            "results": [
                {"generated_answer": "This is a test answer."},
                {"generated_answer": "Another answer here."},
                {"generated_answer": "Short."},
            ]
        }
        results_path = tmp_path / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f)

        # Calculate stats
        stats = calculate_answer_length_stats(str(results_path))

        # Verify results
        assert stats['total_answers'] == 3
        assert stats['tokenizer_model'] == "openai/gpt-oss-120b"
        assert 'mean' in stats
        assert 'stdev' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert len(stats['lengths']) == 3
        assert all(isinstance(length, int) for length in stats['lengths'])

    def test_calculate_stats_with_list_format(self, tmp_path):
        """Test with results as list instead of dict"""
        results_data = [
            {"generated_answer": "Answer 1"},
            {"generated_answer": "Answer 2"},
        ]
        results_path = tmp_path / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f)

        stats = calculate_answer_length_stats(str(results_path))

        assert stats['total_answers'] == 2
        assert len(stats['lengths']) == 2

    def test_calculate_stats_skip_none_answers(self, tmp_path):
        """Test that None answers are skipped"""
        results_data = {
            "results": [
                {"generated_answer": "Valid answer"},
                {"generated_answer": None},
                {"generated_answer": "Another valid answer"},
            ]
        }
        results_path = tmp_path / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f)

        stats = calculate_answer_length_stats(str(results_path))

        # Should only count 2 valid answers
        assert stats['total_answers'] == 2
        assert len(stats['lengths']) == 2

    def test_calculate_stats_skip_missing_field(self, tmp_path):
        """Test that results missing generated_answer are skipped"""
        results_data = {
            "results": [
                {"generated_answer": "Valid answer"},
                {"other_field": "No answer here"},
                {"generated_answer": "Another valid answer"},
            ]
        }
        results_path = tmp_path / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f)

        stats = calculate_answer_length_stats(str(results_path))

        # Should only count 2 valid answers
        assert stats['total_answers'] == 2
        assert len(stats['lengths']) == 2

    def test_calculate_stats_empty_string(self, tmp_path):
        """Test handling of empty string answers"""
        results_data = {
            "results": [
                {"generated_answer": ""},
                {"generated_answer": "Valid answer"},
            ]
        }
        results_path = tmp_path / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f)

        stats = calculate_answer_length_stats(str(results_path))

        # Should count both (empty string has 0 tokens)
        assert stats['total_answers'] == 2
        assert len(stats['lengths']) == 2
        assert stats['min'] == 0  # Empty string should have 0 tokens

    def test_calculate_stats_no_valid_answers_raises_error(self, tmp_path):
        """Test that error is raised when no valid answers found"""
        results_data = {
            "results": [
                {"generated_answer": None},
                {"other_field": "No answer"},
            ]
        }
        results_path = tmp_path / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f)

        with pytest.raises(ValueError, match="沒有找到任何有效的 generated_answer"):
            calculate_answer_length_stats(str(results_path))

    def test_calculate_stats_single_answer_stdev(self, tmp_path):
        """Test that stdev is 0 for single answer"""
        results_data = {
            "results": [
                {"generated_answer": "Only one answer"},
            ]
        }
        results_path = tmp_path / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f)

        stats = calculate_answer_length_stats(str(results_path))

        assert stats['total_answers'] == 1
        assert stats['stdev'] == 0.0  # Single value has no standard deviation


class TestSaveStatistics:
    """Tests for save_statistics function"""

    def test_save_statistics_creates_file(self, tmp_path):
        """Test that statistics are saved correctly"""
        stats = {
            'tokenizer_model': 'openai/gpt-oss-120b',
            'total_answers': 3,
            'mean': 10.5,
            'stdev': 2.5,
            'min': 8,
            'max': 13,
            'median': 10.0,
            'lengths': [8, 10, 13],
        }

        output_path = tmp_path / "stats.json"
        save_statistics(stats, str(output_path))

        # Verify file exists
        assert output_path.exists()

        # Load and verify content
        with open(output_path, 'r') as f:
            saved_data = json.load(f)

        assert 'summary' in saved_data
        assert 'lengths' in saved_data
        assert saved_data['summary']['tokenizer_model'] == 'openai/gpt-oss-120b'
        assert saved_data['summary']['total_answers'] == 3
        assert saved_data['summary']['mean'] == 10.5
        assert saved_data['summary']['sample_size'] == 3
        assert saved_data['lengths'] == [8, 10, 13]

    def test_save_statistics_format(self, tmp_path):
        """Test that saved JSON has correct format"""
        stats = {
            'tokenizer_model': 'test-model',
            'total_answers': 2,
            'mean': 5.0,
            'stdev': 1.0,
            'min': 4,
            'max': 6,
            'median': 5.0,
            'lengths': [4, 6],
        }

        output_path = tmp_path / "stats.json"
        save_statistics(stats, str(output_path))

        with open(output_path, 'r') as f:
            saved_data = json.load(f)

        # Summary should not contain lengths array
        assert 'lengths' not in saved_data['summary']
        # But should have sample_size
        assert 'sample_size' in saved_data['summary']
        # Lengths should be in separate field
        assert isinstance(saved_data['lengths'], list)


@pytest.fixture
def sample_results_file(tmp_path):
    """Fixture to create a sample results.json file"""
    results_data = {
        "experiment_metadata": {
            "experiment_name": "test_experiment",
            "model_type": "gpt-oss",
        },
        "results": [
            {
                "unique_id": "test-0",
                "question": "What is 2+2?",
                "generated_answer": "The answer is 4.",
                "ground_truth": "4",
            },
            {
                "unique_id": "test-1",
                "question": "What is the capital of France?",
                "generated_answer": "Paris is the capital of France.",
                "ground_truth": "Paris",
            },
            {
                "unique_id": "test-2",
                "question": "Simple question",
                "generated_answer": "Yes.",
                "ground_truth": "Yes",
            },
        ]
    }

    results_path = tmp_path / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f)

    return results_path


class TestIntegration:
    """Integration tests using sample data"""

    def test_full_workflow(self, sample_results_file, tmp_path):
        """Test complete workflow from loading to saving"""
        # Calculate stats
        stats = calculate_answer_length_stats(str(sample_results_file))

        # Verify calculation
        assert stats['total_answers'] == 3
        assert len(stats['lengths']) == 3
        assert all(length > 0 for length in stats['lengths'])

        # Save statistics
        output_path = tmp_path / "output_stats.json"
        save_statistics(stats, str(output_path))

        # Verify saved file
        assert output_path.exists()
        with open(output_path, 'r') as f:
            saved_data = json.load(f)

        assert saved_data['summary']['total_answers'] == 3
        assert len(saved_data['lengths']) == 3
