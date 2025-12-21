"""
Tests for DataLoader class in run_attn_visual.py

Following TDD Red-Green-Refactor cycle for Phase 1
"""

import pytest
import json
import tempfile
import os
from pathlib import Path


class TestDataLoader:
    """Test suite for DataLoader class"""

    @pytest.fixture
    def valid_results_data(self):
        """Fixture providing valid results data"""
        return {
            "results": [
                {
                    "question": "What is 2+2?",
                    "ground_truth": "4",
                    "processed_reasoning": "Let me calculate: 2+2=4",
                    "generated_answer": "Thus, the answer is 4",
                    "is_correct": True
                },
                {
                    "question": "What is 3+3?",
                    "ground_truth": "6",
                    "processed_reasoning": "Let me calculate: 3+3=6",
                    "generated_answer": "Thus, the answer is 5",
                    "is_correct": False
                },
                {
                    "question": "What is 5+5?",
                    "ground_truth": "10",
                    "processed_reasoning": "Let me calculate: 5+5=10",
                    "generated_answer": "Thus, the answer is 10",
                    "is_correct": True
                }
            ]
        }

    @pytest.fixture
    def invalid_results_data(self):
        """Fixture providing data with missing required fields"""
        return {
            "results": [
                {
                    "question": "What is 2+2?",
                    # missing ground_truth
                    "processed_reasoning": "Let me calculate: 2+2=4",
                    "generated_answer": "Thus, the answer is 4",
                    "is_correct": True
                }
            ]
        }

    @pytest.fixture
    def temp_json_file(self, valid_results_data):
        """Create a temporary JSON file with valid data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_results_data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_load_valid_json(self, temp_json_file, valid_results_data):
        """Test loading a valid JSON file"""
        from run_attn_visual import DataLoader

        loader = DataLoader()
        results = loader.load_results(temp_json_file)

        assert isinstance(results, list)
        assert len(results) == 3
        assert results == valid_results_data['results']

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error"""
        from run_attn_visual import DataLoader

        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_results('/nonexistent/path/file.json')

    def test_validate_result_with_all_fields(self):
        """Test validation passes for result with all required fields"""
        from run_attn_visual import DataLoader

        loader = DataLoader()
        result = {
            "question": "What is 2+2?",
            "ground_truth": "4",
            "processed_reasoning": "Let me calculate",
            "generated_answer": "The answer is 4",
            "is_correct": True
        }

        assert loader.validate_result(result) is True

    def test_validate_result_missing_fields(self):
        """Test validation fails for result missing required fields"""
        from run_attn_visual import DataLoader

        loader = DataLoader()

        # Missing ground_truth
        result1 = {
            "question": "What is 2+2?",
            "processed_reasoning": "Let me calculate",
            "generated_answer": "The answer is 4",
            "is_correct": True
        }
        assert loader.validate_result(result1) is False

        # Missing is_correct
        result2 = {
            "question": "What is 2+2?",
            "ground_truth": "4",
            "processed_reasoning": "Let me calculate",
            "generated_answer": "The answer is 4"
        }
        assert loader.validate_result(result2) is False

    def test_filter_correct_only(self, valid_results_data):
        """Test filtering only is_correct=True results"""
        from run_attn_visual import DataLoader

        loader = DataLoader()
        results = valid_results_data['results']

        filtered = loader.filter_correct_only(results)

        assert len(filtered) == 2  # Only 2 correct results
        assert all(r['is_correct'] for r in filtered)
        assert filtered[0]['ground_truth'] == "4"
        assert filtered[1]['ground_truth'] == "10"

    def test_filter_correct_only_empty_list(self):
        """Test filtering empty results list"""
        from run_attn_visual import DataLoader

        loader = DataLoader()
        filtered = loader.filter_correct_only([])

        assert filtered == []

    def test_filter_correct_only_all_incorrect(self):
        """Test filtering when all results are incorrect"""
        from run_attn_visual import DataLoader

        loader = DataLoader()
        results = [
            {"is_correct": False, "question": "Q1"},
            {"is_correct": False, "question": "Q2"}
        ]

        filtered = loader.filter_correct_only(results)

        assert filtered == []

    def test_load_and_validate_workflow(self, temp_json_file):
        """Test complete load and validate workflow"""
        from run_attn_visual import DataLoader

        loader = DataLoader()
        results = loader.load_results(temp_json_file)

        # Validate all results
        valid_results = [r for r in results if loader.validate_result(r)]

        assert len(valid_results) == 3

        # Filter for correct only
        correct_results = loader.filter_correct_only(valid_results)

        assert len(correct_results) == 2
