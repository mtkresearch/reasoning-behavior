"""
Tests for run_experiment.py

Following TDD approach:
- Test model_type parameter override functionality
"""

import pytest
from run_experiment import extract_model_from_path


class TestModelTypeOverride:
    """Test model_type parameter override functionality"""

    def test_extract_model_from_path(self):
        """Test that extract_model_from_path correctly extracts model type from path"""
        # Test case 1: olmo path
        path = "data/AIME2025__R10/olmo/p1/results.json"
        assert extract_model_from_path(path) == "olmo"

        # Test case 2: olmo--base path (if it exists)
        path = "data/AIME2025__R10/olmo--base/p1/results.json"
        assert extract_model_from_path(path) == "olmo--base"

        # Test case 3: deepseek path
        path = "data/AIME2025__R10/deepseek/p1/results.json"
        assert extract_model_from_path(path) == "deepseek"

    def test_model_type_parameter_usage(self):
        """
        Test that model_type parameter is used when provided.

        This is a placeholder test to document the expected behavior:
        - When --model_type is provided, it should override extract_model_from_path()
        - When --model_type is None, use extract_model_from_path()

        Example usage:
        python run_experiment.py --flow "mask('number')" \
            --results_path data/AIME2025__R10/olmo/p1/results.json \
            --model_type olmo--base

        Expected: model_type should be 'olmo--base', not 'olmo'
        """
        # This test documents the expected behavior
        # Actual integration testing should be done manually or with end-to-end tests
        pass
