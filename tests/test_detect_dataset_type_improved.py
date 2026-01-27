"""
Tests for improved detect_dataset_type function

Tests the enhanced detection mechanism including:
1. Science dataset detection (GPQA-Diamond)
2. Structure-based detection (test_cases vs answer)
3. Multiple result sampling for consistency verification
4. Mixed dataset warnings
5. Edge cases and various ID formats
"""

import pytest
from run_experiment import detect_dataset_type


class TestScienceDatasetDetection:
    """Tests for science dataset detection"""

    def test_detect_science_dataset_from_gpqa_id(self):
        """Test detection of science dataset from GPQA unique_id"""
        results = [
            {'unique_id': 'gpqa-001', 'question': 'Science question', 'answer': 'A'},
            {'unique_id': 'gpqa-002', 'question': 'Science question', 'answer': 'B'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'science'

    def test_detect_science_dataset_from_gsm8k_id(self):
        """Test detection of science dataset from GSM8K unique_id"""
        results = [
            {'unique_id': 'gsm8k-001', 'question': 'Math reasoning', 'answer': '42'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'science'

    def test_detect_science_dataset_from_answer_structure(self):
        """Test detection of science dataset from single-letter answer (A/B/C/D)"""
        results = [
            {'unique_id': 'unknown-001', 'question': 'Question', 'answer': 'A'},
            {'unique_id': 'unknown-002', 'question': 'Question', 'answer': 'B'},
            {'unique_id': 'unknown-003', 'question': 'Question', 'answer': 'C'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'science'


class TestStructureBasedDetection:
    """Tests for detection based on result structure"""

    def test_detect_code_from_test_cases_structure(self):
        """Test detection of code dataset from test_cases presence"""
        results = [
            {
                'unique_id': 'unknown-dataset-1',
                'question': 'Write code',
                'test_cases': [(['1', '2'], '3'), (['2', '3'], '5')]
            },
            {
                'unique_id': 'unknown-dataset-2',
                'question': 'Write code',
                'test_cases': [(['5', '5'], '10')]
            }
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'code'

    def test_detect_math_from_numeric_answer(self):
        """Test detection of math dataset from numeric answer"""
        results = [
            {'unique_id': 'unknown-001', 'question': 'Calculate', 'answer': '42'},
            {'unique_id': 'unknown-002', 'question': 'Calculate', 'answer': '12'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'


class TestMultipleResultSampling:
    """Tests for sampling multiple results for consistency"""

    def test_consistent_type_across_sample(self):
        """Test that consistent types across sample are detected correctly"""
        results = [
            {'unique_id': 'codeelo-1-A-1', 'test_cases': [(['1'], '1')]},
            {'unique_id': 'codeelo-1-B-2', 'test_cases': [(['2'], '2')]},
            {'unique_id': 'codeelo-1-C-3', 'test_cases': [(['3'], '3')]},
            {'unique_id': 'codeelo-1-D-4', 'test_cases': [(['4'], '4')]}  # More results
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'code'

    def test_samples_first_three_results(self):
        """Test that function samples first 3 results (or all if fewer)"""
        results = [
            {'unique_id': 'aime2025-I-1', 'answer': '10'},
            {'unique_id': 'aime2025-I-2', 'answer': '20'},
            {'unique_id': 'aime2025-I-3', 'answer': '30'},
            # If more samples were checked, might detect inconsistency
            {'unique_id': 'codeelo-1-A-1', 'test_cases': [(['1'], '1')]}
        ]

        dataset_type = detect_dataset_type(results)

        # Should detect as math (first 3 are consistent)
        assert dataset_type == 'math'

    def test_works_with_single_result(self):
        """Test that detection works with single result"""
        results = [
            {'unique_id': 'codeelo-123-A-0', 'test_cases': [(['1'], '1')]}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'code'

    def test_works_with_two_results(self):
        """Test that detection works with exactly two results"""
        results = [
            {'unique_id': 'gpqa-001', 'answer': 'A'},
            {'unique_id': 'gpqa-002', 'answer': 'B'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'science'


class TestCaseInsensitivity:
    """Tests for case-insensitive detection"""

    def test_uppercase_id_detection(self):
        """Test detection works with uppercase IDs"""
        results = [
            {'unique_id': 'CODEFORCES-1234-A-0', 'test_cases': [(['1'], '1')]}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'code'

    def test_mixed_case_id_detection(self):
        """Test detection works with mixed case IDs"""
        results = [
            {'unique_id': 'GPQA-Diamond-001', 'answer': 'A'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'science'

    def test_single_letter_answer_uppercase(self):
        """Test detection of single-letter answer (uppercase)"""
        results = [
            {'unique_id': 'unknown-001', 'answer': 'A'},
            {'unique_id': 'unknown-002', 'answer': 'C'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'science'

    def test_single_letter_answer_lowercase(self):
        """Test detection of single-letter answer (lowercase)"""
        results = [
            {'unique_id': 'unknown-001', 'answer': 'a'},
            {'unique_id': 'unknown-002', 'answer': 'b'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'science'


class TestExtendedPatterns:
    """Tests for extended dataset patterns"""

    def test_detect_codeeml(self):
        """Test detection of CodeELM-like dataset"""
        results = [
            {'unique_id': 'codeeml-001', 'test_cases': [(['a'], 'b')]}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'code'

    def test_detect_codeelo(self):
        """Test detection of CodeElo dataset"""
        results = [
            {'unique_id': 'codeelo-001', 'test_cases': [(['1'], '2')]}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'code'

    def test_detect_mathdialogue(self):
        """Test detection of MathDialogue dataset"""
        results = [
            {'unique_id': 'mathdialogue-001', 'answer': '100'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_unique_id(self):
        """Test handling of empty unique_id"""
        results = [
            {'unique_id': '', 'answer': '42'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'  # Defaults to math

    def test_missing_unique_id(self):
        """Test handling of missing unique_id"""
        results = [
            {'question': 'Problem', 'answer': '100'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'  # Defaults to math

    def test_missing_answer_field(self):
        """Test handling of result without answer field"""
        results = [
            {'unique_id': 'unknown-001', 'question': 'Problem'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'  # Defaults to math

    def test_non_letter_single_answer(self):
        """Test that numeric single answer is not treated as science"""
        results = [
            {'unique_id': 'unknown-001', 'answer': '1'},
            {'unique_id': 'unknown-002', 'answer': '2'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'  # Numbers are math, not science

    def test_mixed_answers_with_letters(self):
        """Test mixed answers including letters"""
        results = [
            {'unique_id': 'unknown-001', 'answer': 'A'},
            {'unique_id': 'unknown-002', 'answer': 'B'},
            {'unique_id': 'unknown-003', 'answer': 'E'}  # E is not valid
        ]

        dataset_type = detect_dataset_type(results)

        # First three samples consistently show A/B for science
        assert dataset_type == 'science'


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with old behavior"""

    def test_original_math_detection(self):
        """Test original math detection still works"""
        results = [
            {'unique_id': 'aime2025-I-0-1', 'answer': '4'},
            {'unique_id': 'aime2025-I-0-2', 'answer': '8'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'

    def test_original_code_detection(self):
        """Test original code detection still works"""
        results = [
            {'unique_id': 'codeelo-1234-A-0', 'test_cases': []},
            {'unique_id': 'codeelo-1234-B-0', 'test_cases': []}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'code'

    def test_original_unknown_defaults_to_math(self):
        """Test original behavior of unknown type defaulting to math"""
        results = [
            {'unique_id': 'completely-unknown-xyz', 'answer': '100'}
        ]

        dataset_type = detect_dataset_type(results)

        assert dataset_type == 'math'
