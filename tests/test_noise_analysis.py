"""
Unit tests for run_noise_analysis.py

Tests noise impact analysis functionality including:
- Noise detection in distributions
- Probability and rank calculations
- Statistics aggregation
"""

import pytest
from run_noise_analysis import (
    normalize_answer,
    analyze_noise_in_distribution,
    calculate_noise_statistics,
    analyze_per_question
)


class TestNormalizeAnswer:
    """Tests for normalize_answer function"""

    def test_normalize_integer(self):
        """Test normalization of integers"""
        assert normalize_answer("42") == "42"
        assert normalize_answer("123") == "123"

    def test_normalize_fraction(self):
        """Test normalization of fractions"""
        assert normalize_answer("2/4") == "1/2"
        assert normalize_answer("3/6") == "1/2"

    def test_normalize_float_to_integer(self):
        """Test normalization of floats that are whole numbers"""
        assert normalize_answer("42.0") == "42"

    def test_normalize_with_whitespace(self):
        """Test normalization strips whitespace"""
        assert normalize_answer("  42  ") == "42"


class TestAnalyzeNoiseInDistribution:
    """Tests for analyze_noise_in_distribution function"""

    def test_noise_found_at_rank_1(self):
        """Test when noise is found at rank 1"""
        distribution = [
            {'answer': '123', 'normalized_answer': '123', 'percentage': 0.5, 'rank': 1},
            {'answer': '42', 'normalized_answer': '42', 'percentage': 0.3, 'rank': 2},
        ]
        prob, rank = analyze_noise_in_distribution(distribution, '123')
        assert prob == 0.5
        assert rank == 1

    def test_noise_found_at_rank_2(self):
        """Test when noise is found at rank 2"""
        distribution = [
            {'answer': '42', 'normalized_answer': '42', 'percentage': 0.6, 'rank': 1},
            {'answer': '123', 'normalized_answer': '123', 'percentage': 0.3, 'rank': 2},
        ]
        prob, rank = analyze_noise_in_distribution(distribution, '123')
        assert prob == 0.3
        assert rank == 2

    def test_noise_not_found(self):
        """Test when noise is not in distribution"""
        distribution = [
            {'answer': '42', 'normalized_answer': '42', 'percentage': 0.6, 'rank': 1},
            {'answer': '43', 'normalized_answer': '43', 'percentage': 0.4, 'rank': 2},
        ]
        prob, rank = analyze_noise_in_distribution(distribution, '123')
        assert prob == 0.0
        assert rank == 3  # max_rank + 1

    def test_empty_distribution(self):
        """Test with empty distribution"""
        distribution = []
        prob, rank = analyze_noise_in_distribution(distribution, '123')
        assert prob is None
        assert rank is None

    def test_noise_normalization(self):
        """Test that noise answer is normalized for comparison"""
        distribution = [
            {'answer': '2/4', 'normalized_answer': '1/2', 'percentage': 0.5, 'rank': 1},
        ]
        # Search for equivalent answer "1/2"
        prob, rank = analyze_noise_in_distribution(distribution, '2/4')
        assert prob == 0.5
        assert rank == 1


class TestCalculateNoiseStatistics:
    """Tests for calculate_noise_statistics function"""

    def test_basic_statistics(self):
        """Test basic noise statistics calculation"""
        results = [
            {
                'distribution': [
                    {'answer': '123', 'normalized_answer': '123', 'percentage': 0.6, 'rank': 1},
                    {'answer': '42', 'normalized_answer': '42', 'percentage': 0.4, 'rank': 2},
                ]
            },
            {
                'distribution': [
                    {'answer': '42', 'normalized_answer': '42', 'percentage': 0.5, 'rank': 1},
                    {'answer': '123', 'normalized_answer': '123', 'percentage': 0.3, 'rank': 2},
                ]
            },
        ]

        stats = calculate_noise_statistics(results, '123')

        assert stats['total_questions'] == 2
        assert stats['questions_with_noise'] == 2
        assert stats['noise_appearance_rate'] == 1.0
        assert abs(stats['avg_noise_prob'] - 0.45) < 0.01  # (0.6 + 0.3) / 2
        assert abs(stats['mean_noise_rank'] - 1.5) < 0.01  # (1 + 2) / 2
        # Std dev: sqrt(((1-1.5)^2 + (2-1.5)^2) / 2) = sqrt(0.25) = 0.5
        assert abs(stats['std_noise_rank'] - 0.5) < 0.01

    def test_noise_not_in_some_distributions(self):
        """Test when noise doesn't appear in all distributions"""
        results = [
            {
                'distribution': [
                    {'answer': '123', 'normalized_answer': '123', 'percentage': 0.6, 'rank': 1},
                ]
            },
            {
                'distribution': [
                    {'answer': '42', 'normalized_answer': '42', 'percentage': 1.0, 'rank': 1},
                ]
            },
        ]

        stats = calculate_noise_statistics(results, '123')

        assert stats['total_questions'] == 2
        assert stats['questions_with_noise'] == 1
        assert stats['noise_appearance_rate'] == 0.5
        assert stats['avg_noise_prob'] == 0.3  # (0.6 + 0.0) / 2

    def test_empty_results(self):
        """Test with empty results list"""
        results = []
        stats = calculate_noise_statistics(results, '123')

        assert stats['total_questions'] == 0
        assert stats['avg_noise_prob'] is None
        assert stats['mean_noise_rank'] is None
        assert stats['std_noise_rank'] is None

    def test_single_question_std_dev(self):
        """Test that single question has zero std dev"""
        results = [
            {
                'distribution': [
                    {'answer': '123', 'normalized_answer': '123', 'percentage': 0.6, 'rank': 1},
                ]
            },
        ]

        stats = calculate_noise_statistics(results, '123')

        assert stats['std_noise_rank'] == 0.0


class TestAnalyzePerQuestion:
    """Tests for analyze_per_question function"""

    def test_per_question_analysis(self):
        """Test per-question analysis"""
        results = [
            {
                'unique_id': 'q1',
                'question_id': 1,
                'distribution': [
                    {'answer': '123', 'normalized_answer': '123', 'percentage': 0.6, 'rank': 1},
                ]
            },
            {
                'unique_id': 'q2',
                'question_id': 2,
                'distribution': [
                    {'answer': '42', 'normalized_answer': '42', 'percentage': 1.0, 'rank': 1},
                ]
            },
        ]

        analysis = analyze_per_question(results, '123')

        assert len(analysis) == 2

        # First question has noise
        assert analysis[0]['unique_id'] == 'q1'
        assert analysis[0]['question_id'] == 1
        assert analysis[0]['noise_prob'] == 0.6
        assert analysis[0]['noise_rank'] == 1
        assert analysis[0]['noise_in_distribution'] is True

        # Second question doesn't have noise
        assert analysis[1]['unique_id'] == 'q2'
        assert analysis[1]['question_id'] == 2
        assert analysis[1]['noise_prob'] == 0.0
        assert analysis[1]['noise_rank'] == 2  # max_rank + 1
        assert analysis[1]['noise_in_distribution'] is False

    def test_per_question_empty_distributions(self):
        """Test that empty distributions are skipped"""
        results = [
            {
                'unique_id': 'q1',
                'question_id': 1,
                'distribution': []
            },
            {
                'unique_id': 'q2',
                'question_id': 2,
                'distribution': [
                    {'answer': '123', 'normalized_answer': '123', 'percentage': 1.0, 'rank': 1},
                ]
            },
        ]

        analysis = analyze_per_question(results, '123')

        # Only second question should be analyzed
        assert len(analysis) == 1
        assert analysis[0]['unique_id'] == 'q2'


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""

    def test_noise_insertion_experiment(self):
        """Test analysis of noise insertion experiment"""
        # Simulate experiment where noise '123' was inserted into reasoning
        results = [
            {
                'unique_id': 'q1',
                'question_id': 1,
                'distribution': [
                    {'answer': '42', 'normalized_answer': '42', 'percentage': 0.5, 'rank': 1, 'is_correct': True},
                    {'answer': '123', 'normalized_answer': '123', 'percentage': 0.3, 'rank': 2, 'is_correct': False},
                    {'answer': '999', 'normalized_answer': '999', 'percentage': 0.2, 'rank': 3, 'is_correct': False},
                ]
            },
            {
                'unique_id': 'q2',
                'question_id': 2,
                'distribution': [
                    {'answer': '123', 'normalized_answer': '123', 'percentage': 0.6, 'rank': 1, 'is_correct': False},
                    {'answer': '17', 'normalized_answer': '17', 'percentage': 0.4, 'rank': 2, 'is_correct': True},
                ]
            },
            {
                'unique_id': 'q3',
                'question_id': 3,
                'distribution': [
                    {'answer': '5', 'normalized_answer': '5', 'percentage': 1.0, 'rank': 1, 'is_correct': True},
                ]
            },
        ]

        stats = calculate_noise_statistics(results, '123')

        # Verify statistics
        assert stats['total_questions'] == 3
        assert stats['questions_with_noise'] == 2
        assert abs(stats['noise_appearance_rate'] - 0.666) < 0.01
        assert abs(stats['avg_noise_prob'] - 0.3) < 0.01  # (0.3 + 0.6 + 0.0) / 3
        assert abs(stats['mean_noise_rank'] - 1.667) < 0.01  # (2 + 1 + 2) / 3 = 5/3 ≈ 1.67

        # Per-question analysis
        per_q = analyze_per_question(results, '123')
        assert len(per_q) == 3

        # Question 2 should show noise at rank 1 (strongest noise effect)
        q2_analysis = [q for q in per_q if q['question_id'] == 2][0]
        assert q2_analysis['noise_rank'] == 1
        assert q2_analysis['noise_prob'] == 0.6
