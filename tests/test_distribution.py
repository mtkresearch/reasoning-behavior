#!/usr/bin/env python3
"""
Unit tests for run_distribution.py

Tests cover:
- Answer extraction from boxed format
- Answer normalization using sympy
- Distribution calculation
- Request building logic
"""

import pytest
from run_distribution import (
    extract_answer_from_completion,
    normalize_answer,
    calculate_distribution,
    calculate_entropy,
    build_sampling_request
)


class TestExtractAnswerFromCompletion:
    """Tests for extract_answer_from_completion function"""

    def test_extract_simple_boxed_with_dollar(self):
        """Should extract answer from $\\boxed{42}$ format"""
        completion = "The answer is $\\boxed{42}$"
        result = extract_answer_from_completion(completion)
        assert result == "42"

    def test_extract_simple_boxed_without_dollar(self):
        """Should extract answer from \\boxed{42} format without $ symbols"""
        completion = "The answer is \\boxed{42}"
        result = extract_answer_from_completion(completion)
        assert result == "42"

    def test_extract_last_occurrence(self):
        """Should extract the last boxed answer when multiple exist"""
        completion = "First $\\boxed{10}$ then $\\boxed{20}$"
        result = extract_answer_from_completion(completion)
        assert result == "20"

    def test_extract_nested_braces(self):
        """Should extract only numeric part from nested braces"""
        completion = "The answer is $\\boxed{\\frac{1}{2}}$"
        result = extract_answer_from_completion(completion)
        # Function only extracts digits, so it gets '1' and '2', returns last one
        assert result == "2"

    def test_no_boxed_format(self):
        """Should return None when no boxed format found"""
        completion = "No answer here"
        result = extract_answer_from_completion(completion)
        assert result is None

    def test_empty_boxed(self):
        """Should return None for empty boxed content"""
        completion = "The answer is $\\boxed{}$"
        result = extract_answer_from_completion(completion)
        assert result is None

    def test_whitespace_in_boxed(self):
        """Should strip whitespace from extracted answer"""
        completion = "The answer is $\\boxed{  42  }$"
        result = extract_answer_from_completion(completion)
        assert result == "42"

    def test_complex_mathematical_expression(self):
        """Should extract only the last numeric value from complex expressions"""
        completion = "Therefore $\\boxed{2x + 3y = 7}$"
        result = extract_answer_from_completion(completion)
        # Function only extracts digits, finds '2', '3', '7', returns last one
        assert result == "7"


class TestNormalizeAnswer:
    """Tests for normalize_answer function"""

    def test_normalize_fraction(self):
        """Should normalize equivalent fractions to same form"""
        assert normalize_answer("2/4") == "1/2"
        assert normalize_answer("3/6") == "1/2"
        assert normalize_answer("4/8") == "1/2"

    def test_normalize_float_to_integer(self):
        """Should convert whole number floats to integers"""
        assert normalize_answer("42.0") == "42"
        assert normalize_answer("100.0") == "100"

    def test_normalize_preserves_decimals(self):
        """Should preserve non-whole number decimals (with sympy precision)"""
        result = normalize_answer("3.14159")
        # Sympy may add trailing zeros for floating point representation
        assert result.startswith("3.14159")

    def test_normalize_keeps_original_on_failure(self):
        """Should return original string if normalization fails"""
        result = normalize_answer("not a number")
        assert result == "not a number"

    def test_normalize_none_input(self):
        """Should return None for None input"""
        result = normalize_answer(None)
        assert result is None

    def test_normalize_empty_string(self):
        """Should return None for empty string"""
        result = normalize_answer("")
        assert result is None

    def test_normalize_whitespace(self):
        """Should strip whitespace before normalization"""
        assert normalize_answer("  1/2  ") == "1/2"

    def test_normalize_latex_fraction(self):
        """Should handle LaTeX fraction format"""
        result = normalize_answer("\\frac{1}{2}")
        assert result == "1/2"

    def test_normalize_algebraic_expression(self):
        """Should simplify algebraic expressions"""
        assert normalize_answer("x + x") == "2*x"
        assert normalize_answer("2*x + 3*x") == "5*x"


class TestCalculateDistribution:
    """Tests for calculate_distribution function"""

    def test_simple_distribution(self):
        """Should calculate correct distribution for simple samples"""
        samples = [
            {"extracted_answer": "42"},
            {"extracted_answer": "42"},
            {"extracted_answer": "43"},
        ]
        result = calculate_distribution(samples, "42")

        assert len(result) == 2
        assert result[0]["count"] == 2
        assert result[0]["percentage"] == 2/3
        assert result[0]["is_correct"] is True
        assert result[0]["rank"] == 1

        assert result[1]["count"] == 1
        assert result[1]["percentage"] == 1/3
        assert result[1]["is_correct"] is False
        assert result[1]["rank"] == 2

    def test_distribution_with_none_answers(self):
        """Should filter out None answers from distribution"""
        samples = [
            {"extracted_answer": "42"},
            {"extracted_answer": None},
            {"extracted_answer": "42"},
        ]
        result = calculate_distribution(samples, "42")

        assert len(result) == 1
        assert result[0]["count"] == 2
        assert result[0]["percentage"] == 1.0  # 100% of valid samples

    def test_distribution_with_equivalent_answers(self):
        """Should group equivalent normalized answers together"""
        samples = [
            {"extracted_answer": "1/2"},
            {"extracted_answer": "2/4"},
            {"extracted_answer": "3/6"},
        ]
        result = calculate_distribution(samples, "1/2")

        # All should be grouped as equivalent
        assert len(result) == 1
        assert result[0]["count"] == 3
        assert result[0]["is_correct"] is True

    def test_distribution_all_none(self):
        """Should return empty list when all answers are None"""
        samples = [
            {"extracted_answer": None},
            {"extracted_answer": None},
        ]
        result = calculate_distribution(samples, "42")

        assert result == []

    def test_distribution_correctness_check(self):
        """Should correctly identify correct answers using normalization"""
        samples = [
            {"extracted_answer": "2/4"},  # Equivalent to 1/2
            {"extracted_answer": "42"},
        ]
        result = calculate_distribution(samples, "1/2")

        correct_entry = next(r for r in result if r["is_correct"])
        assert correct_entry["normalized_answer"] == "1/2"

    def test_distribution_ranking_by_frequency(self):
        """Should rank answers by frequency (most common first)"""
        samples = [
            {"extracted_answer": "A"},
            {"extracted_answer": "B"},
            {"extracted_answer": "B"},
            {"extracted_answer": "C"},
            {"extracted_answer": "C"},
            {"extracted_answer": "C"},
        ]
        result = calculate_distribution(samples, "A")

        # Should be sorted by count descending
        assert result[0]["answer"] == "C"
        assert result[0]["count"] == 3
        assert result[0]["rank"] == 1

        assert result[1]["answer"] == "B"
        assert result[1]["count"] == 2
        assert result[1]["rank"] == 2

        assert result[2]["answer"] == "A"
        assert result[2]["count"] == 1
        assert result[2]["rank"] == 3

    def test_distribution_uses_original_answer_for_display(self):
        """Should use first occurrence of raw answer for display"""
        samples = [
            {"extracted_answer": "2/4"},
            {"extracted_answer": "1/2"},
            {"extracted_answer": "3/6"},
        ]
        result = calculate_distribution(samples, "1/2")

        # Should display first occurrence ("2/4")
        assert result[0]["answer"] == "2/4"
        assert result[0]["normalized_answer"] == "1/2"

    def test_distribution_dense_ranking_same_count(self):
        """Should use dense ranking: same count gets same rank"""
        samples = [
            {"extracted_answer": "A"},
            {"extracted_answer": "A"},
            {"extracted_answer": "A"},  # count=3, rank=1
            {"extracted_answer": "B"},
            {"extracted_answer": "B"},  # count=2, rank=2
            {"extracted_answer": "C"},
            {"extracted_answer": "C"},  # count=2, rank=2 (same as B)
            {"extracted_answer": "D"},  # count=1, rank=3 (not 4!)
        ]
        result = calculate_distribution(samples, "X")

        # Find each answer in result
        rank_a = next(r["rank"] for r in result if r["answer"] == "A")
        rank_b = next(r["rank"] for r in result if r["answer"] == "B")
        rank_c = next(r["rank"] for r in result if r["answer"] == "C")
        rank_d = next(r["rank"] for r in result if r["answer"] == "D")

        # Dense ranking: 1, 2, 2, 3 (not 1, 2, 3, 4)
        assert rank_a == 1
        assert rank_b == 2
        assert rank_c == 2
        assert rank_d == 3

    def test_distribution_dense_ranking_all_same_count(self):
        """Should assign rank=1 to all answers with same count"""
        samples = [
            {"extracted_answer": "A"},
            {"extracted_answer": "B"},
            {"extracted_answer": "C"},
        ]
        result = calculate_distribution(samples, "X")

        # All have count=1, so all should have rank=1
        assert all(r["rank"] == 1 for r in result)


class TestCalculateEntropy:
    """Tests for calculate_entropy function"""

    def test_entropy_uniform_distribution(self):
        """Should calculate maximum entropy for uniform distribution"""
        # Two equally likely outcomes: entropy = 1.0 bit
        dist = [
            {'answer': 'A', 'percentage': 0.5},
            {'answer': 'B', 'percentage': 0.5}
        ]
        entropy = calculate_entropy(dist)
        assert abs(entropy - 1.0) < 0.01

    def test_entropy_single_answer(self):
        """Should return 0 entropy for single certain answer"""
        dist = [{'answer': 'A', 'percentage': 1.0}]
        entropy = calculate_entropy(dist)
        assert entropy == 0.0

    def test_entropy_empty_distribution(self):
        """Should return 0 for empty distribution"""
        entropy = calculate_entropy([])
        assert entropy == 0

    def test_entropy_skewed_distribution(self):
        """Should calculate correct entropy for skewed distribution"""
        # 90% one answer, 10% another
        dist = [
            {'answer': 'A', 'percentage': 0.9},
            {'answer': 'B', 'percentage': 0.1}
        ]
        entropy = calculate_entropy(dist)
        # Entropy should be low but non-zero
        assert 0 < entropy < 1.0
        # Verify it's less than uniform distribution
        assert entropy < 0.5

    def test_entropy_three_way_uniform(self):
        """Should calculate correct entropy for three-way uniform distribution"""
        import math
        dist = [
            {'answer': 'A', 'percentage': 1/3},
            {'answer': 'B', 'percentage': 1/3},
            {'answer': 'C', 'percentage': 1/3}
        ]
        entropy = calculate_entropy(dist)
        expected = math.log2(3)  # log2(3) ≈ 1.585
        assert abs(entropy - expected) < 0.01

    def test_entropy_ignores_zero_percentage(self):
        """Should handle zero percentages correctly"""
        dist = [
            {'answer': 'A', 'percentage': 0.7},
            {'answer': 'B', 'percentage': 0.3},
            {'answer': 'C', 'percentage': 0.0}  # Should be ignored
        ]
        # Should not crash due to log2(0)
        entropy = calculate_entropy(dist)
        assert entropy > 0


class TestBuildSamplingRequest:
    """Tests for build_sampling_request function"""

    def test_build_request_answer_free_gen(self):
        """Should build correct request for answer-free generation"""
        task = build_sampling_request(
            question="What is 2+2?",
            processed_reasoning="Let's think step by step...",
            answer_free_gen=True,
            model_type="gpt-oss",
            temperature=0.5,
            task_index=0
        )

        # Verify task properties
        assert task.index == 0
        assert task.request.model_type == "gpt-oss"
        assert task.request.temperature == 0.5
        assert task.request.max_tokens == 3000  # Answer-free gen uses 3000
        assert "What is 2+2?" in task.request.prompt
        assert "Let's think step by step..." in task.request.prompt

    def test_build_request_with_prefill(self):
        """Should build correct request with answer prefill"""
        task = build_sampling_request(
            question="What is 2+2?",
            processed_reasoning="Let's think step by step...",
            answer_free_gen=False,
            model_type="gpt-oss",
            temperature=0.7,
            task_index=5
        )

        # Verify task properties
        assert task.index == 5
        assert task.request.model_type == "gpt-oss"
        assert task.request.temperature == 0.7
        assert task.request.max_tokens == 20  # With prefill uses 20 (short numeric answer)
        assert "What is 2+2?" in task.request.prompt
        assert "Thus, the answer is" in task.request.prompt

    def test_build_request_with_metadata(self):
        """Should attach metadata to the task"""
        metadata = {"unique_id": "test_123", "sample_id": 42}
        task = build_sampling_request(
            question="Test question",
            processed_reasoning="Test reasoning",
            answer_free_gen=False,
            model_type="gpt-oss",
            temperature=0.5,
            task_index=0,
            metadata=metadata
        )

        assert task.metadata == metadata
        assert task.metadata["unique_id"] == "test_123"
        assert task.metadata["sample_id"] == 42

    def test_build_request_without_metadata(self):
        """Should use empty dict when no metadata provided"""
        task = build_sampling_request(
            question="Test question",
            processed_reasoning="Test reasoning",
            answer_free_gen=False,
            model_type="gpt-oss",
            temperature=0.5,
            task_index=0
        )

        assert task.metadata == {}


class TestCalculateDistributionStats:
    """Tests for calculate_distribution_stats function"""

    def test_all_stats_calculation(self):
        """Should calculate all statistics: entropy, answer_prob, mean_rank, std_rank, top1_acc"""
        results = [
            {
                "distribution": [
                    {"answer": "42", "count": 10, "percentage": 0.5, "is_correct": True, "rank": 1},
                    {"answer": "43", "count": 10, "percentage": 0.5, "is_correct": False, "rank": 1},
                ]
            },
            {
                "distribution": [
                    {"answer": "100", "count": 8, "percentage": 0.8, "is_correct": False, "rank": 1},
                    {"answer": "150", "count": 2, "percentage": 0.2, "is_correct": True, "rank": 2},
                ]
            },
        ]

        from run_distribution import calculate_distribution_stats
        stats = calculate_distribution_stats(results)

        # Check all keys present
        assert 'avg_entropy' in stats
        assert 'avg_answer_prob' in stats
        assert 'mean_rank' in stats
        assert 'std_rank' in stats
        assert 'top1_accuracy' in stats

        # Check rank stats (should match previous tests)
        # Ranks: [1, 2], Mean: 1.5, Std: 0.5
        assert abs(stats['mean_rank'] - 1.5) < 0.01
        assert abs(stats['std_rank'] - 0.5) < 0.01

        # Check answer probability
        # Answer probs: [0.5, 0.2], Mean: 0.35
        assert abs(stats['avg_answer_prob'] - 0.35) < 0.01

        # Check entropy
        # Question 1: entropy = 1.0 (uniform)
        # Question 2: entropy ≈ 0.72
        # Average ≈ 0.86
        assert stats['avg_entropy'] > 0

        # Check top-1 accuracy
        # Question 1: correct answer at rank 1 (True)
        # Question 2: correct answer at rank 2 (False)
        # Top-1 accuracy: 1/2 = 0.5
        assert abs(stats['top1_accuracy'] - 0.5) < 0.01

    def test_correct_answer_not_in_distribution_for_prob(self):
        """Should use 0 probability when correct answer not in distribution"""
        results = [
            {
                "distribution": [
                    {"answer": "42", "count": 10, "percentage": 1.0, "is_correct": False, "rank": 1},
                ]
            },
        ]

        from run_distribution import calculate_distribution_stats
        stats = calculate_distribution_stats(results)

        # Answer probability should be 0
        assert stats['avg_answer_prob'] == 0.0
        # Rank should be 2 (max_rank + 1)
        assert stats['mean_rank'] == 2.0
        # Top-1 accuracy should be 0 (correct answer not at rank 1)
        assert stats['top1_accuracy'] == 0.0

    def test_empty_results(self):
        """Should return None for all stats when no valid distributions"""
        results = [
            {"distribution": []},
        ]

        from run_distribution import calculate_distribution_stats
        stats = calculate_distribution_stats(results)

        assert stats['avg_entropy'] is None
        assert stats['avg_answer_prob'] is None
        assert stats['mean_rank'] is None
        assert stats['std_rank'] is None
        assert stats['top1_accuracy'] is None

    def test_all_correct_at_rank_one(self):
        """Should return top1_accuracy=1.0 when all correct answers are at rank 1"""
        results = [
            {
                "distribution": [
                    {"answer": "42", "count": 10, "percentage": 0.6, "is_correct": True, "rank": 1},
                    {"answer": "43", "count": 5, "percentage": 0.4, "is_correct": False, "rank": 2},
                ]
            },
            {
                "distribution": [
                    {"answer": "100", "count": 8, "percentage": 0.8, "is_correct": True, "rank": 1},
                    {"answer": "150", "count": 2, "percentage": 0.2, "is_correct": False, "rank": 2},
                ]
            },
        ]

        from run_distribution import calculate_distribution_stats
        stats = calculate_distribution_stats(results)

        # All correct answers at rank 1
        assert stats['top1_accuracy'] == 1.0
        assert stats['mean_rank'] == 1.0
        assert stats['std_rank'] == 0.0


class TestCalculateAnswerRankStats:
    """Tests for calculate_answer_rank_stats function"""

    def test_simple_rank_stats(self):
        """Should calculate correct mean and std for simple case"""
        # Mock results with distributions
        results = [
            {
                "distribution": [
                    {"answer": "42", "count": 10, "is_correct": True, "rank": 1},
                    {"answer": "43", "count": 5, "is_correct": False, "rank": 2},
                ]
            },
            {
                "distribution": [
                    {"answer": "100", "count": 8, "is_correct": False, "rank": 1},
                    {"answer": "150", "count": 6, "is_correct": True, "rank": 2},
                ]
            },
            {
                "distribution": [
                    {"answer": "7", "count": 10, "is_correct": False, "rank": 1},
                    {"answer": "9", "count": 9, "is_correct": True, "rank": 2},
                ]
            },
        ]

        from run_distribution import calculate_answer_rank_stats
        mean, std = calculate_answer_rank_stats(results)

        # Ranks: [1, 2, 2]
        # Mean: (1 + 2 + 2) / 3 = 5/3 = 1.667
        # Variance: [(1-1.667)^2 + (2-1.667)^2 + (2-1.667)^2] / 3
        #         = [0.444 + 0.111 + 0.111] / 3 = 0.222
        # Std: sqrt(0.222) = 0.471
        assert abs(mean - 5/3) < 0.01
        assert abs(std - 0.471) < 0.01

    def test_correct_answer_not_in_distribution(self):
        """Should use max_rank + 1 when correct answer not in distribution"""
        results = [
            {
                "distribution": [
                    {"answer": "42", "count": 10, "is_correct": False, "rank": 1},
                    {"answer": "43", "count": 5, "is_correct": False, "rank": 2},
                    {"answer": "44", "count": 2, "is_correct": False, "rank": 3},
                ]
            },
        ]

        from run_distribution import calculate_answer_rank_stats
        mean, std = calculate_answer_rank_stats(results)

        # Rank: [4] (max_rank=3, so 3+1=4)
        # Mean: 4.0
        # Std: 0.0 (only one value)
        assert mean == 4.0
        assert std == 0.0

    def test_empty_distribution(self):
        """Should skip questions with empty distribution"""
        results = [
            {"distribution": []},
            {
                "distribution": [
                    {"answer": "42", "count": 10, "is_correct": True, "rank": 1},
                ]
            },
        ]

        from run_distribution import calculate_answer_rank_stats
        mean, std = calculate_answer_rank_stats(results)

        # Only second result counted: rank = [1]
        assert mean == 1.0
        assert std == 0.0

    def test_all_empty_distributions(self):
        """Should return None when all distributions are empty"""
        results = [
            {"distribution": []},
            {"distribution": []},
        ]

        from run_distribution import calculate_answer_rank_stats
        mean, std = calculate_answer_rank_stats(results)

        assert mean is None
        assert std is None

    def test_all_correct_at_rank_one(self):
        """Should return mean=1, std=0 when all correct answers are rank 1"""
        results = [
            {
                "distribution": [
                    {"answer": "42", "count": 10, "is_correct": True, "rank": 1},
                ]
            },
            {
                "distribution": [
                    {"answer": "100", "count": 8, "is_correct": True, "rank": 1},
                ]
            },
        ]

        from run_distribution import calculate_answer_rank_stats
        mean, std = calculate_answer_rank_stats(results)

        assert mean == 1.0
        assert std == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
