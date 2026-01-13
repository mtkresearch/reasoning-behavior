"""
Tests for automatic output path generation
"""

import pytest
import hashlib


def compute_flow_hash(flow: str) -> str:
    """Compute 8-character hash from flow string"""
    hash_obj = hashlib.md5(flow.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


class TestGenerateOutputPathFromFlow:
    """Tests for generate_output_path_from_flow function"""

    def test_single_mask_processor(self):
        """Test path generation with single mask processor"""
        from run_experiment import generate_output_path_from_flow

        flow = "mask('number')"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_mask_and_shuffle(self):
        """Test path generation with mask and shuffle"""
        from run_experiment import generate_output_path_from_flow

        flow = "mask('number'),shuffle('line')"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_full_pipeline(self):
        """Test path generation with full pipeline"""
        from run_experiment import generate_output_path_from_flow

        flow = "truncate('answer_and_after'),mask('number'),shuffle('line')"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_nlines_mode(self):
        """Test path generation with n-lines mode"""
        from run_experiment import generate_output_path_from_flow

        flow = "mask('n-lines')"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_empty_flow(self):
        """Test path generation with empty flow"""
        from run_experiment import generate_output_path_from_flow

        path = generate_output_path_from_flow("dummy.json", "")
        expected_hash = compute_flow_hash("")

        assert path == f"exp/{expected_hash}/results.json"

    def test_truncate_ratio(self):
        """Test path generation with ratio truncation"""
        from run_experiment import generate_output_path_from_flow

        flow = "truncate('last_ratio',ratio=0.3)"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_word_shuffle(self):
        """Test path generation with word shuffle"""
        from run_experiment import generate_output_path_from_flow

        flow = "shuffle('word')"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_alphabet_mask(self):
        """Test path generation with alphabet mask"""
        from run_experiment import generate_output_path_from_flow

        flow = "mask('alphabet')"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_complex_pipeline(self):
        """Test path generation with complex pipeline"""
        from run_experiment import generate_output_path_from_flow

        flow = "truncate('last_ratio',ratio=0.3),mask('number-advance'),shuffle('token')"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_insert_processor(self):
        """Test path generation with insert processor"""
        from run_experiment import generate_output_path_from_flow

        flow = "insert('fix',sentence='Answer: 123.',count=5)"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"

    def test_insert_with_shuffle(self):
        """Test path generation with insert and shuffle"""
        from run_experiment import generate_output_path_from_flow

        flow = "insert('fix',sentence='Noise',count=3),shuffle('line')"
        path = generate_output_path_from_flow("dummy.json", flow)
        expected_hash = compute_flow_hash(flow)

        assert path == f"exp/{expected_hash}/results.json"
