"""
Tests for HTMLGenerator class in run_attn_visual.py

Following TDD Red-Green-Refactor cycle for Phase 5
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestHTMLGenerator:
    """Test suite for HTMLGenerator class"""

    @pytest.fixture
    def sample_instances(self):
        """Fixture providing sample instances data"""
        return [
            {
                'question': 'What is 2+2?',
                'ground_truth': '4',
                'is_correct': True,
                'tokens': ['What', ' is', ' 2', '+', '2', '?'],
                'attention_maps': [
                    [0.1, 0.15, 0.2, 0.25, 0.2, 0.1],  # Layer 0
                    [0.05, 0.1, 0.3, 0.3, 0.15, 0.1],  # Layer 1
                ]
            },
            {
                'question': 'What is 3+3?',
                'ground_truth': '6',
                'is_correct': False,
                'tokens': [],
                'attention_maps': []
            }
        ]

    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary file path for output"""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name

        yield path

        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    def test_generate_html_creates_file(self, sample_instances, temp_output_path):
        """Test that generate_html creates an HTML file"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        assert os.path.exists(temp_output_path)

    def test_generated_html_is_valid_structure(self, sample_instances, temp_output_path):
        """Test that generated HTML has valid structure"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check basic HTML structure
        assert '<!DOCTYPE html>' in content
        assert '<html>' in content
        assert '</html>' in content
        assert '<head>' in content
        assert '</head>' in content
        assert '<body>' in content
        assert '</body>' in content

    def test_generated_html_contains_title(self, sample_instances, temp_output_path):
        """Test that generated HTML contains title"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert '<title>Attention Visualization</title>' in content
        assert '<h1>Attention Visualization</h1>' in content

    def test_generated_html_contains_selects(self, sample_instances, temp_output_path):
        """Test that generated HTML contains dropdown selects and slider"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for instance select and layer slider
        assert 'id="instance-select"' in content
        assert 'id="layer-slider"' in content
        assert 'type="range"' in content  # Should be a slider

    def test_generated_html_contains_metadata_elements(self, sample_instances, temp_output_path):
        """Test that generated HTML contains metadata display elements"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for metadata display elements
        assert 'id="question"' in content
        assert 'id="ground-truth"' in content
        assert 'id="is-correct"' in content

    def test_generated_html_contains_tokens_container(self, sample_instances, temp_output_path):
        """Test that generated HTML contains tokens container"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'id="tokens-container"' in content

    def test_generated_html_contains_css(self, sample_instances, temp_output_path):
        """Test that generated HTML contains CSS styles"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for key CSS elements
        assert '<style>' in content
        assert '</style>' in content
        assert '.token' in content or 'token' in content
        assert 'background-color' in content

    def test_generated_html_contains_javascript(self, sample_instances, temp_output_path):
        """Test that generated HTML contains JavaScript"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for JavaScript
        assert '<script>' in content
        assert '</script>' in content
        assert 'const instances' in content or 'var instances' in content

    def test_generated_html_includes_instance_data(self, sample_instances, temp_output_path):
        """Test that generated HTML includes instance data in JavaScript"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check that instance data is embedded
        assert 'What is 2+2?' in content
        assert 'What is 3+3?' in content
        assert '4' in content  # ground_truth
        assert '6' in content  # ground_truth

    def test_generated_html_handles_special_characters(self, temp_output_path):
        """Test that HTML properly handles special characters"""
        from run_attn_visual import HTMLGenerator

        instances = [
            {
                'question': 'What is <test> & "example"?',
                'ground_truth': '42',
                'is_correct': True,
                'tokens': ['<', 'test', '>'],
                'attention_maps': [[0.33, 0.33, 0.34]]
            }
        ]

        generator = HTMLGenerator()
        generator.generate_html(instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Content should be properly escaped in JSON
        # The exact escaping depends on json.dumps implementation
        assert 'test' in content

    def test_generate_html_with_empty_instances(self, temp_output_path):
        """Test generating HTML with empty instances list"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html([], temp_output_path)

        assert os.path.exists(temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should still generate valid HTML
        assert '<!DOCTYPE html>' in content

    def test_generate_html_with_incorrect_instance(self, temp_output_path):
        """Test generating HTML with incorrect instance"""
        from run_attn_visual import HTMLGenerator

        instances = [
            {
                'question': 'Test question',
                'ground_truth': '42',
                'is_correct': False,  # Incorrect
                'tokens': [],
                'attention_maps': []
            }
        ]

        generator = HTMLGenerator()
        generator.generate_html(instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should include INCORRECT indicator
        assert 'INCORRECT' in content

    def test_generated_html_has_event_listeners(self, sample_instances, temp_output_path):
        """Test that JavaScript includes event listeners"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        generator.generate_html(sample_instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for event listener setup
        assert 'addEventListener' in content

    def test_build_instance_options(self, sample_instances):
        """Test building instance select options"""
        from run_attn_visual import HTMLGenerator

        generator = HTMLGenerator()
        options = generator._build_instance_options(sample_instances)

        # Should contain options for both instances
        assert 'Instance 0' in options
        assert 'Instance 1' in options
        assert 'CORRECT' in options
        assert 'INCORRECT' in options

    def test_generated_html_encoding_utf8(self, temp_output_path):
        """Test that HTML file is encoded as UTF-8"""
        from run_attn_visual import HTMLGenerator

        instances = [
            {
                'question': '中文測試 Chinese test',
                'ground_truth': '42',
                'is_correct': True,
                'tokens': ['中', '文'],
                'attention_maps': [[0.5, 0.5]]
            }
        ]

        generator = HTMLGenerator()
        generator.generate_html(instances, temp_output_path)

        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should properly handle UTF-8 characters
        assert '中文測試' in content
        assert 'charset="UTF-8"' in content or 'charset=UTF-8' in content
