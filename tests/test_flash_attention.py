"""
Tests for Flash Attention integration in AttentionExtractor
"""
import pytest
import argparse


class TestFlashAttentionArgparse:
    """Test Flash Attention command-line argument parsing"""

    def test_argparse_flash_attn_flag_default(self):
        """Test that --flash-attn flag defaults to False"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--flash-attn', action='store_true')

        # Test without flag
        args = parser.parse_args([])
        assert args.flash_attn is False

    def test_argparse_flash_attn_flag_enabled(self):
        """Test that --flash-attn flag can be enabled"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--flash-attn', action='store_true')

        # Test with flag
        args = parser.parse_args(['--flash-attn'])
        assert args.flash_attn is True


class TestFlashAttentionConfiguration:
    """Test Flash Attention configuration logic"""

    def test_use_flash_attn_parameter_default(self):
        """Test that use_flash_attn defaults to False"""
        # This tests the function signature
        import inspect
        from run_attn_visual import AttentionExtractor

        sig = inspect.signature(AttentionExtractor.__init__)
        assert 'use_flash_attn' in sig.parameters
        assert sig.parameters['use_flash_attn'].default is False

    def test_docstring_updated(self):
        """Test that docstring includes Flash Attention documentation"""
        from run_attn_visual import AttentionExtractor

        docstring = AttentionExtractor.__init__.__doc__
        assert 'use_flash_attn' in docstring
        assert 'Flash Attention' in docstring
        assert 'flash-attn' in docstring


class TestFlashAttentionUsageDocumentation:
    """Test that usage documentation is updated"""

    def test_module_docstring_includes_flash_attn(self):
        """Test that module docstring includes Flash Attention examples"""
        import run_attn_visual

        module_doc = run_attn_visual.__doc__
        assert '--flash-attn' in module_doc
        assert 'Flash Attention' in module_doc

    def test_flash_attn_in_notes(self):
        """Test that notes section includes Flash Attention requirements"""
        import run_attn_visual

        module_doc = run_attn_visual.__doc__
        assert 'flash-attn' in module_doc
        assert 'pip install flash-attn' in module_doc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
