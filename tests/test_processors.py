"""
Unit tests for processors.py

This test file covers all Processor classes:
- Processor (base class)
- MaskProcessor
- TruncateProcessor
- ShuffleProcessor
- InsertProcessor
- QuestionProcessor
- RemoveProcessor
- ReplaceProcessor
"""

import pytest


class TestProcessorBase:
    """Tests for Processor base class interface"""

    def test_processor_base_interface(self):
        """Test that Processor base class has required methods"""
        from processors import Processor

        # Verify it's an abstract base class
        assert hasattr(Processor, 'process')
        assert hasattr(Processor, 'get_metadata')

        # Verify we cannot instantiate it directly
        with pytest.raises(TypeError):
            Processor()


class TestMaskProcessor:
    """Tests for MaskProcessor"""

    def test_mask_processor_number_mode(self, sample_context):
        """Test MaskProcessor with mode='number'"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='number')
        reasoning = "First, we calculate 2 + 2 = 4. Then multiply by 3: 4 × 3 = 12."

        result = processor.process(reasoning, sample_context)

        # All numbers should be masked
        assert '2' not in result
        assert '4' not in result
        assert '3' not in result
        assert '12' not in result
        assert '█' in result

    def test_mask_processor_answer_mode(self, sample_context):
        """Test MaskProcessor with mode='answer'"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='answer')
        reasoning = "First, we calculate 2 + 2 = 4. The answer is 12."
        context = {"answer": "12", "question": "What is the answer?"}

        result = processor.process(reasoning, context)

        # Only answer (12) should be masked
        assert '2' in result
        assert '4' in result
        assert '12' not in result
        assert '█' in result

    def test_mask_processor_custom_mask_char(self, sample_context):
        """Test MaskProcessor with custom mask character"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='number', mask_char='*')
        reasoning = "Calculate 2 + 2 = 4"

        result = processor.process(reasoning, sample_context)

        assert '*' in result
        assert '█' not in result

    def test_mask_processor_get_metadata(self, sample_context):
        """Test MaskProcessor.get_metadata()"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='number', mask_char='█')
        reasoning = "Calculate 2 + 2 = 4"

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'mask'
        assert metadata['mode'] == 'number'
        assert metadata['mask_char'] == '█'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_mask_processor_nlines_mode(self, sample_context):
        """Test MaskProcessor with mode='n-lines' and num_prev_lines parameter"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='n-lines', num_prev_lines=2)
        reasoning = """Line 1: compute 5
Line 2: compute 10
Line 3: compute 15
Line 4: The answer is 12."""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # Last 3 lines (answer line + 2 prev) should have numbers masked
        assert '5' in result  # Line 1 not masked
        assert '10' not in result  # Line 2 masked
        assert '15' not in result  # Line 3 masked
        assert '12' not in result  # Line 4 masked

    def test_mask_processor_alphabet_mode(self, sample_context):
        """Test MaskProcessor with mode='alphabet'"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='alphabet')
        reasoning = "Calculate 2 + 2 = 4"

        result = processor.process(reasoning, sample_context)

        # Letters should be masked, numbers preserved
        assert '2' in result
        assert '4' in result
        assert 'Calculate' not in result
        assert '█' in result

    def test_mask_processor_non_number_mode(self, sample_context):
        """Test MaskProcessor with mode='non-number' - mask all non-digit characters"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='non-number')
        reasoning = "Calculate 2 + 2 = 4"

        result = processor.process(reasoning, sample_context)

        # Numbers should be preserved, all non-digits masked
        assert '2' in result
        assert '4' in result
        assert 'Calculate' not in result
        assert 'C' not in result
        assert '+' not in result
        assert '=' not in result
        assert '█' in result

    def test_mask_processor_non_number_mode_with_custom_char(self, sample_context):
        """Test MaskProcessor with mode='non-number' and custom mask character"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='non-number', mask_char=' ')
        reasoning = "Answer is 42"

        result = processor.process(reasoning, sample_context)

        # Numbers preserved, non-digits replaced with space
        assert '4' in result
        assert '2' in result
        assert 'Answer' not in result
        assert 'is' not in result
        # Should have spaces where letters/symbols were
        assert ' ' in result

    def test_mask_processor_non_number_mode_complex(self, sample_context):
        """Test MaskProcessor with mode='non-number' on complex reasoning"""
        from processors import MaskProcessor

        processor = MaskProcessor(mode='non-number')
        reasoning = """Step 1: Calculate 5 + 3 = 8
Step 2: Multiply by 2 = 16
The answer is 42."""

        result = processor.process(reasoning, sample_context)

        # All digits should be preserved
        assert '1' in result
        assert '5' in result
        assert '3' in result
        assert '8' in result
        assert '2' in result
        assert '16' in result
        assert '42' in result

        # All letters should be masked
        assert 'Step' not in result
        assert 'Calculate' not in result
        assert 'Multiply' not in result
        assert 'answer' not in result


class TestTruncateProcessor:
    """Tests for TruncateProcessor"""

    def test_truncate_processor_answer_and_after(self, sample_context):
        """Test TruncateProcessor with mode='answer_and_after'"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer_and_after')
        reasoning = """Step 1: Calculate something
Step 2: Do more work
The answer is 12.
This should be removed."""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        assert "Step 1" in result
        assert "Step 2" in result
        assert "The answer is 12" not in result
        assert "This should be removed" not in result

    def test_truncate_processor_before_answer(self, sample_context):
        """Test TruncateProcessor with mode='before_answer'"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_answer')
        reasoning = """Step 1: Calculate something
Step 2: Do more work
The answer is 12.
This should be kept."""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        assert "Step 1" not in result
        assert "Step 2" not in result
        assert "The answer is 12" in result
        assert "This should be kept" in result

    def test_truncate_processor_last_n_lines(self, sample_context):
        """Test TruncateProcessor with mode='last_n_lines'"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='last_n_lines', n=2)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" not in result
        assert "Line 5" not in result

    def test_truncate_processor_last_ratio(self, sample_context):
        """Test TruncateProcessor with mode='last_ratio'"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='last_ratio', ratio=0.3)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""

        result = processor.process(reasoning, sample_context)

        lines = [l for l in result.split('\n') if l.strip()]
        # 30% of 10 = 3, so should have 7 lines remaining
        assert len(lines) == 7

    def test_truncate_processor_get_metadata(self, sample_context):
        """Test TruncateProcessor.get_metadata()"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='last_n_lines', n=2)
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'truncate'
        assert metadata['mode'] == 'last_n_lines'
        assert metadata['n'] == 2
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert 'removed_lines' in metadata

    def test_truncate_processor_answer_mode_basic(self, sample_context):
        """Test TruncateProcessor with mode='answer' removes exact answer"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = """Step 1: Calculate something
Step 2: The answer is 12.
Step 3: Therefore 12 is correct."""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # Answer "12" should be removed
        assert "12" not in result
        # Other content should be preserved
        assert "Step 1: Calculate something" in result
        assert "Step 2: The answer is" in result
        assert "Step 3: Therefore" in result

    def test_truncate_processor_answer_mode_multiple_occurrences(self, sample_context):
        """Test TruncateProcessor with mode='answer' removes all occurrences"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = """First: 12
Second: 12 again
Third: the value is 12!"""
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # All occurrences of "12" should be removed
        assert "12" not in result
        assert "First:" in result
        assert "Second:" in result
        assert "again" in result
        assert "Third: the value is" in result

    def test_truncate_processor_answer_mode_word_boundary(self, sample_context):
        """Test TruncateProcessor with mode='answer' respects word boundaries"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = """Calculate 42 and 421 and 142.
The number is 42."""
        context = {"answer": "42"}

        result = processor.process(reasoning, context)

        # Only standalone "42" should be removed (word boundary)
        assert "421" in result  # 421 should remain
        assert "142" in result  # 142 should remain
        # But standalone 42 should be removed
        assert "Calculate  and 421" in result or "Calculate  " in result

    def test_truncate_processor_answer_mode_empty_answer(self, sample_context):
        """Test TruncateProcessor with mode='answer' handles empty answer"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = "This is some reasoning text."
        context = {"answer": ""}

        result = processor.process(reasoning, context)

        # Reasoning should be unchanged
        assert result == reasoning

    def test_truncate_processor_answer_mode_no_match(self, sample_context):
        """Test TruncateProcessor with mode='answer' when answer not in text"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = "This text does not contain the answer."
        context = {"answer": "999"}

        result = processor.process(reasoning, context)

        # Reasoning should be unchanged
        assert result == reasoning

    def test_truncate_processor_answer_mode_special_chars(self, sample_context):
        """Test TruncateProcessor with mode='answer' handles special regex chars"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = "The answer is 3.14 which is pi."
        context = {"answer": "3.14"}

        result = processor.process(reasoning, context)

        # Answer with special regex char (.) should be removed safely
        assert "3.14" not in result
        assert "The answer is" in result
        assert "which is pi" in result

    def test_truncate_processor_answer_mode_get_metadata(self, sample_context):
        """Test TruncateProcessor mode='answer' metadata"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='answer')
        reasoning = "The answer is 12. Final: 12."
        context = {"answer": "12"}

        _ = processor.process(reasoning, context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'truncate'
        assert metadata['mode'] == 'answer'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_truncate_processor_after_line_basic(self, sample_context):
        """Test TruncateProcessor with mode='after_line' keeps first N lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', n=3)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" not in result
        assert "Line 5" not in result

    def test_truncate_processor_after_line_default_n(self, sample_context):
        """Test TruncateProcessor with mode='after_line' default n=10"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line')
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # Default n=10, but only 5 lines, so all should remain
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 5

    def test_truncate_processor_after_line_n_larger_than_lines(self, sample_context):
        """Test TruncateProcessor after_line when n > number of lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', n=100)
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Should keep all lines since n > line count
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3

    def test_truncate_processor_after_line_n_zero(self, sample_context):
        """Test TruncateProcessor after_line with n=0 returns empty"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', n=0)
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Should return empty string
        assert result == ""

    def test_truncate_processor_after_line_with_empty_lines(self, sample_context):
        """Test TruncateProcessor after_line ignores empty lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', n=2)
        reasoning = """Line 1

Line 2

Line 3

Line 4"""

        result = processor.process(reasoning, sample_context)

        # Should keep only first 2 non-empty lines
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 2
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" not in result

    def test_truncate_processor_after_line_get_metadata(self, sample_context):
        """Test TruncateProcessor mode='after_line' metadata"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', n=3)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'truncate'
        assert metadata['mode'] == 'after_line'
        assert metadata['n'] == 3
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert 'removed_lines' in metadata
        assert metadata['removed_lines'] == 2  # 5 lines - 3 kept = 2 removed

    def test_truncate_processor_after_line_with_ratio_basic(self, sample_context):
        """Test TruncateProcessor after_line with r (ratio) parameter keeps first X% lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', r=0.4)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # 40% of 5 lines = 2 lines
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 2
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" not in result
        assert "Line 4" not in result
        assert "Line 5" not in result

    def test_truncate_processor_after_line_with_ratio_10_percent(self, sample_context):
        """Test TruncateProcessor after_line with r=0.1 (10%)"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', r=0.1)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""

        result = processor.process(reasoning, sample_context)

        # 10% of 10 lines = 1 line
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 1
        assert "Line 1" in result
        assert "Line 2" not in result

    def test_truncate_processor_after_line_with_ratio_50_percent(self, sample_context):
        """Test TruncateProcessor after_line with r=0.5 (50%)"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', r=0.5)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""

        result = processor.process(reasoning, sample_context)

        # 50% of 10 lines = 5 lines
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 5
        assert "Line 1" in result
        assert "Line 5" in result
        assert "Line 6" not in result

    def test_truncate_processor_after_line_with_ratio_100_percent(self, sample_context):
        """Test TruncateProcessor after_line with r=1.0 (100%)"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', r=1.0)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # 100% of 5 lines = 5 lines (keep all)
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 5

    def test_truncate_processor_after_line_with_ratio_zero(self, sample_context):
        """Test TruncateProcessor after_line with r=0 returns empty"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', r=0)
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Should return empty string
        assert result == ""

    def test_truncate_processor_after_line_with_ratio_small_keeps_at_least_one(self, sample_context):
        """Test TruncateProcessor after_line with very small ratio keeps at least 1 line"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', r=0.01)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # 1% of 5 lines = 0.05, but should keep at least 1 line
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 1
        assert "Line 1" in result

    def test_truncate_processor_after_line_n_takes_priority_over_r(self, sample_context):
        """Test TruncateProcessor after_line when both n and r provided, n takes priority"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', n=3, r=0.5)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""

        result = processor.process(reasoning, sample_context)

        # Should use n=3 (not r=0.5 which would give 5 lines)
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" not in result

    def test_truncate_processor_after_line_with_ratio_get_metadata(self, sample_context):
        """Test TruncateProcessor mode='after_line' with r parameter metadata"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', r=0.4)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'truncate'
        assert metadata['mode'] == 'after_line'
        assert metadata['r'] == 0.4
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert 'removed_lines' in metadata
        assert metadata['removed_lines'] == 3  # 5 lines - 2 kept = 3 removed

    def test_truncate_processor_after_line_with_ratio_ignores_empty_lines(self, sample_context):
        """Test TruncateProcessor after_line with r ignores empty lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='after_line', r=0.5)
        reasoning = """Line 1

Line 2

Line 3

Line 4"""

        result = processor.process(reasoning, sample_context)

        # Should keep 50% of non-empty lines (2 out of 4)
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 2
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" not in result

    def test_truncate_processor_before_line_with_ratio_basic(self, sample_context):
        """Test TruncateProcessor before_line with r (ratio) parameter removes first X% lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', r=0.4)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # Remove 40% of 5 lines = 2 lines, keep remaining 3 lines
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3
        assert "Line 1" not in result
        assert "Line 2" not in result
        assert "Line 3" in result
        assert "Line 4" in result
        assert "Line 5" in result

    def test_truncate_processor_before_line_with_ratio_10_percent(self, sample_context):
        """Test TruncateProcessor before_line with r=0.1 (10%)"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', r=0.1)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""

        result = processor.process(reasoning, sample_context)

        # Remove 10% of 10 lines = 1 line, keep remaining 9 lines
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 9
        result_lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert "Line 1" not in result_lines
        assert "Line 2" in result_lines
        assert "Line 10" in result_lines

    def test_truncate_processor_before_line_with_ratio_50_percent(self, sample_context):
        """Test TruncateProcessor before_line with r=0.5 (50%)"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', r=0.5)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""

        result = processor.process(reasoning, sample_context)

        # Remove 50% of 10 lines = 5 lines, keep remaining 5 lines
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 5
        result_lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert "Line 1" not in result_lines
        assert "Line 5" not in result_lines
        assert "Line 6" in result_lines
        assert "Line 10" in result_lines

    def test_truncate_processor_before_line_with_ratio_100_percent(self, sample_context):
        """Test TruncateProcessor before_line with r=1.0 (100%)"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', r=1.0)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # Remove 100% of 5 lines = all lines removed
        assert result == ""

    def test_truncate_processor_before_line_with_ratio_zero(self, sample_context):
        """Test TruncateProcessor before_line with r=0 keeps all lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', r=0)
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Should keep all lines (no removal)
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_truncate_processor_before_line_with_ratio_small_removes_at_least_one(self, sample_context):
        """Test TruncateProcessor before_line with very small ratio removes at least 1 line"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', r=0.01)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # 1% of 5 lines = 0.05, but should remove at least 1 line
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 4
        assert "Line 1" not in result
        assert "Line 2" in result
        assert "Line 5" in result

    def test_truncate_processor_before_line_basic(self, sample_context):
        """Test TruncateProcessor before_line with n parameter removes first N lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', n=3)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # Remove first 3 lines, keep remaining 2 lines
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 2
        assert "Line 1" not in result
        assert "Line 2" not in result
        assert "Line 3" not in result
        assert "Line 4" in result
        assert "Line 5" in result

    def test_truncate_processor_before_line_n_takes_priority_over_r(self, sample_context):
        """Test TruncateProcessor before_line when both n and r provided, n takes priority"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', n=3, r=0.5)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""

        result = processor.process(reasoning, sample_context)

        # Should use n=3 (not r=0.5 which would remove 5 lines)
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 7
        result_lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert "Line 1" not in result_lines
        assert "Line 2" not in result_lines
        assert "Line 3" not in result_lines
        assert "Line 4" in result_lines
        assert "Line 10" in result_lines

    def test_truncate_processor_before_line_with_ratio_get_metadata(self, sample_context):
        """Test TruncateProcessor mode='before_line' with r parameter metadata"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', r=0.4)
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'truncate'
        assert metadata['mode'] == 'before_line'
        assert metadata['r'] == 0.4
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert 'removed_lines' in metadata
        assert metadata['removed_lines'] == 2  # 2 lines removed (5 - 3 remaining)

    def test_truncate_processor_before_line_with_ratio_ignores_empty_lines(self, sample_context):
        """Test TruncateProcessor before_line with r ignores empty lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', r=0.5)
        reasoning = """Line 1

Line 2

Line 3

Line 4"""

        result = processor.process(reasoning, sample_context)

        # Should remove 50% of non-empty lines (2 out of 4)
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 2
        assert "Line 1" not in result
        assert "Line 2" not in result
        assert "Line 3" in result
        assert "Line 4" in result

    def test_truncate_processor_before_line_default_n(self, sample_context):
        """Test TruncateProcessor before_line with default n=10"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line')
        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        result = processor.process(reasoning, sample_context)

        # Default n=10, but only 5 lines, so all should be removed
        assert result == ""

    def test_truncate_processor_before_line_n_larger_than_lines(self, sample_context):
        """Test TruncateProcessor before_line when n > number of lines"""
        from processors import TruncateProcessor

        processor = TruncateProcessor(mode='before_line', n=100)
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Should remove all lines since n > line count
        assert result == ""


class TestShuffleProcessor:
    """Tests for ShuffleProcessor"""

    def test_shuffle_processor_line_mode(self, sample_context):
        """Test ShuffleProcessor with mode='line'"""
        from processors import ShuffleProcessor

        processor = ShuffleProcessor(mode='line', seed=42)
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        result = processor.process(reasoning, sample_context)

        # All lines should still be present
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" in result

    def test_shuffle_processor_word_mode(self, sample_context):
        """Test ShuffleProcessor with mode='word'"""
        from processors import ShuffleProcessor

        processor = ShuffleProcessor(mode='word', seed=42)
        reasoning = "The quick brown fox jumps"

        result = processor.process(reasoning, sample_context)

        # All words should still be present
        words = set(reasoning.split())
        result_words = set(result.split())
        assert words == result_words

    def test_shuffle_processor_token_mode(self, sample_context):
        """Test ShuffleProcessor with mode='token'"""
        from processors import ShuffleProcessor

        processor = ShuffleProcessor(mode='token', seed=42, tokenizer_model='gpt2')
        reasoning = "The quick brown fox"

        result = processor.process(reasoning, sample_context)

        assert len(result) > 0

    def test_shuffle_processor_seed_reproducible(self, sample_context):
        """Test that same seed produces same shuffle"""
        from processors import ShuffleProcessor

        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        processor1 = ShuffleProcessor(mode='line', seed=42)
        processor2 = ShuffleProcessor(mode='line', seed=42)

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        assert result1 == result2

    def test_shuffle_processor_get_metadata(self, sample_context):
        """Test ShuffleProcessor.get_metadata()"""
        from processors import ShuffleProcessor

        processor = ShuffleProcessor(mode='line', seed=42)
        reasoning = """Line 1
Line 2
Line 3"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'shuffle'
        assert metadata['mode'] == 'line'
        assert metadata['seed'] == 42
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_shuffle_processor_token_mode_with_model_type(self, sample_context):
        """Test ShuffleProcessor with mode='token' using model_type parameter"""
        from processors import ShuffleProcessor

        # Test with different model types
        for model_type in ['deepseek', 'gpt-oss', 'qwen3']:
            processor = ShuffleProcessor(mode='token', seed=42, model_type=model_type)
            reasoning = "The quick brown fox jumps over the lazy dog"

            result = processor.process(reasoning, sample_context)

            assert len(result) > 0, f"Failed for model_type={model_type}"

            # Verify metadata includes model_type
            metadata = processor.get_metadata()
            assert metadata['model_type'] == model_type

    def test_shuffle_processor_token_mode_model_type_reproducible(self, sample_context):
        """Test that ShuffleProcessor with model_type produces reproducible results"""
        from processors import ShuffleProcessor

        reasoning = "The quick brown fox jumps over the lazy dog"

        processor1 = ShuffleProcessor(mode='token', seed=42, model_type='gpt-oss')
        processor2 = ShuffleProcessor(mode='token', seed=42, model_type='gpt-oss')

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        assert result1 == result2


class TestInsertProcessor:
    """Tests for InsertProcessor"""

    def test_insert_processor_fix_mode_random_position(self, sample_context):
        """Test InsertProcessor with mode='fix' and position='random'"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Maybe the answer is 123.',
            position='random',
            count=3,
            seed=42
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        result = processor.process(reasoning, sample_context)

        # Inserted sentence should appear
        assert 'Maybe the answer is 123.' in result

        # All original lines should still be present
        assert 'Line 1' in result
        assert 'Line 2' in result
        assert 'Line 3' in result
        assert 'Line 4' in result

        # Should have more lines after insertion
        original_lines = len([l for l in reasoning.split('\n') if l.strip()])
        result_lines = len([l for l in result.split('\n') if l.strip()])
        assert result_lines == original_lines + 3

    def test_insert_processor_seed_reproducible(self, sample_context):
        """Test that same seed produces same insertion positions"""
        from processors import InsertProcessor

        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        processor1 = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=3,
            seed=42
        )
        processor2 = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=3,
            seed=42
        )

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        # Same seed should produce identical results
        assert result1 == result2

    def test_insert_processor_different_seeds(self, sample_context):
        """Test that different seeds produce different insertion positions"""
        from processors import InsertProcessor

        reasoning = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        processor1 = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=3,
            seed=42
        )
        processor2 = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=3,
            seed=99
        )

        result1 = processor1.process(reasoning, sample_context)
        result2 = processor2.process(reasoning, sample_context)

        # Different seeds should likely produce different results
        # (not guaranteed, but highly probable)
        assert result1 != result2

    def test_insert_processor_default_parameters(self, sample_context):
        """Test InsertProcessor with default parameters"""
        from processors import InsertProcessor

        processor = InsertProcessor(mode='fix')
        reasoning = """Line 1
Line 2"""

        result = processor.process(reasoning, sample_context)

        # Default sentence should appear
        assert 'Maybe the answer is 123.' in result

        # Original lines should be present
        assert 'Line 1' in result
        assert 'Line 2' in result

    def test_insert_processor_get_metadata(self, sample_context):
        """Test InsertProcessor.get_metadata()"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Test insertion.',
            position='random',
            count=2,
            seed=42
        )
        reasoning = """Line 1
Line 2
Line 3"""

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'insert'
        assert metadata['mode'] == 'fix'
        assert metadata['sentence'] == 'Test insertion.'
        assert metadata['position'] == 'random'
        assert metadata['count'] == 2
        assert metadata['seed'] == 42
        assert 'insertion_positions' in metadata
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

        # Check that insertion_positions is a list with correct length
        assert isinstance(metadata['insertion_positions'], list)
        assert len(metadata['insertion_positions']) == 2

    def test_insert_processor_invalid_mode(self, sample_context):
        """Test InsertProcessor with invalid mode raises error"""
        from processors import InsertProcessor

        processor = InsertProcessor(mode='invalid_mode')
        reasoning = "Line 1"

        with pytest.raises(ValueError, match="Invalid insert mode"):
            processor.process(reasoning, sample_context)

    def test_insert_processor_invalid_position(self, sample_context):
        """Test InsertProcessor with invalid position raises error"""
        from processors import InsertProcessor

        processor = InsertProcessor(mode='fix', position='invalid_position')
        reasoning = "Line 1"

        with pytest.raises(ValueError, match="Invalid position strategy"):
            processor.process(reasoning, sample_context)

    def test_insert_processor_count_zero(self, sample_context):
        """Test InsertProcessor with count=0 (no insertions)"""
        from processors import InsertProcessor

        processor = InsertProcessor(mode='fix', count=0)
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # No insertions should occur
        assert result == reasoning

    def test_insert_processor_count_multiple(self, sample_context):
        """Test InsertProcessor with large count"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=10,
            seed=42
        )
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Should have 10 insertions
        original_lines = len([l for l in reasoning.split('\n') if l.strip()])
        result_lines = len([l for l in result.split('\n') if l.strip()])
        assert result_lines == original_lines + 10

        # Count occurrences of inserted sentence
        assert result.count('Noise') == 10

    def test_insert_processor_custom_sentence(self, sample_context):
        """Test InsertProcessor with custom sentence"""
        from processors import InsertProcessor

        custom_sentence = "This is a custom noise sentence with numbers 999."
        processor = InsertProcessor(
            mode='fix',
            sentence=custom_sentence,
            position='random',
            count=1,
            seed=42
        )
        reasoning = "Line 1"

        result = processor.process(reasoning, sample_context)

        # Custom sentence should appear
        assert custom_sentence in result

    def test_insert_processor_empty_reasoning(self, sample_context):
        """Test InsertProcessor with empty reasoning"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=1,
            seed=42
        )
        reasoning = ""

        result = processor.process(reasoning, sample_context)

        # Should insert into empty text
        assert 'Noise' in result

    def test_insert_processor_single_line(self, sample_context):
        """Test InsertProcessor with single-line reasoning"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count=2,
            seed=42
        )
        reasoning = "Single line"

        result = processor.process(reasoning, sample_context)

        # Original line should be present
        assert 'Single line' in result

        # Should have insertions
        assert result.count('Noise') == 2

    def test_insert_processor_percentage_count_100_percent(self, sample_context):
        """Test InsertProcessor with count='100% # of answer'"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count='100% # of answer',
            seed=42
        )
        reasoning = "The answer is 12. We verify 12 is correct. Final: 12."
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # Answer appears 3 times, so should insert 3 times (100% of 3 = 3)
        assert result.count('Noise') == 3

        # Verify metadata
        metadata = processor.get_metadata()
        assert metadata['count'] == '100% # of answer'
        assert metadata['actual_count'] == 3

    def test_insert_processor_percentage_count_200_percent(self, sample_context):
        """Test InsertProcessor with count='200% # of answer'"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Test',
            position='random',
            count='200% # of answer',
            seed=42
        )
        reasoning = "Answer: 42. Check: 42."
        context = {"answer": "42"}

        result = processor.process(reasoning, context)

        # Answer appears 2 times, so should insert 4 times (200% of 2 = 4)
        assert result.count('Test') == 4

        # Verify metadata
        metadata = processor.get_metadata()
        assert metadata['count'] == '200% # of answer'
        assert metadata['actual_count'] == 4

    def test_insert_processor_percentage_count_50_percent(self, sample_context):
        """Test InsertProcessor with count='50% # of answer' (rounding)"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count='50% # of answer',
            seed=42
        )
        reasoning = "Answer: 5. Verify: 5. Final: 5."
        context = {"answer": "5"}

        result = processor.process(reasoning, context)

        # Answer appears 3 times, so should insert 2 times (50% of 3 = 1.5, rounds to 2)
        assert result.count('Noise') == 2

        # Verify metadata
        metadata = processor.get_metadata()
        assert metadata['count'] == '50% # of answer'
        assert metadata['actual_count'] == 2

    def test_insert_processor_percentage_count_zero_occurrences(self, sample_context):
        """Test InsertProcessor with percentage count when answer doesn't appear"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count='100% # of answer',
            seed=42
        )
        reasoning = "Some text without the answer."
        context = {"answer": "999"}

        result = processor.process(reasoning, context)

        # Answer appears 0 times, so should insert 0 times
        assert 'Noise' not in result

        # Verify metadata
        metadata = processor.get_metadata()
        assert metadata['count'] == '100% # of answer'
        assert metadata['actual_count'] == 0

    def test_insert_processor_percentage_count_word_boundary(self, sample_context):
        """Test InsertProcessor percentage count respects word boundaries"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count='100% # of answer',
            seed=42
        )
        reasoning = "Calculate 42 and 421 and 142. Answer: 42."
        context = {"answer": "42"}

        result = processor.process(reasoning, context)

        # Only standalone 42 should be counted (appears 2 times), not 421 or 142
        assert result.count('Noise') == 2

    def test_insert_processor_percentage_count_missing_context(self, sample_context):
        """Test InsertProcessor percentage count raises error when answer missing"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            position='random',
            count='100% # of answer'
        )
        reasoning = "Some reasoning"

        # Context without answer
        context = {"question": "What is 2+2?"}

        with pytest.raises(ValueError, match="requires 'answer' or 'ground_truth' in context"):
            processor.process(reasoning, context)

    def test_insert_processor_percentage_count_invalid_format(self, sample_context):
        """Test InsertProcessor with invalid percentage format"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='fix',
            sentence='Noise',
            count='invalid format'
        )
        reasoning = "Test"

        with pytest.raises(ValueError, match="Invalid count format"):
            processor.process(reasoning, sample_context)

    def test_insert_processor_percentage_count_in_pipeline(self, sample_context):
        """Test InsertProcessor percentage count works in pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "insert('fix',sentence='Noise',count='100% # of answer',seed=42)"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "The answer is 12. Therefore 12."
        context = {"answer": "12"}
        result, metadata_list = pipeline.execute(reasoning, context)

        # Answer appears 2 times, should insert 2 times
        assert result.count('Noise') == 2
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'insert'
        assert metadata_list[0]['count'] == '100% # of answer'
        assert metadata_list[0]['actual_count'] == 2

    def test_insert_processor_percentage_count_case_insensitive(self, sample_context):
        """Test InsertProcessor percentage format is case insensitive"""
        from processors import InsertProcessor

        # Test uppercase
        processor1 = InsertProcessor(
            mode='fix',
            sentence='NOISE1',
            count='100% # OF ANSWER',
            seed=42
        )

        # Test mixed case
        processor2 = InsertProcessor(
            mode='fix',
            sentence='NOISE2',
            count='100% # Of AnSwEr',
            seed=42
        )

        reasoning = "Result: 5. Check: 5."
        context = {"answer": "5"}

        result1 = processor1.process(reasoning, context)
        result2 = processor2.process(reasoning, context)

        # Both should insert 2 times (answer appears 2 times)
        assert result1.count('NOISE1') == 2
        assert result2.count('NOISE2') == 2

    def test_insert_processor_head_mode_basic(self, sample_context):
        """Test InsertProcessor with mode='head' inserts at the beginning"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='This is the first line.'
        )
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Inserted sentence should be the first line
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'This is the first line.'
        assert lines[1] == 'Line 1'
        assert lines[2] == 'Line 2'
        assert lines[3] == 'Line 3'

        # Total line count should be original + 1
        original_lines = len([l for l in reasoning.split('\n') if l.strip()])
        result_lines = len([l for l in result.split('\n') if l.strip()])
        assert result_lines == original_lines + 1

    def test_insert_processor_head_mode_with_answer_placeholder(self, sample_context):
        """Test InsertProcessor head mode with {ANSWER} placeholder"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='The answer is {ANSWER}.'
        )
        reasoning = """Line 1
Line 2
Line 3"""

        context = {'answer': '42'}
        result = processor.process(reasoning, context)

        # {ANSWER} should be replaced with actual answer
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'The answer is 42.'
        assert lines[1] == 'Line 1'

    def test_insert_processor_head_mode_answer_placeholder_case_insensitive(self, sample_context):
        """Test InsertProcessor head mode {ANSWER} is case insensitive"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='Result: {answer}, Check: {ANSWER}'
        )
        reasoning = "Original line"

        context = {'answer': '99'}
        result = processor.process(reasoning, context)

        # Both {answer} and {ANSWER} should be replaced
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Result: 99, Check: 99'

    def test_insert_processor_head_mode_without_placeholder(self, sample_context):
        """Test InsertProcessor head mode without {ANSWER} placeholder"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='Static text without placeholder.'
        )
        reasoning = """Line 1
Line 2"""

        context = {'answer': '42'}
        result = processor.process(reasoning, context)

        # Should insert static text without modification
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Static text without placeholder.'

    def test_insert_processor_head_mode_ground_truth_fallback(self, sample_context):
        """Test InsertProcessor head mode uses ground_truth if answer not available"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='Answer: {ANSWER}'
        )
        reasoning = "Line 1"

        # Use ground_truth instead of answer
        context = {'ground_truth': '123'}
        result = processor.process(reasoning, context)

        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Answer: 123'

    def test_insert_processor_head_mode_get_metadata(self, sample_context):
        """Test InsertProcessor head mode metadata"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='Header: {ANSWER}'
        )
        reasoning = """Line 1
Line 2"""

        context = {'answer': '42'}
        _ = processor.process(reasoning, context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'insert'
        assert metadata['mode'] == 'head'
        assert metadata['sentence'] == 'Header: {ANSWER}'
        assert metadata['actual_count'] == 1
        assert metadata['insertion_positions'] == [0]
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_insert_processor_head_mode_empty_reasoning(self, sample_context):
        """Test InsertProcessor head mode with empty reasoning"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='First line: {ANSWER}'
        )
        reasoning = ""

        context = {'answer': '99'}
        result = processor.process(reasoning, context)

        # Should just be the inserted sentence
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert len(lines) == 1
        assert lines[0] == 'First line: 99'

    def test_insert_processor_head_mode_multiple_answer_placeholders(self, sample_context):
        """Test InsertProcessor head mode with multiple {ANSWER} placeholders"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='Answer: {ANSWER}, Double: {ANSWER}'
        )
        reasoning = "Original"

        context = {'answer': '10'}
        result = processor.process(reasoning, context)

        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Answer: 10, Double: 10'

    def test_insert_processor_head_mode_answer_with_backslash(self, sample_context):
        """Test InsertProcessor head mode with answer containing backslash"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='The answer is {ANSWER}.'
        )
        reasoning = "Original line"

        # Test with answer containing backslash (common in LaTeX)
        context = {'answer': r'\frac{1}{2}'}
        result = processor.process(reasoning, context)

        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == r'The answer is \frac{1}{2}.'

    def test_insert_processor_head_mode_answer_with_special_regex_chars(self, sample_context):
        """Test InsertProcessor head mode with answer containing special regex characters"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='head',
            sentence='Answer: {ANSWER}'
        )
        reasoning = "Line 1"

        # Test with various special regex characters
        special_answers = [
            r'\d+',  # Backslash and plus
            '$100',  # Dollar sign
            '(a+b)',  # Parentheses and plus
            'x^2',  # Caret
            'a|b',  # Pipe
        ]

        for special_answer in special_answers:
            context = {'answer': special_answer}
            result = processor.process(reasoning, context)
            lines = [l.strip() for l in result.split('\n') if l.strip()]
            assert lines[0] == f'Answer: {special_answer}', f"Failed for answer: {special_answer}"

    def test_insert_processor_tail_mode_basic(self, sample_context):
        """Test InsertProcessor with mode='tail' inserts at the end"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='This is the last line.'
        )
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # Inserted sentence should be the last line
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Line 1'
        assert lines[1] == 'Line 2'
        assert lines[2] == 'Line 3'
        assert lines[3] == 'This is the last line.'

        # Total line count should be original + 1
        original_lines = len([l for l in reasoning.split('\n') if l.strip()])
        result_lines = len([l for l in result.split('\n') if l.strip()])
        assert result_lines == original_lines + 1

    def test_insert_processor_tail_mode_with_answer_placeholder(self, sample_context):
        """Test InsertProcessor tail mode with {ANSWER} placeholder"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='The answer is {ANSWER}.'
        )
        reasoning = """Line 1
Line 2
Line 3"""

        context = {'answer': '42'}
        result = processor.process(reasoning, context)

        # {ANSWER} should be replaced with actual answer
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Line 1'
        assert lines[-1] == 'The answer is 42.'

    def test_insert_processor_tail_mode_answer_placeholder_case_insensitive(self, sample_context):
        """Test InsertProcessor tail mode {ANSWER} is case insensitive"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='Result: {answer}, Check: {ANSWER}'
        )
        reasoning = "Original line"

        context = {'answer': '99'}
        result = processor.process(reasoning, context)

        # Both {answer} and {ANSWER} should be replaced
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[-1] == 'Result: 99, Check: 99'

    def test_insert_processor_tail_mode_without_placeholder(self, sample_context):
        """Test InsertProcessor tail mode without {ANSWER} placeholder"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='Static text without placeholder.'
        )
        reasoning = """Line 1
Line 2"""

        context = {'answer': '42'}
        result = processor.process(reasoning, context)

        # Should insert static text without modification
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[-1] == 'Static text without placeholder.'

    def test_insert_processor_tail_mode_ground_truth_fallback(self, sample_context):
        """Test InsertProcessor tail mode uses ground_truth if answer not available"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='Answer: {ANSWER}'
        )
        reasoning = "Line 1"

        # Use ground_truth instead of answer
        context = {'ground_truth': '123'}
        result = processor.process(reasoning, context)

        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[-1] == 'Answer: 123'

    def test_insert_processor_tail_mode_get_metadata(self, sample_context):
        """Test InsertProcessor tail mode metadata"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='Footer: {ANSWER}'
        )
        reasoning = """Line 1
Line 2"""

        context = {'answer': '42'}
        _ = processor.process(reasoning, context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'insert'
        assert metadata['mode'] == 'tail'
        assert metadata['sentence'] == 'Footer: {ANSWER}'
        assert metadata['actual_count'] == 1
        assert len(metadata['insertion_positions']) == 1
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_insert_processor_tail_mode_empty_reasoning(self, sample_context):
        """Test InsertProcessor tail mode with empty reasoning"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='Last line: {ANSWER}'
        )
        reasoning = ""

        context = {'answer': '99'}
        result = processor.process(reasoning, context)

        # Should just be the inserted sentence
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert len(lines) == 1
        assert lines[0] == 'Last line: 99'

    def test_insert_processor_tail_mode_multiple_answer_placeholders(self, sample_context):
        """Test InsertProcessor tail mode with multiple {ANSWER} placeholders"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='Answer: {ANSWER}, Double: {ANSWER}'
        )
        reasoning = "Original"

        context = {'answer': '10'}
        result = processor.process(reasoning, context)

        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[-1] == 'Answer: 10, Double: 10'

    def test_insert_processor_tail_mode_answer_with_backslash(self, sample_context):
        """Test InsertProcessor tail mode with answer containing backslash"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='The answer is {ANSWER}.'
        )
        reasoning = "Original line"

        # Test with answer containing backslash (common in LaTeX)
        context = {'answer': r'\frac{1}{2}'}
        result = processor.process(reasoning, context)

        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[-1] == r'The answer is \frac{1}{2}.'

    def test_insert_processor_tail_mode_answer_with_special_regex_chars(self, sample_context):
        """Test InsertProcessor tail mode with answer containing special regex characters"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='tail',
            sentence='Answer: {ANSWER}'
        )
        reasoning = "Line 1"

        # Test with various special regex characters
        special_answers = [
            r'\d+',  # Backslash and plus
            '$100',  # Dollar sign
            '(a+b)',  # Parentheses and plus
            'x^2',  # Caret
            'a|b',  # Pipe
        ]

        for special_answer in special_answers:
            context = {'answer': special_answer}
            result = processor.process(reasoning, context)
            lines = [l.strip() for l in result.split('\n') if l.strip()]
            assert lines[-1] == f'Answer: {special_answer}', f"Failed for answer: {special_answer}"

    def test_insert_processor_middle_mode_basic(self, sample_context):
        """Test InsertProcessor with mode='middle' inserts at the middle"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='This is the middle line.'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        result = processor.process(reasoning, sample_context)

        # Inserted sentence should be at the middle (position 2 for 4 lines: 4//2=2)
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Line 1'
        assert lines[1] == 'Line 2'
        assert lines[2] == 'This is the middle line.'
        assert lines[3] == 'Line 3'
        assert lines[4] == 'Line 4'

        # Total line count should be original + 1
        original_lines = len([l for l in reasoning.split('\n') if l.strip()])
        result_lines = len([l for l in result.split('\n') if l.strip()])
        assert result_lines == original_lines + 1

    def test_insert_processor_middle_mode_odd_lines(self, sample_context):
        """Test InsertProcessor middle mode with odd number of lines"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Middle line.'
        )
        reasoning = """Line 1
Line 2
Line 3"""

        result = processor.process(reasoning, sample_context)

        # For 3 lines, middle position is 3//2=1
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Line 1'
        assert lines[1] == 'Middle line.'
        assert lines[2] == 'Line 2'
        assert lines[3] == 'Line 3'

    def test_insert_processor_middle_mode_with_answer_placeholder(self, sample_context):
        """Test InsertProcessor middle mode with {ANSWER} placeholder"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='The answer is {ANSWER}.'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        context = {'answer': '42'}
        result = processor.process(reasoning, context)

        # {ANSWER} should be replaced with actual answer
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert 'The answer is 42.' in lines

    def test_insert_processor_middle_mode_answer_placeholder_case_insensitive(self, sample_context):
        """Test InsertProcessor middle mode {ANSWER} is case insensitive"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Result: {answer}, Check: {ANSWER}'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        context = {'answer': '99'}
        result = processor.process(reasoning, context)

        # Both {answer} and {ANSWER} should be replaced
        assert 'Result: 99, Check: 99' in result

    def test_insert_processor_middle_mode_without_placeholder(self, sample_context):
        """Test InsertProcessor middle mode without {ANSWER} placeholder"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Static text without placeholder.'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        context = {'answer': '42'}
        result = processor.process(reasoning, context)

        # Should insert static text without modification
        assert 'Static text without placeholder.' in result

    def test_insert_processor_middle_mode_ground_truth_fallback(self, sample_context):
        """Test InsertProcessor middle mode uses ground_truth if answer not available"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Answer: {ANSWER}'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        # Use ground_truth instead of answer
        context = {'ground_truth': '123'}
        result = processor.process(reasoning, context)

        assert 'Answer: 123' in result

    def test_insert_processor_middle_mode_get_metadata(self, sample_context):
        """Test InsertProcessor middle mode metadata"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Middle: {ANSWER}'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        context = {'answer': '42'}
        _ = processor.process(reasoning, context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'insert'
        assert metadata['mode'] == 'middle'
        assert metadata['sentence'] == 'Middle: {ANSWER}'
        assert metadata['actual_count'] == 1
        assert len(metadata['insertion_positions']) == 1
        assert metadata['insertion_positions'][0] == 2  # 4 lines // 2 = 2
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_insert_processor_middle_mode_empty_reasoning(self, sample_context):
        """Test InsertProcessor middle mode with empty reasoning"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Middle line: {ANSWER}'
        )
        reasoning = ""

        context = {'answer': '99'}
        result = processor.process(reasoning, context)

        # Should just be the inserted sentence (middle of empty is position 0)
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert len(lines) == 1
        assert lines[0] == 'Middle line: 99'

    def test_insert_processor_middle_mode_single_line(self, sample_context):
        """Test InsertProcessor middle mode with single line"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Inserted.'
        )
        reasoning = "Single line"

        result = processor.process(reasoning, sample_context)

        # For 1 line, middle position is 1//2=0 (insert at beginning)
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Inserted.'
        assert lines[1] == 'Single line'

    def test_insert_processor_middle_mode_two_lines(self, sample_context):
        """Test InsertProcessor middle mode with two lines"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Middle.'
        )
        reasoning = """Line 1
Line 2"""

        result = processor.process(reasoning, sample_context)

        # For 2 lines, middle position is 2//2=1
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert lines[0] == 'Line 1'
        assert lines[1] == 'Middle.'
        assert lines[2] == 'Line 2'

    def test_insert_processor_middle_mode_multiple_answer_placeholders(self, sample_context):
        """Test InsertProcessor middle mode with multiple {ANSWER} placeholders"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Answer: {ANSWER}, Double: {ANSWER}'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        context = {'answer': '10'}
        result = processor.process(reasoning, context)

        assert 'Answer: 10, Double: 10' in result

    def test_insert_processor_middle_mode_answer_with_backslash(self, sample_context):
        """Test InsertProcessor middle mode with answer containing backslash"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='The answer is {ANSWER}.'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        # Test with answer containing backslash (common in LaTeX)
        context = {'answer': r'\frac{1}{2}'}
        result = processor.process(reasoning, context)

        assert r'The answer is \frac{1}{2}.' in result

    def test_insert_processor_middle_mode_answer_with_special_regex_chars(self, sample_context):
        """Test InsertProcessor middle mode with answer containing special regex characters"""
        from processors import InsertProcessor

        processor = InsertProcessor(
            mode='middle',
            sentence='Answer: {ANSWER}'
        )
        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        # Test with various special regex characters
        special_answers = [
            r'\d+',  # Backslash and plus
            '$100',  # Dollar sign
            '(a+b)',  # Parentheses and plus
            'x^2',  # Caret
            'a|b',  # Pipe
        ]

        for special_answer in special_answers:
            context = {'answer': special_answer}
            result = processor.process(reasoning, context)
            assert f'Answer: {special_answer}' in result, f"Failed for answer: {special_answer}"


class TestQuestionProcessor:
    """Tests for QuestionProcessor"""

    def test_question_remove_mode(self, sample_context):
        """Test QuestionProcessor with mode='remove'"""
        from processors import QuestionProcessor

        processor = QuestionProcessor(mode='remove')
        reasoning = "Step 1: Calculate something\nStep 2: Get answer"

        # Context should be modified in place
        context = sample_context.copy()
        original_question = context['question']

        result = processor.process(reasoning, context)

        # Reasoning should be unchanged
        assert result == reasoning

        # Question should be empty
        assert context['question'] == ''

        # Original question should be stored in metadata
        metadata = processor.get_metadata()
        assert metadata['processor'] == 'question'
        assert metadata['mode'] == 'remove'
        assert metadata['original_question'] == original_question

    def test_question_invalid_mode(self, sample_context):
        """Test QuestionProcessor with invalid mode"""
        from processors import QuestionProcessor

        processor = QuestionProcessor(mode='invalid')
        reasoning = "Test reasoning"

        with pytest.raises(ValueError, match="Invalid question mode"):
            processor.process(reasoning, sample_context)


class TestRemoveProcessor:
    """Tests for RemoveProcessor"""

    def test_remove_blank_mode_basic(self, sample_context):
        """Test RemoveProcessor with mode='blank' on basic text"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "hello\n\n   world   \t\n\n   test"

        result = processor.process(reasoning, sample_context)

        # All continuous blanks should be consolidated to single space
        assert result == "hello world test"

    def test_remove_blank_mode_reasoning_text(self, sample_context):
        """Test RemoveProcessor with mode='blank' on reasoning-like text"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = """Let me solve this problem step by step.


First, I will analyze the question.


   Then,   I calculate:

x = 5


Therefore, the answer is 5."""

        result = processor.process(reasoning, sample_context)

        # All continuous blanks should be consolidated
        expected = "Let me solve this problem step by step. First, I will analyze the question. Then, I calculate: x = 5 Therefore, the answer is 5."
        assert result == expected

    def test_remove_blank_mode_multiple_spaces(self, sample_context):
        """Test RemoveProcessor removes multiple spaces"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "word1    word2     word3      word4"

        result = processor.process(reasoning, sample_context)

        assert result == "word1 word2 word3 word4"

    def test_remove_blank_mode_multiple_newlines(self, sample_context):
        """Test RemoveProcessor removes multiple newlines"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "Line 1\n\n\n\nLine 2\n\n\nLine 3"

        result = processor.process(reasoning, sample_context)

        assert result == "Line 1 Line 2 Line 3"

    def test_remove_blank_mode_tabs(self, sample_context):
        """Test RemoveProcessor removes tabs"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "word1\t\t\tword2\t\tword3"

        result = processor.process(reasoning, sample_context)

        assert result == "word1 word2 word3"

    def test_remove_blank_mode_mixed_whitespace(self, sample_context):
        """Test RemoveProcessor with mixed whitespace characters"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "word1 \t\n\n  \t  word2   \n\t   word3"

        result = processor.process(reasoning, sample_context)

        assert result == "word1 word2 word3"

    def test_remove_blank_mode_leading_trailing_whitespace(self, sample_context):
        """Test RemoveProcessor strips leading/trailing whitespace"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "   \n\n  hello world  \n\n   "

        result = processor.process(reasoning, sample_context)

        assert result == "hello world"

    def test_remove_blank_mode_empty_string(self, sample_context):
        """Test RemoveProcessor with empty string"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = ""

        result = processor.process(reasoning, sample_context)

        assert result == ""

    def test_remove_blank_mode_only_whitespace(self, sample_context):
        """Test RemoveProcessor with only whitespace"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "   \n\n\t\t   \n   "

        result = processor.process(reasoning, sample_context)

        assert result == ""

    def test_remove_blank_mode_single_word(self, sample_context):
        """Test RemoveProcessor with single word"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "hello"

        result = processor.process(reasoning, sample_context)

        assert result == "hello"

    def test_remove_blank_mode_preserves_content(self, sample_context):
        """Test RemoveProcessor preserves non-whitespace content"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "Calculate  \n\n  2 + 2  =   \n  4   and   multiply   \n\n by  3:   \n 4 × 3 = 12"

        result = processor.process(reasoning, sample_context)

        # All non-whitespace characters should be preserved
        assert "Calculate" in result
        assert "2 + 2 = 4" in result
        assert "multiply" in result
        assert "3:" in result
        assert "4 × 3 = 12" in result

    def test_remove_processor_get_metadata(self, sample_context):
        """Test RemoveProcessor.get_metadata()"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='blank')
        reasoning = "hello\n\n\nworld   test"

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'remove'
        assert metadata['mode'] == 'blank'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

        # Verify stats are computed correctly
        assert metadata['input_stats']['lines'] > 0
        assert metadata['output_stats']['lines'] > 0

    def test_remove_processor_invalid_mode(self, sample_context):
        """Test RemoveProcessor with invalid mode raises error"""
        from processors import RemoveProcessor

        processor = RemoveProcessor(mode='invalid_mode')
        reasoning = "Test reasoning"

        with pytest.raises(ValueError, match="Invalid remove mode"):
            processor.process(reasoning, sample_context)


class TestReplaceProcessor:
    """Tests for ReplaceProcessor"""

    def test_replace_processor_whitespace_basic(self, sample_context):
        """Test ReplaceProcessor replacing all whitespace with single space"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\s', replacement=' ')
        reasoning = "hello\nworld\ttest"

        result = processor.process(reasoning, sample_context)

        # All whitespace should be replaced with single space
        assert result == "hello world test"

    def test_replace_processor_digits(self, sample_context):
        """Test ReplaceProcessor replacing all digits"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d', replacement='X')
        reasoning = "Calculate 2 + 2 = 4"

        result = processor.process(reasoning, sample_context)

        # All digits should be replaced with 'X'
        assert result == "Calculate X + X = X"

    def test_replace_processor_word(self, sample_context):
        """Test ReplaceProcessor replacing specific word"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\banswer\b', replacement='result')
        reasoning = "The answer is 42. Calculate the answer."

        result = processor.process(reasoning, sample_context)

        # Word 'answer' should be replaced with 'result'
        assert result == "The result is 42. Calculate the result."

    def test_replace_processor_remove_pattern(self, sample_context):
        """Test ReplaceProcessor removing pattern (empty replacement)"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d+', replacement='')
        reasoning = "Step 1: Calculate 42 in step 2."

        result = processor.process(reasoning, sample_context)

        # All digits should be removed
        assert result == "Step : Calculate  in step ."

    def test_replace_processor_multiple_replacements(self, sample_context):
        """Test ReplaceProcessor counts multiple replacements"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d', replacement='#')
        reasoning = "1 2 3 4 5"

        result = processor.process(reasoning, sample_context)

        # Should replace 5 digits
        assert result == "# # # # #"

        metadata = processor.get_metadata()
        assert metadata['num_replacements'] == 5

    def test_replace_processor_no_match(self, sample_context):
        """Test ReplaceProcessor when pattern doesn't match"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d', replacement='X')
        reasoning = "No numbers here"

        result = processor.process(reasoning, sample_context)

        # Should be unchanged
        assert result == "No numbers here"

        metadata = processor.get_metadata()
        assert metadata['num_replacements'] == 0

    def test_replace_processor_complex_pattern(self, sample_context):
        """Test ReplaceProcessor with complex regex pattern"""
        from processors import ReplaceProcessor

        # Replace mathematical operators
        processor = ReplaceProcessor(pattern=r'[+\-*/=]', replacement='OP')
        reasoning = "2 + 3 - 1 * 4 / 2 = 8"

        result = processor.process(reasoning, sample_context)

        assert result == "2 OP 3 OP 1 OP 4 OP 2 OP 8"

    def test_replace_processor_newline_to_space(self, sample_context):
        """Test ReplaceProcessor replacing newlines with spaces"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\n', replacement=' ')
        reasoning = "Line 1\nLine 2\nLine 3"

        result = processor.process(reasoning, sample_context)

        assert result == "Line 1 Line 2 Line 3"

    def test_replace_processor_multiline_reasoning(self, sample_context):
        """Test ReplaceProcessor on multiline reasoning text"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\s+', replacement=' ')
        reasoning = """Step 1: Calculate something


Step 2: Do more work

Final answer: 42"""

        result = processor.process(reasoning, sample_context)

        # Multiple spaces/newlines should be replaced with single space
        assert "Step 1: Calculate something Step 2: Do more work Final answer: 42" == result

    def test_replace_processor_get_metadata(self, sample_context):
        """Test ReplaceProcessor.get_metadata()"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\d', replacement='X')
        reasoning = "Numbers: 1 2 3"

        _ = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'replace'
        assert metadata['pattern'] == r'\d'
        assert metadata['replacement'] == 'X'
        assert metadata['num_replacements'] == 3
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_replace_processor_empty_reasoning(self, sample_context):
        """Test ReplaceProcessor with empty reasoning"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\s', replacement=' ')
        reasoning = ""

        result = processor.process(reasoning, sample_context)

        assert result == ""

        metadata = processor.get_metadata()
        assert metadata['num_replacements'] == 0

    def test_replace_processor_default_replacement(self, sample_context):
        """Test ReplaceProcessor with default empty replacement"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern=r'\s')
        reasoning = "a b c"

        result = processor.process(reasoning, sample_context)

        # Should remove all whitespace (default replacement is '')
        assert result == "abc"

    def test_replace_processor_answer_placeholder_basic(self, sample_context):
        """Test ReplaceProcessor with {ANSWER} placeholder replaces ground truth"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern='{ANSWER}', replacement='999')
        reasoning = "The answer is 12. We verify that 12 is correct."
        context = {"answer": "12", "question": "What is the answer?"}

        result = processor.process(reasoning, context)

        # Ground truth answer (12) should be replaced with 999
        assert "12" not in result
        assert "999" in result
        assert result == "The answer is 999. We verify that 999 is correct."

    def test_replace_processor_answer_placeholder_lowercase(self, sample_context):
        """Test ReplaceProcessor with {answer} lowercase placeholder"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern='{answer}', replacement='XYZ')
        reasoning = "The final answer is 12."
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # Should work with lowercase {answer} too
        assert "12" not in result
        assert "XYZ" in result
        assert result == "The final answer is XYZ."

    def test_replace_processor_answer_placeholder_word_boundary(self, sample_context):
        """Test ReplaceProcessor {ANSWER} placeholder respects word boundaries"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern='{ANSWER}', replacement='XXX')
        reasoning = "Calculate 42 and 421 and 142. The answer is 42."
        context = {"answer": "42"}

        result = processor.process(reasoning, context)

        # Only standalone 42 should be replaced (word boundary)
        assert "421" in result  # 421 should remain
        assert "142" in result  # 142 should remain
        # Check that standalone 42 was replaced (not part of 421 or 142)
        assert result.count("42") == 2  # Only in 421 and 142
        assert "XXX" in result
        assert "Calculate XXX and 421 and 142. The answer is XXX." == result

    def test_replace_processor_answer_placeholder_multiple_occurrences(self, sample_context):
        """Test ReplaceProcessor {ANSWER} replaces all occurrences"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern='{ANSWER}', replacement='777')
        reasoning = "First: 12, Second: 12, Third: 12!"
        context = {"answer": "12"}

        result = processor.process(reasoning, context)

        # All occurrences of answer should be replaced
        assert "12" not in result
        assert result.count("777") == 3
        assert result == "First: 777, Second: 777, Third: 777!"

    def test_replace_processor_answer_placeholder_metadata(self, sample_context):
        """Test ReplaceProcessor {ANSWER} placeholder includes actual_pattern in metadata"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern='{ANSWER}', replacement='999')
        reasoning = "The answer is 12."
        context = {"answer": "12"}

        _ = processor.process(reasoning, context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'replace'
        assert metadata['pattern'] == '{ANSWER}'
        assert metadata['replacement'] == '999'
        assert metadata['actual_pattern'] is not None
        # Should show the expanded pattern (word boundary + escaped answer)
        assert '12' in metadata['actual_pattern']
        assert metadata['num_replacements'] == 1
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata

    def test_replace_processor_answer_placeholder_missing_context(self, sample_context):
        """Test ReplaceProcessor {ANSWER} raises error when answer missing from context"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern='{ANSWER}', replacement='999')
        reasoning = "Some reasoning"

        # Context without answer
        context = {"question": "What is 2+2?"}

        with pytest.raises(ValueError, match="'\\{ANSWER\\}' placeholder requires"):
            processor.process(reasoning, context)

    def test_replace_processor_answer_placeholder_in_pipeline(self, sample_context):
        """Test ReplaceProcessor {ANSWER} works in pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "replace('{ANSWER}',replacement='ABC')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "The answer is 12. Therefore 12 is correct."
        context = {"answer": "12", "question": "Test?"}
        result, metadata_list = pipeline.execute(reasoning, context)

        # Should replace answer with ABC
        assert "12" not in result
        assert "ABC" in result
        assert result == "The answer is ABC. Therefore ABC is correct."
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'replace'
        assert metadata_list[0]['pattern'] == '{ANSWER}'

    def test_replace_processor_answer_placeholder_with_shuffle(self, sample_context):
        """Test ReplaceProcessor {ANSWER} combined with shuffle"""
        from pipeline import parse_flow, Pipeline

        flow_str = "replace('{ANSWER}',replacement='999'),shuffle('word',seed=42)"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "The answer is 12."
        context = {"answer": "12"}
        result, metadata_list = pipeline.execute(reasoning, context)

        # Answer should be replaced, then shuffled
        assert "12" not in result
        assert "999" in result
        assert len(metadata_list) == 2
        assert metadata_list[0]['processor'] == 'replace'
        assert metadata_list[1]['processor'] == 'shuffle'

    def test_replace_processor_answer_placeholder_special_chars(self, sample_context):
        """Test ReplaceProcessor {ANSWER} handles answers with special regex chars"""
        from processors import ReplaceProcessor

        processor = ReplaceProcessor(pattern='{ANSWER}', replacement='X')
        reasoning = "The answer is 3.14 which is pi."
        context = {"answer": "3.14"}

        result = processor.process(reasoning, context)

        # Answer with special regex char (.) should be replaced safely
        assert "3.14" not in result
        assert "X" in result
        assert result == "The answer is X which is pi."


class TestReasonIsProcessor:
    """Tests for ReasonIsProcessor - replaces reasoning with custom text pattern"""

    def test_reason_is_answer_mode_basic(self, sample_context):
        """Test ReasonIsProcessor replaces reasoning with pure answer"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='{ANSWER}')
        reasoning = """Let's solve this step by step.
First, we calculate 2 + 2 = 4.
Then, we multiply by 3: 4 × 3 = 12.
Therefore, the answer is 12."""

        result = processor.process(reasoning, sample_context)

        # Should replace entire reasoning with just the answer
        assert result == "12"

    def test_reason_is_answer_with_boxed_answer(self, sample_context):
        """Test ReasonIsProcessor with boxed answer format"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='{ANSWER}')
        reasoning = """Step 1: Calculate 5 + 3 = 8
Step 2: Multiply by 2 = 16
Final answer: \\boxed{16}"""

        result = processor.process(reasoning, sample_context)

        # Should use ground_truth from context
        assert result == "12"

    def test_reason_is_answer_extracts_from_context(self, sample_context):
        """Test ReasonIsProcessor uses answer from context"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='{ANSWER}')
        reasoning = "Some complex reasoning that we don't care about"

        context = {
            "question": "What is 5 + 5?",
            "answer": "10",
            "ground_truth": "10"
        }

        result = processor.process(reasoning, context)

        # Should use answer from context, not parse from reasoning
        assert result == "10"

    def test_reason_is_answer_with_different_answer_formats(self, sample_context):
        """Test ReasonIsProcessor with various answer types"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='{ANSWER}')

        # Test with numeric answer
        context1 = sample_context.copy()
        context1['answer'] = "42"
        result1 = processor.process("reasoning...", context1)
        assert result1 == "42"

        # Test with fractional answer
        context2 = sample_context.copy()
        context2['answer'] = "3/4"
        result2 = processor.process("reasoning...", context2)
        assert result2 == "3/4"

        # Test with text answer
        context3 = sample_context.copy()
        context3['answer'] = "impossible"
        result3 = processor.process("reasoning...", context3)
        assert result3 == "impossible"

    def test_reason_is_answer_get_metadata(self, sample_context):
        """Test ReasonIsProcessor.get_metadata()"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='{ANSWER}')
        reasoning = "Long reasoning text that will be replaced"

        result = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'reason_is'
        assert metadata['text'] == '{ANSWER}'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert metadata['input_stats']['chars'] > 0
        assert metadata['output_stats']['chars'] == len("12")

    def test_reason_is_invalid_mode(self, sample_context):
        """Test ReasonIsProcessor with custom text without placeholder"""
        from processors import ReasonIsProcessor

        # Text without {ANSWER} placeholder should just return the text as-is
        processor = ReasonIsProcessor(text='The result is fixed')
        reasoning = "Some reasoning"

        result = processor.process(reasoning, sample_context)

        # Should return the fixed text
        assert result == "The result is fixed"

    def test_reason_is_missing_answer_in_context(self, sample_context):
        """Test ReasonIsProcessor raises error when answer missing from context"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='{ANSWER}')
        reasoning = "Some reasoning"

        # Context without answer
        context = {"question": "What is 2+2?"}

        with pytest.raises(KeyError, match="'answer' not found in context"):
            processor.process(reasoning, context)

    def test_reason_is_in_pipeline(self, sample_context):
        """Test ReasonIsProcessor works in pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "reason_is('{ANSWER}')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Complex reasoning with multiple steps and calculations"
        context = sample_context.copy()
        result, metadata_list = pipeline.execute(reasoning, context)

        # Should replace reasoning with answer
        assert result == "12"
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'reason_is'
        assert metadata_list[0]['text'] == '{ANSWER}'

    def test_reason_is_combined_with_mask(self, sample_context):
        """Test ReasonIsProcessor combined with mask processor"""
        from pipeline import parse_flow, Pipeline

        # First mask numbers, then replace with answer
        flow_str = "mask('number'),reason_is('{ANSWER}')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Calculate: 2 + 2 = 4"
        context = sample_context.copy()
        result, metadata_list = pipeline.execute(reasoning, context)

        # After mask and reason_is, should just be the answer
        assert result == "12"
        assert len(metadata_list) == 2
        assert metadata_list[0]['processor'] == 'mask'
        assert metadata_list[1]['processor'] == 'reason_is'

    def test_reason_is_empty_reasoning(self, sample_context):
        """Test ReasonIsProcessor with empty reasoning"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='{ANSWER}')
        reasoning = ""

        result = processor.process(reasoning, sample_context)

        # Should still return the answer even with empty reasoning
        assert result == "12"

    def test_reason_is_answer_with_illustrate_mode(self, sample_context):
        """Test ReasonIsProcessor with illustrated answer format"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Thus, the answer is {ANSWER}.')
        reasoning = "Long reasoning that will be replaced"

        result = processor.process(reasoning, sample_context)

        # Should return "Thus, the answer is {answer}"
        assert result == "Thus, the answer is 12."

    def test_reason_is_answer_with_illustrate_different_answers(self, sample_context):
        """Test ReasonIsProcessor illustrated format with various answers"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Thus, the answer is {ANSWER}.')

        # Test with numeric answer
        context1 = sample_context.copy()
        context1['answer'] = "42"
        result1 = processor.process("reasoning...", context1)
        assert result1 == "Thus, the answer is 42."

        # Test with fractional answer
        context2 = sample_context.copy()
        context2['answer'] = "3/4"
        result2 = processor.process("reasoning...", context2)
        assert result2 == "Thus, the answer is 3/4."

        # Test with text answer
        context3 = sample_context.copy()
        context3['answer'] = "impossible"
        result3 = processor.process("reasoning...", context3)
        assert result3 == "Thus, the answer is impossible."

    def test_reason_is_answer_with_illustrate_in_pipeline(self, sample_context):
        """Test ReasonIsProcessor illustrated format in pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "reason_is('Thus, the answer is {ANSWER}.')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Complex reasoning with multiple steps"
        context = sample_context.copy()
        result, metadata_list = pipeline.execute(reasoning, context)

        # Should replace reasoning with "Thus, the answer is {answer}"
        assert result == "Thus, the answer is 12."
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'reason_is'
        assert metadata_list[0]['text'] == 'Thus, the answer is {ANSWER}.'

    def test_reason_is_answer_with_illustrate_get_metadata(self, sample_context):
        """Test ReasonIsProcessor illustrated format metadata"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Thus, the answer is {ANSWER}.')
        reasoning = "Some reasoning text"

        result = processor.process(reasoning, sample_context)
        metadata = processor.get_metadata()

        assert metadata['processor'] == 'reason_is'
        assert metadata['text'] == 'Thus, the answer is {ANSWER}.'
        assert 'input_stats' in metadata
        assert 'output_stats' in metadata
        assert metadata['output_stats']['chars'] == len("Thus, the answer is 12.")

    def test_reason_is_with_special_characters_in_answer(self, sample_context):
        """Test ReasonIsProcessor handles answers with special regex characters"""
        from processors import ReasonIsProcessor

        # Test with LaTeX backslash (common in math problems)
        processor = ReasonIsProcessor(text='Thus, the answer is {ANSWER}.')
        context = {'answer': r'\frac{1}{2}', 'question': 'What is the fraction?'}
        result = processor.process("Some reasoning", context)
        assert result == r'Thus, the answer is \frac{1}{2}.'

        # Test with dollar sign
        context2 = {'answer': '$100', 'question': 'What is the price?'}
        result2 = processor.process("Some reasoning", context2)
        assert result2 == 'Thus, the answer is $100.'

        # Test with parentheses (regex special chars)
        context3 = {'answer': '(a+b)', 'question': 'What is the expression?'}
        result3 = processor.process("Some reasoning", context3)
        assert result3 == 'Thus, the answer is (a+b).'

        # Test with multiple backslashes
        context4 = {'answer': r'\text{solution}', 'question': 'What is it?'}
        result4 = processor.process("Some reasoning", context4)
        assert result4 == r'Thus, the answer is \text{solution}.'

    def test_reason_is_with_math_expression_basic(self, sample_context):
        """Test ReasonIsProcessor with basic math expression"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='The answer is {ANSWER * 0.9}')
        context = {'answer': 100, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # 100 * 0.9 = 90 (int preserved)
        assert result == 'The answer is 90'

    def test_reason_is_with_math_expression_float_result(self, sample_context):
        """Test ReasonIsProcessor with float answer maintains float type"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='The answer is {ANSWER * 0.9}')
        context = {'answer': 100.0, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # 100.0 * 0.9 = 90.0 (float preserved)
        assert result == 'The answer is 90.0'

    def test_reason_is_with_math_expression_int_answer_string(self, sample_context):
        """Test ReasonIsProcessor with string integer answer"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='The answer is {ANSWER * 0.9}')
        context = {'answer': '100', 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # '100' * 0.9 = 90 (int preserved because original is int string)
        assert result == 'The answer is 90'

    def test_reason_is_with_multiple_expressions(self, sample_context):
        """Test ReasonIsProcessor with multiple math expressions"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='The answer is {ANSWER} or {ANSWER * 0.9}')
        context = {'answer': 100, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # Should have both 100 and 90
        assert result == 'The answer is 100 or 90'

    def test_reason_is_with_multiple_expressions_float(self, sample_context):
        """Test ReasonIsProcessor with multiple expressions and float answer"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='The answer is {ANSWER} or {ANSWER * 0.9}')
        context = {'answer': 100.0, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # Should have both 100.0 and 90.0
        assert result == 'The answer is 100.0 or 90.0'

    def test_reason_is_with_addition(self, sample_context):
        """Test ReasonIsProcessor with addition operation"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='The result is {ANSWER + 10}')
        context = {'answer': 42, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        assert result == 'The result is 52'

    def test_reason_is_with_subtraction(self, sample_context):
        """Test ReasonIsProcessor with subtraction operation"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='The result is {ANSWER - 5}')
        context = {'answer': 42, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        assert result == 'The result is 37'

    def test_reason_is_with_division(self, sample_context):
        """Test ReasonIsProcessor with division operation"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Half of the answer is {ANSWER / 2}')
        context = {'answer': 42, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # 42 / 2 = 21 (int preserved)
        assert result == 'Half of the answer is 21'

    def test_reason_is_with_multiplication(self, sample_context):
        """Test ReasonIsProcessor with multiplication operation"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Double is {ANSWER * 2}')
        context = {'answer': 21, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        assert result == 'Double is 42'

    def test_reason_is_with_complex_expression(self, sample_context):
        """Test ReasonIsProcessor with complex math expression"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Result: {ANSWER * 2 + 10}')
        context = {'answer': 20, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # 20 * 2 + 10 = 50
        assert result == 'Result: 50'

    def test_reason_is_with_rounding(self, sample_context):
        """Test ReasonIsProcessor rounds float results for int answers"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='90% is {ANSWER * 0.9}')
        context = {'answer': 33, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # 33 * 0.9 = 29.7, should round to 30 for int answer
        assert result == '90% is 30'

    def test_reason_is_with_negative_number(self, sample_context):
        """Test ReasonIsProcessor with negative numbers"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Negative is {ANSWER * -1}')
        context = {'answer': 42, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        assert result == 'Negative is -42'

    def test_reason_is_without_answer_placeholder(self, sample_context):
        """Test ReasonIsProcessor with expression not containing ANSWER"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='The value is {5 + 5}')
        context = {'answer': 42, 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # Expression without ANSWER should be returned as-is
        assert result == 'The value is {5 + 5}'

    def test_reason_is_case_insensitive_answer(self, sample_context):
        """Test ReasonIsProcessor is case insensitive for ANSWER placeholder"""
        from processors import ReasonIsProcessor

        # Test lowercase 'answer'
        processor1 = ReasonIsProcessor(text='Result: {answer * 2}')
        context = {'answer': 21, 'question': 'What is the number?'}
        result1 = processor1.process("Some reasoning", context)
        assert result1 == 'Result: 42'

        # Test mixed case 'AnSwEr'
        processor2 = ReasonIsProcessor(text='Result: {AnSwEr * 2}')
        result2 = processor2.process("Some reasoning", context)
        assert result2 == 'Result: 42'

    def test_reason_is_with_invalid_expression(self, sample_context):
        """Test ReasonIsProcessor raises error for invalid expression"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Result: {ANSWER * invalid}')
        context = {'answer': 42, 'question': 'What is the number?'}

        with pytest.raises(ValueError, match="Failed to evaluate expression"):
            processor.process("Some reasoning", context)

    def test_reason_is_with_safe_math_functions(self, sample_context):
        """Test ReasonIsProcessor allows safe math functions"""
        from processors import ReasonIsProcessor

        # Test abs()
        processor1 = ReasonIsProcessor(text='Absolute: {abs(ANSWER * -1)}')
        context = {'answer': 42, 'question': 'What is the number?'}
        result1 = processor1.process("Some reasoning", context)
        assert result1 == 'Absolute: 42'

        # Test round()
        processor2 = ReasonIsProcessor(text='Rounded: {round(ANSWER * 0.9)}')
        result2 = processor2.process("Some reasoning", context)
        assert result2 == 'Rounded: 38'

    def test_reason_is_math_expression_in_pipeline(self, sample_context):
        """Test ReasonIsProcessor math expressions work in pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "reason_is('The answer is {ANSWER} or {ANSWER * 0.9}')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = "Complex reasoning"
        context = {'answer': 100, 'question': 'What is it?'}
        result, metadata_list = pipeline.execute(reasoning, context)

        assert result == 'The answer is 100 or 90'
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'reason_is'

    def test_reason_is_with_non_numeric_answer(self, sample_context):
        """Test ReasonIsProcessor with non-numeric answer returns answer as-is"""
        from processors import ReasonIsProcessor

        processor = ReasonIsProcessor(text='Result: {ANSWER * 2}')
        context = {'answer': 'impossible', 'question': 'What is the number?'}
        result = processor.process("Some reasoning", context)

        # Non-numeric answer should be returned as string
        assert result == 'Result: impossible'
