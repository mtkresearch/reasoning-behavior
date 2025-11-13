"""
Unit tests for pipeline.py

This test file covers:
- Pipeline class
- Flow parser (parse_flow function)
- End-to-end pipeline execution
"""

import pytest


class TestPipeline:
    """Tests for Pipeline class"""

    def test_pipeline_empty(self, sample_context):
        """Test empty pipeline returns input unchanged"""
        from pipeline import Pipeline

        pipeline = Pipeline([])
        reasoning = "Original text"

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        assert result == reasoning
        assert metadata_list == []

    def test_pipeline_single_processor(self, sample_context):
        """Test pipeline with single processor"""
        from pipeline import Pipeline
        from processors import MaskProcessor

        processors = [MaskProcessor(mode='number')]
        pipeline = Pipeline(processors)
        reasoning = "Calculate 2 + 2 = 4"

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        # Numbers should be masked
        assert '2' not in result
        assert '4' not in result
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'mask'

    def test_pipeline_multiple_processors(self, sample_context):
        """Test pipeline with multiple processors in sequence"""
        from pipeline import Pipeline
        from processors import TruncateProcessor, MaskProcessor, ShuffleProcessor

        processors = [
            TruncateProcessor(mode='last_n_lines', n=2),
            MaskProcessor(mode='number'),
            ShuffleProcessor(mode='line', seed=42)
        ]
        pipeline = Pipeline(processors)
        reasoning = """Line 1: value is 10
Line 2: value is 20
Line 3: value is 30
Line 4: value is 40
Line 5: value is 50"""

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        # Should have been truncated (removed last 2 lines), masked, then shuffled
        assert '50' not in result  # Last lines removed
        assert '40' not in result  # Last lines removed
        # Numbers in remaining lines should be masked
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3  # Only 3 lines remaining
        assert len(metadata_list) == 3
        assert metadata_list[0]['processor'] == 'truncate'
        assert metadata_list[1]['processor'] == 'mask'
        assert metadata_list[2]['processor'] == 'shuffle'

    def test_pipeline_metadata_collection(self, sample_context):
        """Test that pipeline collects metadata from all processors"""
        from pipeline import Pipeline
        from processors import MaskProcessor, ShuffleProcessor

        processors = [
            MaskProcessor(mode='number', mask_char='*'),
            ShuffleProcessor(mode='word', seed=123)
        ]
        pipeline = Pipeline(processors)
        reasoning = "The answer is 42"

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        assert len(metadata_list) == 2

        # Check first processor metadata
        assert metadata_list[0]['processor'] == 'mask'
        assert metadata_list[0]['mode'] == 'number'
        assert metadata_list[0]['mask_char'] == '*'

        # Check second processor metadata
        assert metadata_list[1]['processor'] == 'shuffle'
        assert metadata_list[1]['mode'] == 'word'
        assert metadata_list[1]['seed'] == 123


class TestParseFlow:
    """Tests for flow parser"""

    def test_parse_flow_single_mask(self):
        """Test parsing single mask step"""
        from pipeline import parse_flow

        flow_str = "mask('number')"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].mode == 'number'
        assert processors[0].__class__.__name__ == 'MaskProcessor'

    def test_parse_flow_single_truncate(self):
        """Test parsing single truncate step"""
        from pipeline import parse_flow

        flow_str = "truncate('last_n_lines',n=5)"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].mode == 'last_n_lines'
        assert processors[0].kwargs['n'] == 5
        assert processors[0].__class__.__name__ == 'TruncateProcessor'

    def test_parse_flow_single_shuffle(self):
        """Test parsing single shuffle step"""
        from pipeline import parse_flow

        flow_str = "shuffle('line')"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].mode == 'line'
        assert processors[0].__class__.__name__ == 'ShuffleProcessor'

    def test_parse_flow_multiple_steps(self):
        """Test parsing multiple steps"""
        from pipeline import parse_flow

        flow_str = "mask('number'),shuffle('line')"
        processors = parse_flow(flow_str)

        assert len(processors) == 2
        assert processors[0].__class__.__name__ == 'MaskProcessor'
        assert processors[1].__class__.__name__ == 'ShuffleProcessor'

    def test_parse_flow_with_custom_mask_char(self):
        """Test parsing with custom mask character"""
        from pipeline import parse_flow

        flow_str = "mask('number',mask_char='*')"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].mask_char == '*'

    def test_parse_flow_truncate_ratio(self):
        """Test parsing truncate with ratio"""
        from pipeline import parse_flow

        flow_str = "truncate('last_ratio',ratio=0.3)"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].mode == 'last_ratio'
        assert processors[0].kwargs['ratio'] == 0.3

    def test_parse_flow_shuffle_with_seed(self):
        """Test parsing shuffle with seed"""
        from pipeline import parse_flow

        flow_str = "shuffle('token',seed=42,tokenizer_model='gpt2')"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].mode == 'token'
        assert processors[0].seed == 42
        assert processors[0].kwargs['tokenizer_model'] == 'gpt2'

    def test_parse_flow_complex_combination(self):
        """Test parsing complex combination"""
        from pipeline import parse_flow

        flow_str = "truncate('last_ratio',ratio=0.3),mask('number',mask_char='*'),shuffle('word',seed=42)"
        processors = parse_flow(flow_str)

        assert len(processors) == 3
        assert processors[0].__class__.__name__ == 'TruncateProcessor'
        assert processors[0].kwargs['ratio'] == 0.3
        assert processors[1].__class__.__name__ == 'MaskProcessor'
        assert processors[1].mask_char == '*'
        assert processors[2].__class__.__name__ == 'ShuffleProcessor'
        assert processors[2].seed == 42

    def test_parse_flow_invalid_processor_type(self):
        """Test that invalid processor type raises error"""
        from pipeline import parse_flow

        with pytest.raises(ValueError, match="Unknown processor type"):
            parse_flow("invalid_processor('test')")

    def test_parse_flow_insert_basic(self):
        """Test parsing insert processor with basic parameters"""
        from pipeline import parse_flow

        flow_str = "insert('fix')"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].mode == 'fix'
        assert processors[0].__class__.__name__ == 'InsertProcessor'

    def test_parse_flow_insert_with_parameters(self):
        """Test parsing insert processor with all parameters"""
        from pipeline import parse_flow

        flow_str = "insert('fix',sentence='Maybe the answer is 123.',count=5,seed=42)"
        processors = parse_flow(flow_str)

        assert len(processors) == 1
        assert processors[0].mode == 'fix'
        assert processors[0].sentence == 'Maybe the answer is 123.'
        assert processors[0].count == 5
        assert processors[0].seed == 42


class TestEndToEndPipeline:
    """End-to-end tests for complete pipeline execution"""

    def test_e2e_mask_and_shuffle(self, sample_context):
        """Test mask + shuffle pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "mask('number'),shuffle('line',seed=42)"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = """Step 1: Calculate 10 + 5 = 15
Step 2: Multiply 15 by 2 = 30
Step 3: The answer is 30"""

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        # Numbers should be masked
        assert '10' not in result
        assert '15' not in result
        assert '30' not in result

        # Should have 3 lines (shuffled)
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) == 3

        # Metadata should have 2 entries
        assert len(metadata_list) == 2
        assert metadata_list[0]['processor'] == 'mask'
        assert metadata_list[1]['processor'] == 'shuffle'

    def test_e2e_truncate_mask_shuffle(self, sample_context):
        """Test truncate + mask + shuffle pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "truncate('last_ratio',ratio=0.3),mask('alphabet'),shuffle('word',seed=123)"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = """Line 1 has text
Line 2 has text
Line 3 has text
Line 4 has text
Line 5 has text
Line 6 has text
Line 7 has text
Line 8 has text
Line 9 has text
Line 10 has text"""

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        # Should have removed ~30% of lines
        # Letters should be masked
        # Words should be shuffled

        assert len(metadata_list) == 3
        assert metadata_list[0]['processor'] == 'truncate'
        assert metadata_list[1]['processor'] == 'mask'
        assert metadata_list[2]['processor'] == 'shuffle'

        # Check statistics
        assert metadata_list[0]['removed_lines'] == 3  # 30% of 10
        assert 'text' not in result  # Letters masked

    def test_e2e_answer_removal_then_mask(self, sample_context):
        """Test answer removal + mask pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "truncate('answer_and_after'),mask('number')"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = """Step 1: Calculate 2 + 2 = 4
Step 2: Multiply 4 by 3 = 12
Therefore, the answer is 12.
This line should be removed."""

        context = {"answer": "12"}

        result, metadata_list = pipeline.execute(reasoning, context)

        # Answer line and after should be removed
        assert "Therefore" not in result
        assert "removed" not in result

        # Numbers should be masked
        assert '2' not in result
        assert '4' not in result

        assert len(metadata_list) == 2

    def test_e2e_insert_only(self, sample_context):
        """Test insert processor standalone"""
        from pipeline import parse_flow, Pipeline

        flow_str = "insert('fix',sentence='Maybe the answer is 123.',position='random',count=3,seed=42)"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = """Line 1
Line 2
Line 3
Line 4"""

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        # Inserted sentence should appear
        assert 'Maybe the answer is 123.' in result

        # All original lines should still be present
        assert 'Line 1' in result
        assert 'Line 2' in result
        assert 'Line 3' in result
        assert 'Line 4' in result

        # Check metadata
        assert len(metadata_list) == 1
        assert metadata_list[0]['processor'] == 'insert'
        assert metadata_list[0]['count'] == 3
        assert len(metadata_list[0]['insertion_positions']) == 3

    def test_e2e_insert_and_shuffle(self, sample_context):
        """Test insert + shuffle pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "insert('fix',sentence='Thus answer: 123.',position='random',count=3,seed=42),shuffle('line',seed=42)"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = """Step 1: Calculate something
Step 2: Do more work
Step 3: Final result"""

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        # Inserted sentence should appear
        assert 'Thus answer: 123.' in result

        # Original lines should be present
        assert 'Step 1' in result
        assert 'Step 2' in result
        assert 'Step 3' in result

        # Check metadata
        assert len(metadata_list) == 2
        assert metadata_list[0]['processor'] == 'insert'
        assert metadata_list[1]['processor'] == 'shuffle'

    def test_e2e_mask_insert_shuffle(self, sample_context):
        """Test mask + insert + shuffle pipeline"""
        from pipeline import parse_flow, Pipeline

        flow_str = "mask('number'),insert('fix',sentence='Noise',position='random',count=2,seed=42),shuffle('line',seed=42)"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = """Calculate 10 + 20 = 30
Result is 30
Answer: 30"""

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        # Numbers should be masked
        assert '10' not in result
        assert '20' not in result
        assert '30' not in result

        # Inserted noise should appear
        assert 'Noise' in result

        # Check metadata
        assert len(metadata_list) == 3
        assert metadata_list[0]['processor'] == 'mask'
        assert metadata_list[1]['processor'] == 'insert'
        assert metadata_list[2]['processor'] == 'shuffle'

    def test_e2e_complex_with_insert(self, sample_context):
        """Test complex pipeline: truncate + mask + insert + shuffle"""
        from pipeline import parse_flow, Pipeline

        flow_str = "truncate('last_n_lines',n=2),mask('number'),insert('fix',sentence='Noise',position='random',count=1,seed=42),shuffle('line',seed=42)"
        processors = parse_flow(flow_str)
        pipeline = Pipeline(processors)

        reasoning = """Line 1: value 10
Line 2: value 20
Line 3: value 30
Line 4: value 40
Line 5: value 50"""

        result, metadata_list = pipeline.execute(reasoning, sample_context)

        # Last 2 lines should be removed
        assert 'Line 4' not in result
        assert 'Line 5' not in result

        # Numbers should be masked
        assert '10' not in result
        assert '20' not in result

        # Noise should be inserted
        assert 'Noise' in result

        # Check metadata
        assert len(metadata_list) == 4
        assert metadata_list[0]['processor'] == 'truncate'
        assert metadata_list[1]['processor'] == 'mask'
        assert metadata_list[2]['processor'] == 'insert'
        assert metadata_list[3]['processor'] == 'shuffle'
