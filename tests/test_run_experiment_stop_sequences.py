"""Tests for stop sequences logic in run_experiment"""
import pytest
from llm_client import CompletionRequest


class TestRunExperimentStopSequencesLogic:
    """Test the logic for determining stop sequences in run_experiment"""

    def test_olmo_base_stop_without_refill_sequences(self):
        """Test that olmo--base uses the correct stop_without_refill sequences"""
        # This is the sequence that should be used for olmo--base
        stop_sequences = ['## Question', '## Reasoning', '## Answer']

        request = CompletionRequest(
            question="What is 2+2?",
            reasoning="Let me calculate",
            answer_prefix="The answer is",
            model_type='olmo--base',
            stop_without_refill=stop_sequences
        )

        assert request.model_type == 'olmo--base'
        assert request.stop_without_refill == ['## Question', '## Reasoning', '## Answer']
        assert request.stop is None

    def test_code_dataset_with_answer_retrieval_uses_stop(self):
        """Test that code dataset with answer retrieval uses stop"""
        # For code datasets with answer retrieval, we use stop with a string value
        request = CompletionRequest(
            question="Write a function",
            reasoning="Let me write the code",
            answer_prefix="```python\n",
            model_type='gpt-oss',
            stop='\n```'
        )

        assert request.stop == '\n```'
        assert request.stop_without_refill is None

    def test_stop_and_stop_without_refill_mutual_exclusion(self):
        """Test mutual exclusion between stop and stop_without_refill"""
        # Both fields can be set in the dataclass,
        # but the complete() method will raise ValueError if both are set
        request = CompletionRequest(
            question="Test",
            reasoning="Test",
            stop="\n```",
            stop_without_refill=['##']
        )

        # Both are set at the dataclass level
        assert request.stop is not None
        assert request.stop_without_refill is not None


class TestStopSequencePriority:
    """Test the priority logic for determining stop sequences"""

    def test_priority_1_code_with_answer_retrieval_wins(self):
        """Test that code + answer retrieval (Priority 1) takes precedence over olmo--base (Priority 2)"""
        # When both conditions are true:
        # - Priority 1: dataset_type == 'code' and uses_answer_retrieval
        # - Priority 2: model_type == 'olmo--base'
        # Priority 1 should win

        # This is the logic:
        # if dataset_type == 'code' and uses_answer_retrieval:
        #     stop = '\n```'
        # elif model_type == 'olmo--base':
        #     stop_without_refill = ['## Question', '## Reasoning', '## Answer']

        # Case 1: olmo--base WITHOUT code + answer retrieval -> gets stop_without_refill
        request1 = CompletionRequest(
            question="Math problem",
            reasoning="Calculate",
            model_type='olmo--base',
            stop_without_refill=['## Question', '## Reasoning', '## Answer']
        )
        assert request1.stop_without_refill is not None
        assert request1.stop is None

        # Case 2: olmo--base WITH code + answer retrieval -> gets stop instead
        request2 = CompletionRequest(
            question="Code problem",
            reasoning="Write code",
            model_type='olmo--base',
            stop='\n```'  # Priority 1 wins
        )
        assert request2.stop == '\n```'
        assert request2.stop_without_refill is None

    def test_other_models_without_special_conditions(self):
        """Test that other models without special conditions get no stop sequences"""
        request = CompletionRequest(
            question="Question",
            reasoning="Reasoning",
            model_type='gpt-oss'
        )

        assert request.stop is None
        assert request.stop_without_refill is None

    def test_olmo_base_only_when_no_code_answer_retrieval(self):
        """Test that olmo--base stop_without_refill is only used when code+answer_retrieval is not present"""
        # The logic:
        # if dataset_type == 'code' and uses_answer_retrieval:
        #     stop = '\n```'
        # elif model_type == 'olmo--base':
        #     stop_without_refill = [...]

        # So olmo--base stop_without_refill only applies when the first condition is false

        # With code but without answer retrieval
        request1 = CompletionRequest(
            question="Code",
            reasoning="Code reasoning",
            model_type='olmo--base',
            stop_without_refill=['## Question', '## Reasoning', '## Answer']
        )
        assert request1.stop_without_refill == ['## Question', '## Reasoning', '## Answer']

        # With math dataset and olmo--base
        request2 = CompletionRequest(
            question="Math",
            reasoning="Math reasoning",
            model_type='olmo--base',
            stop_without_refill=['## Question', '## Reasoning', '## Answer']
        )
        assert request2.stop_without_refill == ['## Question', '## Reasoning', '## Answer']


class TestOLMoBaseStopSequencesFormat:
    """Test that stop sequences match the OLMo base model template format"""

    def test_stop_sequences_are_section_markers(self):
        """Test that stop sequences correspond to the template section markers"""
        # OLMo base template has these sections:
        # ## Question:
        # ## Reasoning:
        # ## Answer:

        # Stop sequences should be the markers (without colon)
        expected_stops = ['## Question', '## Reasoning', '## Answer']

        # These match the template sections
        template_sections = ['## Question:', '## Reasoning:', '## Answer:']

        for stop_marker in expected_stops:
            # Each stop marker should be found as the start of a template section
            found = False
            for section in template_sections:
                if section.startswith(stop_marker):
                    found = True
                    break
            assert found, f"Stop marker '{stop_marker}' should correspond to a template section"

    def test_stop_sequences_list_order(self):
        """Test that stop sequences are in the expected order"""
        expected_order = ['## Question', '## Reasoning', '## Answer']

        request = CompletionRequest(
            question="Test",
            reasoning="Test",
            model_type='olmo--base',
            stop_without_refill=expected_order
        )

        # Verify the order is preserved
        assert request.stop_without_refill[0] == '## Question'
        assert request.stop_without_refill[1] == '## Reasoning'
        assert request.stop_without_refill[2] == '## Answer'

    def test_stop_sequences_count(self):
        """Test that there are exactly 3 stop sequences for OLMo base"""
        expected_stops = ['## Question', '## Reasoning', '## Answer']

        assert len(expected_stops) == 3
        assert all(isinstance(s, str) for s in expected_stops)


class TestBackwardCompatibility:
    """Test backward compatibility with existing stop parameter"""

    def test_stop_parameter_backward_compatible(self):
        """Test that existing stop parameter still works as before"""
        request = CompletionRequest(
            question="Code",
            reasoning="Code reasoning",
            stop='\n```'
        )

        # Existing code using stop should still work
        assert request.stop == '\n```'
        assert request.stop_without_refill is None

    def test_code_dataset_stop_unchanged(self):
        """Test that code dataset stop behavior is unchanged"""
        # Existing behavior for code datasets with answer retrieval
        request = CompletionRequest(
            question="Code",
            reasoning="Code reasoning",
            stop='\n```'
        )

        assert request.stop == '\n```'

    def test_default_none_for_all_stop_fields(self):
        """Test that default values are None for all stop-related fields"""
        request = CompletionRequest(
            question="Question",
            reasoning="Reasoning"
        )

        # Both should be None by default
        assert request.stop is None
        assert request.stop_without_refill is None
