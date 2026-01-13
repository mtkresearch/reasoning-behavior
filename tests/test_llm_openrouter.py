"""
Unit tests for LLMClient with OpenRouter GPT-OSS-120b

These tests verify that the LLMClient correctly interacts with OpenRouter's GPT-OSS-120b model.
Tests include both chat completion and text completion modes, with and without reasoning enabled.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from llm_client import LLMClient, Request, CompletionRequest, Response, Task


class TestLLMClientOpenRouterInit:
    """Tests for LLMClient initialization with OpenRouter mode"""

    def test_init_openrouter_with_env_key(self, monkeypatch):
        """Test initialization with API key from environment"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
        client = LLMClient(mode="openrouter")

        assert client.mode == "openrouter"
        assert client.api_key == "test-key-123"
        assert client.base_url == "https://openrouter.ai/api/v1"

    def test_init_openrouter_with_explicit_key(self):
        """Test initialization with explicitly provided API key"""
        client = LLMClient(mode="openrouter", api_key="explicit-key")

        assert client.mode == "openrouter"
        assert client.api_key == "explicit-key"
        assert client.base_url == "https://openrouter.ai/api/v1"

    def test_init_openrouter_without_key_raises_error(self, monkeypatch):
        """Test that missing API key raises ValueError"""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
            LLMClient(mode="openrouter")

    def test_init_openrouter_custom_base_url(self, monkeypatch):
        """Test initialization with custom base URL"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = LLMClient(
            mode="openrouter",
            base_url="https://custom.openrouter.ai/api/v1"
        )

        assert client.base_url == "https://custom.openrouter.ai/api/v1"


class TestLLMClientGetModel:
    """Tests for _get_model method with OpenRouter"""

    def test_get_model_gpt_oss(self, monkeypatch):
        """Test that gpt-oss returns correct model identifier"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = LLMClient(mode="openrouter")

        model = client._get_model('gpt-oss')
        assert model == 'openai/gpt-oss-120b'

    def test_get_model_non_gpt_oss_raises_assertion(self, monkeypatch):
        """Test that unsupported model types raise ValueError"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = LLMClient(mode="openrouter")

        with pytest.raises(ValueError):
            client._get_model('deepseek')


class TestLLMClientGenerateGPTOSS:
    """Tests for generate method with GPT-OSS model"""

    @patch('llm_client.requests.post')
    def test_generate_gpt_oss_with_reasoning(self, mock_post, monkeypatch):
        """Test chat completion with reasoning enabled"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'The answer is 42',
                    'reasoning_content': 'Let me think about this problem step by step...'
                },
                'finish_reason': 'stop'
            }]
        }
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")
        request = Request(
            queries=["What is the answer to life?"],
            model_type='gpt-oss',
            reasoning_on=True
        )

        reasoning_content, content, messages = client.generate(request)

        assert reasoning_content == 'Let me think about this problem step by step...'
        assert content == 'The answer is 42'
        assert len(messages) == 2  # system + user

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['url'] == 'https://openrouter.ai/api/v1/chat/completions'
        assert 'Bearer test-key' in call_args[1]['headers']['Authorization']

    @patch('llm_client.requests.post')
    def test_generate_gpt_oss_without_reasoning(self, mock_post, monkeypatch):
        """Test chat completion with reasoning disabled"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'The answer is 42'
                },
                'finish_reason': 'stop'
            }]
        }
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")
        request = Request(
            queries=["What is the answer?"],
            model_type='gpt-oss',
            reasoning_on=False
        )

        reasoning_content, content, messages = client.generate(request)

        assert reasoning_content is None
        assert content == 'The answer is 42'

    @patch('llm_client.requests.post')
    def test_generate_gpt_oss_multi_turn(self, mock_post, monkeypatch):
        """Test multi-turn conversation"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # First response
        mock_response_1 = MagicMock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'First answer',
                    'reasoning_content': 'First reasoning',
                    'reasoning_details': {'tokens': 100}
                },
                'finish_reason': 'stop'
            }]
        }

        # Second response
        mock_response_2 = MagicMock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Second answer',
                    'reasoning_content': 'Second reasoning'
                },
                'finish_reason': 'stop'
            }]
        }

        mock_post.side_effect = [mock_response_1, mock_response_2]

        client = LLMClient(mode="openrouter")
        request = Request(
            queries=["First question?", "Second question?"],
            model_type='gpt-oss',
            reasoning_on=True
        )

        reasoning_content, content, messages = client.generate(request)

        # Should return last turn's content
        assert content == 'Second answer'
        assert reasoning_content == 'Second reasoning'

        # Should have system, user, assistant (with reasoning_details), user
        assert len(messages) == 4
        assert messages[2]['role'] == 'assistant'
        assert messages[2]['content'] == 'First answer'
        assert 'reasoning_details' in messages[2]
        assert messages[2]['reasoning_details'] == {'tokens': 100}


class TestLLMClientCompleteGPTOSS:
    """Tests for complete method (text completion) with GPT-OSS"""

    @patch('llm_client.requests.post')
    def test_complete_gpt_oss_basic(self, mock_post, monkeypatch):
        """Test basic text completion"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'text': 'Completed text here',
                'finish_reason': 'stop'
            }]
        }
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")
        request = CompletionRequest(
            question="What is 2 + 2?",
            reasoning="Let me calculate this",
            answer_prefix="The answer is",
            model_type='gpt-oss',
            max_tokens=1024
        )

        content = client.complete(request)

        assert content == 'Completed text here'

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['url'] == 'https://openrouter.ai/api/v1/completions'

    @patch('llm_client.requests.post')
    def test_complete_gpt_oss_with_temperature(self, mock_post, monkeypatch):
        """Test text completion with temperature parameter"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'text': 'Creative response',
                'finish_reason': 'stop'
            }]
        }
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")
        request = CompletionRequest(
            question="Write a story about a cat",
            reasoning="Let me think creatively",
            answer_prefix="Once upon a time",
            model_type='gpt-oss',
            temperature=0.9,
            max_tokens=2048
        )

        content = client.complete(request)

        assert content == 'Creative response'


class TestLLMClientErrorHandling:
    """Tests for error handling with OpenRouter API"""

    @patch('llm_client.requests.post')
    def test_generate_api_error_response(self, mock_post, monkeypatch):
        """Test handling of API error in response"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'finish_reason': 'error',
                'error': 'Rate limit exceeded'
            }]
        }
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")
        request = Request(queries=["Test"], model_type='gpt-oss')

        with pytest.raises(Exception, match="API error: Rate limit exceeded"):
            client.generate(request)

    @patch('llm_client.requests.post')
    def test_generate_content_filter(self, mock_post, monkeypatch):
        """Test handling of content filter"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'finish_reason': 'content_filter'
            }]
        }
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")
        request = Request(queries=["Test"], model_type='gpt-oss')

        with pytest.raises(Exception, match="Content filtered"):
            client.generate(request)

    @patch('llm_client.requests.post')
    def test_generate_no_choices(self, mock_post, monkeypatch):
        """Test handling of empty choices in response"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'choices': []}
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")
        request = Request(queries=["Test"], model_type='gpt-oss')

        with pytest.raises(Exception, match="no choices"):
            client.generate(request)

    @patch('llm_client.requests.post')
    def test_generate_http_error(self, mock_post, monkeypatch):
        """Test handling of HTTP errors"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")
        request = Request(queries=["Test"], model_type='gpt-oss')

        with pytest.raises(Exception, match="401 Unauthorized"):
            client.generate(request)


class TestLLMClientConcurrentGeneration:
    """Tests for concurrent generation with OpenRouter"""

    @patch('llm_client.requests.post')
    def test_generate_concurrent_multiple_tasks(self, mock_post, monkeypatch):
        """Test concurrent generation of multiple tasks"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Answer',
                    'reasoning_content': 'Reasoning'
                },
                'finish_reason': 'stop'
            }]
        }
        mock_post.return_value = mock_response

        client = LLMClient(mode="openrouter")

        tasks = [
            Task(
                index=i,
                request=Request(
                    queries=[f"Question {i}"],
                    model_type='gpt-oss',
                    reasoning_on=True
                )
            )
            for i in range(3)
        ]

        results = list(client.generate_concurrent(tasks, max_workers=2))

        assert len(results) == 3
        for task in results:
            assert task.response.success is True
            assert task.response.content == 'Answer'
            assert task.response.reasoning_content == 'Reasoning'

    @patch('llm_client.requests.post')
    def test_generate_concurrent_with_failures(self, mock_post, monkeypatch):
        """Test concurrent generation with some failures"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock responses: first succeeds, second fails
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            'choices': [{
                'message': {'content': 'Success'},
                'finish_reason': 'stop'
            }]
        }

        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.raise_for_status.side_effect = Exception("Server error")

        mock_post.side_effect = [success_response, fail_response]

        client = LLMClient(mode="openrouter")

        tasks = [
            Task(index=0, request=Request(queries=["Q1"], model_type='gpt-oss')),
            Task(index=1, request=Request(queries=["Q2"], model_type='gpt-oss'))
        ]

        results = list(client.generate_concurrent(tasks, max_workers=2))

        assert len(results) == 2

        # One should succeed
        successes = [t for t in results if t.response.success]
        assert len(successes) == 1

        # One should fail
        failures = [t for t in results if not t.response.success]
        assert len(failures) == 1
        assert "Server error" in failures[0].response.err_message


class TestLLMClientExtraBody:
    """Tests for _get_extra_body method with GPT-OSS"""

    def test_get_extra_body_gpt_oss_reasoning_on(self, monkeypatch):
        """Test extra_body configuration for GPT-OSS with reasoning enabled"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = LLMClient(mode="openrouter")

        request = Request(
            queries=["Test"],
            model_type='gpt-oss',
            reasoning_on=True
        )

        extra_body = client._get_extra_body(request)

        assert extra_body['reasoning']['enabled'] is True
        assert 'provider' not in extra_body  # provider is now separate

    def test_get_extra_body_gpt_oss_reasoning_off(self, monkeypatch):
        """Test extra_body configuration for GPT-OSS with reasoning disabled"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = LLMClient(mode="openrouter")

        request = Request(
            queries=["Test"],
            model_type='gpt-oss',
            reasoning_on=False
        )

        extra_body = client._get_extra_body(request)

        assert extra_body['reasoning']['enabled'] is False
        assert 'provider' not in extra_body  # provider is now separate


class TestLLMClientProviderPreferences:
    """Tests for _get_provider_preferences method"""

    def test_get_provider_preferences_gpt_oss(self, monkeypatch):
        """Test provider preferences for GPT-OSS"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = LLMClient(mode="openrouter")

        request = Request(
            queries=["Test"],
            model_type='gpt-oss',
            reasoning_on=True
        )

        provider_prefs = client._get_provider_preferences(request)

        assert provider_prefs is not None
        assert 'quantizations' in provider_prefs
        assert 'bf16' in provider_prefs['quantizations']
        assert 'fp16' in provider_prefs['quantizations']

    def test_get_provider_preferences_other_models(self, monkeypatch):
        """Test provider preferences for non-GPT-OSS models"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = LLMClient(mode="openrouter")

        request = Request(
            queries=["Test"],
            model_type='qwen3',
            reasoning_on=True
        )

        provider_prefs = client._get_provider_preferences(request)

        assert provider_prefs is None
