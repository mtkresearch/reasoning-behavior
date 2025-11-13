import json
import os

import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Request:
    queries: List[str]
    model_type: str = 'deepseek'
    system_prompt: str = "You are a helpful assistant"
    reasoning_on: bool = True
    temperature: Optional[float] = None
    min_tokens: Optional[int] = None


@dataclass
class CompletionRequest:
    prompt: str
    model_type: str = 'gpt-oss'
    temperature: Optional[float] = None
    max_tokens: int = 20480
    min_tokens: Optional[int] = None


@dataclass
class Response:
    content: str
    history: str
    elapsed_seconds: float
    success: bool
    reasoning_content: Optional[str] = None
    err_message: Optional[str] = None


@dataclass
class Task:
    index: int
    request: Request
    response: Optional[Response] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMClient:
    def __init__(self, mode: str = "openrouter", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize LLM Client with mode selection.

        Args:
            mode: "openrouter" or "local". Default is "openrouter"
            api_key: Optional API key override
            base_url: Optional base URL override
        """
        self.mode = mode

        # Set default values based on mode
        if mode == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            self.base_url = base_url or "https://openrouter.ai/api/v1"
        elif mode == "local":
            self.api_key = api_key or "EMPTY"
            self.base_url = base_url or "http://localhost:8001/v1"
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'openrouter' or 'local'")

        self.timeout = 180.0  # 3 minutes connection timeout

    def _get_model(self, model_type):
        if self.mode == 'local':
            # Use requests to get model list
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout
            )
            response.raise_for_status()
            models = response.json()
            return models['data'][0]['id']
        elif self.mode == 'openrouter':
            assert model_type == 'gpt-oss'
            return 'openai/gpt-oss-120b'

    def _parse_deepseek_reasoning_content(self, content):
        think_count = content.count('</think>')
        if think_count != 1:
            raise ValueError(f"Expected exactly 1 </think> tag, found {think_count}")
        reasoning_content, content = content.split('</think>')
        return reasoning_content, content

    def generate(self, request, timeout=3600*2):
        if request.model_type == 'deepseek':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "<think>Hmm</think>I am DeepSeek"},
                {"role": "user", "content": request.queries[0]},
            ]
            extra_body = {"chat_template_kwargs": {"thinking": request.reasoning_on}}
        elif request.model_type == 'gpt-oss':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.queries[0]},
            ]
            extra_body = {
                "reasoning": {"enabled": request.reasoning_on},
                'provider': {'quantizations': ['bf16', 'fp16']},
            }
        elif request.model_type == 'qwen3':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.queries[0]},
            ]
            extra_body = {}

        # Initialize variables to be used across iterations
        reasoning_details = None
        content = None

        for k in range(len(request.queries)):
            if k >= 1:
                # Preserve reasoning_details when appending assistant message
                assistant_msg = {"role": "assistant", "content": content}
                if reasoning_details is not None:
                    assistant_msg["reasoning_details"] = reasoning_details
                messages.append(assistant_msg)
                messages.append({"role": "user", "content": request.queries[k]})

            payload = {
                'model': self._get_model(request.model_type),
                'messages': messages,
                'temperature': request.temperature,
            }

            if extra_body:
                payload['extra_body'] = extra_body

            # Note: min_tokens is not supported by OpenRouter API
            # if request.min_tokens is not None:
            #     payload['min_tokens'] = request.min_tokens

            # Make API request using requests library
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=timeout
            )

            # Check HTTP errors
            response.raise_for_status()

            # Parse JSON response
            response_data = response.json()

            # Check for errors in response
            if not response_data.get('choices') or len(response_data['choices']) == 0:
                raise Exception("API returned no choices")

            choice = response_data['choices'][0]
            message = choice.get('message', {})

            # Check finish_reason for errors
            finish_reason = choice.get('finish_reason')
            if finish_reason == 'error':
                error_msg = choice.get('error', 'Unknown error')
                raise Exception(f"API error: {error_msg}")
            elif finish_reason == 'content_filter':
                raise Exception("API: Content filtered")
            elif finish_reason == 'length':
                # This is a warning, not an error - content was truncated due to max_tokens
                pass

            # Extract reasoning_details for preservation in next turn
            reasoning_details = message.get('reasoning_details')

            reasoning_content = None
            if request.model_type == 'deepseek':
                if request.reasoning_on:
                    reasoning_content, content = self._parse_deepseek_reasoning_content(message.get('content'))
                else:
                    content = message.get('content')

            elif request.model_type == 'gpt-oss':
                if request.reasoning_on:
                    reasoning_content = message.get('reasoning_content')
                content = message.get('content')

            elif request.model_type == 'qwen3':
                if request.reasoning_on:
                    reasoning_content, content = self._parse_deepseek_reasoning_content(message.get('content'))
                else:
                    raise NotImplementedError

        return reasoning_content, content, messages

    def complete(self, request: CompletionRequest, timeout=3600*2):
        """Text completion (not chat completion)"""
        payload = {
            'model': self._get_model(request.model_type),
            'prompt': request.prompt,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
        }

        # Note: min_tokens is not supported by OpenRouter API
        # if request.min_tokens is not None:
        #     payload['min_tokens'] = request.min_tokens

        # Make API request using requests library
        response = requests.post(
            url=f"{self.base_url}/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=timeout
        )

        # Check HTTP errors
        response.raise_for_status()

        # Parse JSON response
        response_data = response.json()

        # Check for errors in response
        if not response_data.get('choices') or len(response_data['choices']) == 0:
            raise Exception("API returned no choices")

        choice = response_data['choices'][0]

        # Check finish_reason for errors
        finish_reason = choice.get('finish_reason')
        if finish_reason == 'error':
            error_msg = choice.get('error', 'Unknown error')
            raise Exception(f"API error: {error_msg}")
        elif finish_reason == 'content_filter':
            raise Exception("API: Content filtered")
        elif finish_reason == 'length':
            # This is a warning, not an error - content was truncated due to max_tokens
            pass

        content = choice.get('text')
        return content

    def generate_concurrent(self, tasks, max_workers=None):
        """
        Generate responses concurrently for multiple tasks.

        Args:
            tasks: Iterable of Task dataclass instances
            max_workers: Maximum number of concurrent workers (default: None, uses ThreadPoolExecutor default)

        Yields:
            Task dataclass instances with response field populated as they complete
        """
        def _generate_task(task):
            try:
                start_time = time.time()
                reasoning_content, content, history = self.generate(task.request)
                elapsed_seconds = int(time.time() - start_time)
                task.response = Response(content=content, history=json.dumps(history), reasoning_content=reasoning_content, elapsed_seconds=elapsed_seconds, success=True)
                return task
            except Exception as e:
                elapsed_seconds = int(time.time() - start_time if 'start_time' in locals() else 0)
                task.response = Response(content="", history="", elapsed_seconds=elapsed_seconds, success=False, err_message=str(e))
                return task

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(_generate_task, task)
                futures.append(future)

            for future in as_completed(futures):
                yield future.result()

    def complete_concurrent(self, tasks, max_workers=None):
        """
        Generate completions concurrently for multiple tasks.

        Args:
            tasks: Iterable of Task dataclass instances (with CompletionRequest)
            max_workers: Maximum number of concurrent workers (default: None, uses ThreadPoolExecutor default)

        Yields:
            Task dataclass instances with response field populated as they complete
        """
        def _complete_task(task):
            try:
                start_time = time.time()
                content = self.complete(task.request)
                elapsed_seconds = int(time.time() - start_time)
                task.response = Response(content=content, history="", reasoning_content=None, elapsed_seconds=elapsed_seconds, success=True)
                return task
            except Exception as e:
                elapsed_seconds = int(time.time() - start_time if 'start_time' in locals() else 0)
                task.response = Response(content="", history="", elapsed_seconds=elapsed_seconds, success=False, err_message=str(e))
                return task

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(_complete_task, task)
                futures.append(future)

            for future in as_completed(futures):
                yield future.result()
