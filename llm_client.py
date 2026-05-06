import json
import os
from time import sleep
import logging

import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEBUG = False

# Setup logger for LLM client
logger = logging.getLogger(__name__)


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
    question: str
    reasoning: str
    answer_prefix: str = ""
    model_type: str = 'gpt-oss'
    temperature: Optional[float] = None
    max_tokens: int = 20480
    min_tokens: Optional[int] = None
    stop: str = None
    stop_without_refill: Optional[List[str]] = None
    system_prompt: str = "You are a helpful assistant"
    reasoning_on: bool = True


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
    def __init__(self, mode: str = "openrouter", api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 180):
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

        self.timeout = timeout

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
            if model_type == 'gpt-oss':
                return 'openai/gpt-oss-120b'
            elif model_type == 'deepseek':
                return 'deepseek/deepseek-chat-v3.1'
            elif model_type == 'deepseek-base':
                return 'deepseek/deepseek-v3.1-base'
            elif model_type == 'olmo':
                return 'allenai/olmo-3.1-32b-instruct'
            elif model_type == 'olmo--base':
                raise ValueError(f"Unsupported model_type for openrouter: {model_type}")
            elif model_type == 'qwen3':
                raise ValueError(f"Unsupported model_type for openrouter: {model_type}")
            else:
                raise ValueError(f"Unsupported model_type for openrouter: {model_type}")

    def _parse_deepseek_reasoning_content(self, content):
        think_count = content.count('</think>')
        if think_count != 1:
            raise ValueError(f"Expected exactly 1 </think> tag, found {think_count}")
        reasoning_content, content = content.split('</think>')
        return reasoning_content, content
    
    def _get_extra_body(self, request, task='chat'):
        extra_body = {}
        if request.model_type == 'deepseek':
            if task == 'chat':
                extra_body["reasoning"] = {"enabled": request.reasoning_on}
        elif request.model_type == 'gpt-oss':
            if task == 'chat':
                extra_body["reasoning"] = {"enabled": request.reasoning_on}
        elif request.model_type == 'olmo':
            if task == 'chat':
                extra_body["reasoning"] = {"enabled": request.reasoning_on}
        elif request.model_type == 'olmo--base':
            pass  # base model 不支援 reasoning.enabled flag
        elif request.model_type == 'qwen3':
            pass
        return extra_body

    def _get_provider_preferences(self, request):
        """Get provider preferences for OpenRouter API"""
        if request.model_type in ['gpt-oss']:
            return {'quantizations': ['fp4']}
        elif request.model_type in ['deepseek']:
            return {'quantizations': ['fp4', 'fp8']}
        return None

    def _apply_completion_template(
        self,
        question: str,
        reasoning: str,
        answer_prefix: str,
        model_type: str,
        system_prompt: str = "You are a helpful assistant",
        reasoning_on: bool = True,
    ) -> str:
        """
        Apply chat template for text completion.

        Both local vLLM and OpenRouter require applying a chat template
        to format the prompt correctly for the completions API.

        Args:
            question: The question/problem to solve
            reasoning: The reasoning content to prefill
            answer_prefix: The answer prefix to continue from
            model_type: Type of model (e.g., 'gpt-oss', 'deepseek-v3')
            system_prompt: System instruction (default: "You are a helpful assistant")
            reasoning_on: Whether to enable reasoning mode (default: True)

        Returns:
            Formatted prompt with template applied
        """
        # Apply templates for specific models (works for both local and openrouter)
        if model_type == 'gpt-oss':
            # Build system message (based on gpt-oss template)
            current_date = datetime.now().strftime("%Y-%m-%d")

            system_message = f"{system_prompt}\n"
            system_message += "Knowledge cutoff: 2024-06\n"
            system_message += f"Current date: {current_date}\n\n"
            # Set reasoning effort based on reasoning_on flag
            reasoning_effort = "high" if reasoning_on else "low"
            system_message += f"Reasoning: {reasoning_effort}\n\n"
            system_message += "# Valid channels: analysis, commentary, final. Channel must be included for every message."

            # Build complete prompt with question, reasoning, and answer_prefix
            template = f"<|start|>system<|message|>{system_message}<|end|>"
            template += f"<|start|>user<|message|>{question}<|end|>"
            template += f"<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
            template += f"<|start|>assistant<|channel|>final<|message|>{answer_prefix}"
            return template

        elif model_type == 'deepseek':
            # DeepSeek uses new chat template format (v3.2)
            # Thinking mode: <｜Assistant｜><think>{reasoning}</think>{answer}
            # Non-thinking mode: <｜Assistant｜></think>{answer}
            template = f"<｜begin▁of▁sentence｜>{system_prompt}<｜User｜>Who are you?<｜Assistant｜></think>I am DeepSeek<｜end▁of▁sentence｜><｜User｜>{question}"

            if reasoning_on and reasoning:
                # Thinking mode with prefilled reasoning
                template += f"<｜Assistant｜><think>{reasoning}"

            else:
                # Non-thinking mode
                template += f"<｜Assistant｜>"

            template += f"</think>{answer_prefix}"
            return template
        
        elif model_type == 'deepseek-base':

            template = f"<｜begin▁of▁sentence｜>{system_prompt}\n\nUSER:\nWho are you?\n\nASSISTANT:\nI am DeepSeek\n\nUSER:\n{question}\n\n"

            if reasoning_on and reasoning:
                # Thinking mode with prefilled reasoning
                template += f"ASSISTANT'S THINK:\n{reasoning}\n\n"

            template += f"ASSISTANT:\n{answer_prefix}"
            return template

        elif model_type == 'olmo':
            # OLMo requires reasoning to be enabled
            if not reasoning_on:
                raise ValueError("OLMo requires reasoning to be enabled (reasoning_on=True)")

            # OLMo uses the Olmo-specific chat template format
            # System message: <|im_start|>system\n{content}<|im_end|>\n
            # User message: <|im_start|>user\n{content}<|im_end|>\n
            # Assistant message: <|im_start|>assistant\n<think>{reasoning}</think>{answer}<|im_end|>\n
            template = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            template += f"<|im_start|>user\n{question}<|im_end|>\n"
            template += f"<|im_start|>assistant\n<think>{reasoning}</think>"
            template += f"{answer_prefix}"
            return template

        elif model_type == 'olmo--base':
            # OLMo base model uses plain text format (no chat tokens)
            template = f"{system_prompt}\n\n"
            template += f"## Question:\n{question}\n\n"

            if reasoning_on and reasoning:
                template += f"## Reasoning:\n{reasoning}\n\n"

            template += f"## Answer:\n{answer_prefix}"
            return template

        # For other models or unknown types
        raise Exception

    def generate(self, request, use_complete_api=False):
        if use_complete_api:
            reasoning_content, content, messages = self.chat_complete_with_complete_api(request)
            return reasoning_content, content, messages
    
        if request.model_type == 'deepseek':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.queries[0]},
            ]
        elif request.model_type == 'deepseek-base':
            raise Exception
        elif request.model_type == 'gpt-oss':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.queries[0]},
            ]
        elif request.model_type == 'olmo':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.queries[0]},
            ]
        elif request.model_type == 'olmo--base':
            raise Exception("OLMo base model only supports completion API. Use complete() method instead.")

        elif request.model_type == 'qwen3':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.queries[0]},
            ]

        extra_body = self._get_extra_body(request)

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
                'max_tokens': 50000
            }

            if extra_body:
                payload['extra_body'] = extra_body

            # Add provider preferences if available
            provider_prefs = self._get_provider_preferences(request)
            if provider_prefs:
                payload['provider'] = provider_prefs

            # Note: min_tokens is not supported by OpenRouter API
            # if request.min_tokens is not None:
            #     payload['min_tokens'] = request.min_tokens

            if DEBUG:
                print(payload)

            # Log request payload
            logger.info(f"[CHAT] Model: {payload['model']}, Temp: {payload.get('temperature')}, Msgs: {len(payload['messages'])}")
            logger.info(f"[CHAT] Messages: {json.dumps(payload['messages'], ensure_ascii=False)}")

            # Make API request using requests library
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=self.timeout
            )

            # Check HTTP errors
            response.raise_for_status()

            # Parse JSON response
            response_data = response.json()
            logger.info(f"[CHAT] Response status: {response.status_code}")
            if DEBUG:
                print(response_data)

            # Check for errors in response
            if not response_data.get('choices') or len(response_data['choices']) == 0:
                logger.error("[CHAT] API returned no choices")
                raise Exception("API returned no choices")

            choice = response_data['choices'][0]
            message = choice.get('message', {})

            # Check finish_reason for errors
            finish_reason = choice.get('finish_reason')
            if finish_reason == 'error':
                error_msg = choice.get('error', 'Unknown error')
                logger.error(f"[CHAT] API error: {error_msg}")
                raise Exception(f"API error: {error_msg}")
            elif finish_reason == 'content_filter':
                logger.warning("[CHAT] Content filtered by API")
                raise Exception("API: Content filtered")
            elif finish_reason == 'length':
                logger.warning("[CHAT] Response truncated (max_tokens limit reached)")

            # Extract reasoning_details for preservation in next turn
            reasoning_details = message.get('reasoning_details')

            reasoning_content = None
            if request.model_type == 'deepseek':
                if request.reasoning_on:
                    reasoning_content = message.get('reasoning')
                content = message.get('content')

            elif request.model_type == 'gpt-oss':
                if request.reasoning_on:
                    # OpenRouter returns reasoning in 'reasoning' field for gpt-oss
                    reasoning_content = message.get('reasoning')
                content = message.get('content')

            elif request.model_type == 'olmo':
                if request.reasoning_on:
                    # OpenRouter returns reasoning in 'reasoning' field for olmo
                    reasoning_content = message.get('reasoning')
                content = message.get('content')

            elif request.model_type == 'qwen3':
                if request.reasoning_on:
                    reasoning_content, content = self._parse_deepseek_reasoning_content(message.get('content'))
                else:
                    raise NotImplementedError

        return reasoning_content, content, messages
    
    def chat_complete_with_complete_api(self, request):
        prompt = self._apply_completion_template(
            question=request.queries[0],
            reasoning='<<<HALT>>>',
            answer_prefix='',
            model_type=request.model_type,
            system_prompt=request.system_prompt,
            reasoning_on=True,
        )
        prompt = prompt.split('<<<HALT>>>')[0]
        payload = {
            'model': self._get_model(request.model_type),
            'prompt': prompt,
            'temperature': request.temperature,
            'stop': ['<｜end▁of▁sentence｜>'],
            'max_tokens': 160000,
        }

        extra_body = self._get_extra_body(request, task='completion')
        if extra_body:
            payload['extra_body'] = extra_body

        # Add provider preferences if available
        provider_prefs = self._get_provider_preferences(request)
        if provider_prefs:
            payload['provider'] = provider_prefs

        response = requests.post(
            url=f"{self.base_url}/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.timeout
        )

        # Check HTTP errors
        try:
            response.raise_for_status()
        except Exception as e:
            error_data = response.json()
            logger.error(f"[COMPLETION] HTTP Error {response.status_code}: {error_data}")
            print(f'Error: {error_data}')
            sleep(1.0)
            raise e

        response_data = response.json()
        reasoning_content = response_data['choices'][0]['reasoning']
        content = response_data['choices'][0]['text']
        messages = None
        return reasoning_content, content, messages

    def complete(self, request: CompletionRequest):
        """Text completion (not chat completion)"""
        # Apply template to format the prompt
        formatted_prompt = self._apply_completion_template(
            question=request.question,
            reasoning=request.reasoning,
            answer_prefix=request.answer_prefix,
            model_type=request.model_type,
            system_prompt=request.system_prompt,
            reasoning_on=request.reasoning_on,
        )

        payload = {
            'model': self._get_model(request.model_type),
            'prompt': formatted_prompt,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
        }

        # Add stop sequences parameter
        # Mutual exclusion check: stop and stop_without_refill cannot be used together
        if request.stop is not None and request.stop_without_refill is not None:
            raise ValueError("stop and stop_without_refill are mutually exclusive. Use only one.")

        # Set stop sequences
        if request.stop is not None:
            payload['stop'] = [request.stop]
        elif request.stop_without_refill is not None:
            payload['stop'] = request.stop_without_refill

        if DEBUG:
            print('prompt:', formatted_prompt)

        extra_body = self._get_extra_body(request, task='completion')
        if extra_body:
            payload['extra_body'] = extra_body

        # Add provider preferences if available
        provider_prefs = self._get_provider_preferences(request)
        if provider_prefs:
            payload['provider'] = provider_prefs

        # Note: min_tokens is not supported by OpenRouter API
        # if request.min_tokens is not None:
        #     payload['min_tokens'] = request.min_tokens

        # Make API request using requests library
        if DEBUG:
            print(payload)

        # Log request payload
        logger.info(f"[COMPLETION] Model: {payload['model']}, Temp: {payload.get('temperature')}, "
                   f"Max tokens: {payload.get('max_tokens')}")
        logger.info(f"[COMPLETION] Prompt: {formatted_prompt}")

        response = requests.post(
            url=f"{self.base_url}/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.timeout
        )

        # Check HTTP errors
        try:
            response.raise_for_status()
        except Exception as e:
            error_data = response.json()
            logger.error(f"[COMPLETION] HTTP Error {response.status_code}: {error_data}")
            print(f'Error: {error_data}')
            sleep(1.0)
            raise e

        # Parse JSON response
        response_data = response.json()
        logger.info(f"[COMPLETION] Response status: {response.status_code}")

        # Check for errors in response
        if not response_data.get('choices') or len(response_data['choices']) == 0:
            raise Exception("API returned no choices")

        choice = response_data['choices'][0]
        if DEBUG:
            print(response_data['choices'])

        # Check finish_reason for errors
        finish_reason = choice.get('finish_reason')
        if finish_reason == 'error':
            error_msg = choice.get('error', 'Unknown error')
            logger.error(f"[COMPLETION] API error: {error_msg}")
            raise Exception(f"API error: {error_msg}")
        elif finish_reason == 'content_filter':
            logger.warning("[COMPLETION] Content filtered by API")
            raise Exception("API: Content filtered")
        elif finish_reason == 'length':
            logger.warning("[COMPLETION] Response truncated (max_tokens limit reached)")

        content = choice.get('text')
        if request.stop and finish_reason == 'stop':
            content += request.stop 
        return content

    def generate_concurrent(self, tasks, max_workers=None, **kwargs):
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
                reasoning_content, content, history = self.generate(task.request, **kwargs)
                elapsed_seconds = int(time.time() - start_time)
                task.response = Response(content=content, history=json.dumps(history), reasoning_content=reasoning_content, elapsed_seconds=elapsed_seconds, success=True)
                return task
            except Exception as e:
                elapsed_seconds = int(time.time() - start_time if 'start_time' in locals() else 0)
                task_id = task.metadata.get('unique_id', f'task_{task.index}') if task.metadata else f'task_{task.index}'
                logger.error(f"[GENERATE] Failed for {task_id}: {str(e)}")
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
                task_id = task.metadata.get('unique_id', f'task_{task.index}') if task.metadata else f'task_{task.index}'
                logger.error(f"[COMPLETE] Failed for {task_id}: {str(e)}")
                task.response = Response(content="", history="", elapsed_seconds=elapsed_seconds, success=False, err_message=str(e))
                return task

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(_complete_task, task)
                futures.append(future)

            for future in as_completed(futures):
                yield future.result()


if __name__ == "__main__":
    # Try-run example for deepseek-base completion
    try:
        # Initialize client
        client = LLMClient(mode="openrouter")

        # Create a completion request
        request = CompletionRequest(
            question="What is 2 + 2?",
            reasoning="Let me think about this step by step. We need to add two and two.",
            answer_prefix="The answer is ",
            model_type="deepseek-base",
            temperature=0.7,
            max_tokens=100,
        )

        print("[Try-run] Deepseek-base completion example")
        print(f"Question: {request.question}")
        print(f"Reasoning: {request.reasoning}")
        print(f"Answer prefix: {request.answer_prefix}")
        print()

        # Call complete
        result = client.complete(request)
        print(f"Result: {result}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
