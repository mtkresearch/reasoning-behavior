from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import time


@dataclass
class Request:
    query: str
    model_type: str = 'deepseek'
    system_prompt: str = "You are a helpful assistant"
    extra_body: Optional[Dict[str, Any]] = None


@dataclass
class Response:
    content: str
    elapsed_seconds: float
    success: bool
    err_message: Optional[str] = None


@dataclass
class Task:
    index: int
    request: Request
    response: Optional[Response] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMClient:
    def __init__(self, api_key="EMPTY", base_url="http://localhost:8001/v1", timeout=3600):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = self._get_model()

    def _get_model(self):
        models = self.client.models.list()
        return models.data[0].id

    def generate(self, messages, extra_body=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body
        )
        return response.choices[0].message.content

    def _prepare_messages_and_extra_body(self, request):
        if request.model_type == 'deepseek':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "<think>Hmm</think>I am DeepSeek"},
                {"role": "user", "content": request.query},
            ]
            extra_body = request.extra_body or {"chat_template_kwargs": {"thinking": True}}
        return messages, extra_body

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
                messages, extra_body = self._prepare_messages_and_extra_body(task.request)
                content = self.generate(messages, extra_body)
                elapsed_seconds = int(time.time() - start_time)
                task.response = Response(content=content, elapsed_seconds=elapsed_seconds, success=True)
                return task
            except Exception as e:
                elapsed_seconds = int(time.time() - start_time if 'start_time' in locals() else 0)
                task.response = Response(content="", elapsed_seconds=elapsed_seconds, success=False, err_message=str(e))
                return task

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(_generate_task, task)
                futures.append(future)

            for future in as_completed(futures):
                yield future.result()
