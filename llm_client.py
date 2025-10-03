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
    reasoning_on: bool = True
    temperature: Optional[float] = None


@dataclass
class Response:
    content: str
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
    def __init__(self, api_key="EMPTY", base_url="http://localhost:8001/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self._get_model()

    def _get_model(self):
        models = self.client.models.list()
        return models.data[0].id

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
                {"role": "user", "content": request.query},
            ]
            kwargs = {
                'extra_body': {"chat_template_kwargs": {"thinking": request.reasoning_on}} 
            }
        elif request.model_type == 'gpt-oss':
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.query},
            ]
            kwargs = {"reasoning_effort": "high"} if request.reasoning_on else {"reasoning_effort": "low"}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=timeout,
            temperature=request.temperature,
            **kwargs,
        )

        reasoning_content = None
        if request.model_type == 'deepseek':
            if request.reasoning_on:
                reasoning_content, content = self._parse_deepseek_reasoning_content(response.choices[0].message.content)
            else:
                content = response.choices[0].message.content

        elif request.model_type == 'gpt-oss':
            if request.reasoning_on:
                reasoning_content = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content

        return reasoning_content, content 

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
                reasoning_content, content = self.generate(task.request)
                elapsed_seconds = int(time.time() - start_time)
                task.response = Response(content=content, reasoning_content=reasoning_content, elapsed_seconds=elapsed_seconds, success=True)
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
