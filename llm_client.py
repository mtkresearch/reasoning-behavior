from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class Task:
    index: int
    messages: List[Dict[str, str]]
    extra_body: Optional[Dict[str, Any]] = None


@dataclass
class Response:
    index: int
    content: str


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

    def generate(self, messages, extra_body=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body
        )
        return response.choices[0].message.content

    def generate_concurrent(self, tasks, max_workers=None):
        """
        Generate responses concurrently for multiple tasks.

        Args:
            tasks: Iterable of Task dataclass instances
            max_workers: Maximum number of concurrent workers (default: None, uses ThreadPoolExecutor default)

        Yields:
            Response dataclass instances as they complete
        """
        def _generate_task(task):
            try:
                content = self.generate(task.messages, task.extra_body)
                return Response(index=task.index, content=content)
            except Exception as e:
                return Response(index=task.index, content=f"Error: {str(e)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(_generate_task, task)
                futures.append(future)

            for future in futures:
                yield future.result()
