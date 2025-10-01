from openai import OpenAI


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
