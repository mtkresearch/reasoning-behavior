#!/usr/bin/env python3
"""
Mock VLLM server for testing text completion API.

This mock server simulates a VLLM server's API endpoints for testing purposes.
It implements the minimal required endpoints:
- GET /v1/models - Returns a list of available models
- POST /v1/completions - Returns text completion results
"""

from flask import Flask, request, jsonify
import time
import re

app = Flask(__name__)

# Mock model configuration
MOCK_MODEL_NAME = "mock-model"


@app.route('/v1/models', methods=['GET'])
def list_models():
    """Return mock model list"""
    return jsonify({
        'data': [
            {
                'id': MOCK_MODEL_NAME,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'mock-organization'
            }
        ],
        'object': 'list'
    })


@app.route('/v1/completions', methods=['POST'])
def completions():
    """
    Handle text completion requests.

    This mock endpoint simulates vLLM's OpenAI-compatible completions API.
    Standard vLLM completions API returns:
    - choices[].text: the completion text
    - choices[].finish_reason: stop/length/etc

    Note: 'reasoning' field is NOT part of standard vLLM/OpenAI completions API.
    It's an extension used by some providers like OpenRouter for specific models.
    For standard vLLM testing, we only return 'text'.
    """
    data = request.json
    prompt = data.get('prompt', '')
    temperature = data.get('temperature', 1.0)
    max_tokens = data.get('max_tokens', 100)

    # Simulate different responses based on prompt content
    if "2 + 2" in prompt or "2+2" in prompt:
        completion_text = "4"
    elif "capital of France" in prompt.lower():
        completion_text = "Paris"
    else:
        # Generic response
        completion_text = "This is a mock completion response."

    # Simulate processing time
    time.sleep(0.1)

    # Standard vLLM/OpenAI completions API response format
    response = {
        'id': f'cmpl-mock-{int(time.time())}',
        'object': 'text_completion',
        'created': int(time.time()),
        'model': data.get('model', MOCK_MODEL_NAME),
        'choices': [
            {
                'text': completion_text,
                'index': 0,
                'logprobs': None,
                'finish_reason': 'stop'
            }
        ],
        'usage': {
            'prompt_tokens': len(prompt.split()),
            'completion_tokens': len(completion_text.split()),
            'total_tokens': len(prompt.split()) + len(completion_text.split())
        }
    }

    return jsonify(response)


def run_server(host='127.0.0.1', port=8001):
    """Run the mock server"""
    print(f"Starting mock VLLM server on {host}:{port}")
    print(f"Model endpoint: http://{host}:{port}/v1/models")
    print(f"Completion endpoint: http://{host}:{port}/v1/completions")
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    run_server()
