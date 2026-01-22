#!/usr/bin/env python3
"""
Example usage of LLMClient with local vLLM server for text completion.

This demonstrates how to use the text completion API in a real scenario.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm_client import LLMClient, CompletionRequest


def example_basic_completion():
    """Basic text completion example"""
    print("Example 1: Basic Text Completion")
    print("=" * 60)

    # Initialize client
    # For Mac testing: use mock server at http://localhost:8001/v1
    # For GPU machine: change to your vLLM server URL
    client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

    # Create completion request
    request = CompletionRequest(
        question="What is 2 + 2?",
        reasoning="Let me think about this step by step. We need to add two and two.",
        answer_prefix="The answer is ",
        model_type="gpt-oss",
        temperature=0.7,
        max_tokens=100,
    )

    # Execute completion
    result = client.complete(request)

    print(f"\nQuestion: {request.question}")
    print(f"Result: {result}")
    print()


def example_with_different_model_types():
    """Example showing different model types"""
    print("Example 2: Different Model Types")
    print("=" * 60)

    client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

    # Different model types use different template formats
    model_configs = [
        {
            "model_type": "gpt-oss",
            "question": "What is the capital of France?",
            "reasoning": "France is a Western European country.",
        },
        # Uncomment when using real vLLM server with these models
        # {
        #     "model_type": "deepseek",
        #     "question": "Explain quantum computing",
        #     "reasoning": "Quantum computing uses quantum mechanics principles.",
        # },
        # {
        #     "model_type": "olmo",
        #     "question": "What is machine learning?",
        #     "reasoning": "Machine learning is a subset of AI.",
        # },
    ]

    for config in model_configs:
        request = CompletionRequest(
            question=config["question"],
            reasoning=config["reasoning"],
            answer_prefix="",
            model_type=config["model_type"],
            temperature=0.7,
            max_tokens=100,
        )

        result = client.complete(request)
        print(f"\nModel: {config['model_type']}")
        print(f"Question: {config['question']}")
        print(f"Result: {result}")
        print("-" * 60)


def example_batch_processing():
    """Example of batch processing with concurrent execution"""
    print("Example 3: Batch Processing")
    print("=" * 60)

    from llm_client import Task

    client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

    # Create multiple tasks
    questions = [
        "What is 1 + 1?",
        "What is 2 + 2?",
        "What is 3 + 3?",
    ]

    tasks = []
    for i, question in enumerate(questions):
        request = CompletionRequest(
            question=question,
            reasoning="Let me calculate this.",
            answer_prefix="The answer is ",
            model_type="gpt-oss",
            temperature=0.7,
            max_tokens=50,
        )
        task = Task(index=i, request=request, metadata={"question": question})
        tasks.append(task)

    # Process concurrently
    print(f"\nProcessing {len(tasks)} tasks concurrently...")
    for completed_task in client.complete_concurrent(tasks, max_workers=3):
        if completed_task.response.success:
            print(f"Task {completed_task.index}: {completed_task.response.content}")
        else:
            print(f"Task {completed_task.index} failed: {completed_task.response.err_message}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("LLMClient Text Completion Examples")
    print("=" * 60)
    print()

    try:
        example_basic_completion()
        example_with_different_model_types()
        example_batch_processing()

        print("=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Start your vLLM server on GPU machine")
        print("2. Update base_url to point to your server")
        print("3. Choose appropriate model_type for your model")
        print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure the mock server is running:")
        print("  python tests/e2e_vllm_text_completion/mock_vllm_server.py")
        return False

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
