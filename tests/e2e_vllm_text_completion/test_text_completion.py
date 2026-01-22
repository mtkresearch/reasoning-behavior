#!/usr/bin/env python3
"""
End-to-end test for text completion with mock VLLM server.

This test verifies that the LLMClient.complete() method works correctly
with a local VLLM server (using a mock server for testing).

Usage:
    1. Start the mock server: python tests/e2e_vllm_text_completion/mock_vllm_server.py
    2. Run this test: python tests/e2e_vllm_text_completion/test_text_completion.py
"""

import sys
import os
import subprocess
import time
import requests

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm_client import LLMClient, CompletionRequest


def wait_for_server(url, timeout=10):
    """Wait for server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"✓ Server is ready at {url}")
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    return False


def test_complete_basic():
    """Test basic text completion"""
    print("\n" + "="*60)
    print("Test 1: Basic Text Completion")
    print("="*60)

    # Initialize client in local mode
    client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

    # Create a completion request
    request = CompletionRequest(
        question="What is 2 + 2?",
        reasoning="Let me think about this step by step. We need to add two and two.",
        answer_prefix="The answer is ",
        model_type="gpt-oss",  # Will use mock model
        temperature=0.7,
        max_tokens=100,
    )

    print(f"\nQuestion: {request.question}")
    print(f"Reasoning: {request.reasoning}")
    print(f"Answer prefix: {request.answer_prefix}")

    # Call complete
    try:
        result = client.complete(request)
        print(f"\n✓ Completion successful!")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"\n✗ Completion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_different_question():
    """Test with a different question"""
    print("\n" + "="*60)
    print("Test 2: Different Question")
    print("="*60)

    client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

    request = CompletionRequest(
        question="What is the capital of France?",
        reasoning="France is a country in Western Europe. Its capital should be a major city.",
        answer_prefix="The capital is ",
        model_type="gpt-oss",
        temperature=0.5,
        max_tokens=50,
    )

    print(f"\nQuestion: {request.question}")
    print(f"Reasoning: {request.reasoning}")
    print(f"Answer prefix: {request.answer_prefix}")

    try:
        result = client.complete(request)
        print(f"\n✓ Completion successful!")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"\n✗ Completion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_with_local_mode_settings():
    """Test that local mode correctly reads from server"""
    print("\n" + "="*60)
    print("Test 3: Local Mode Settings")
    print("="*60)

    client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

    # Verify that _get_model works correctly
    print("\nTesting _get_model() with local mode...")
    try:
        model = client._get_model("gpt-oss")
        print(f"✓ Model retrieved: {model}")
        assert model == "mock-model", f"Expected 'mock-model', got '{model}'"
    except Exception as e:
        print(f"✗ Failed to get model: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("="*60)
    print("End-to-End Text Completion Test")
    print("="*60)

    # Check if server is running
    server_url = "http://localhost:8001/v1/models"
    print(f"\nChecking server at {server_url}...")

    if not wait_for_server(server_url, timeout=5):
        print("\n✗ Mock server is not running!")
        print("\nPlease start the server first:")
        print("  python tests/e2e_vllm_text_completion/mock_vllm_server.py")
        return False

    # Run tests
    results = []
    results.append(("Basic Completion", test_complete_basic()))
    results.append(("Different Question", test_complete_different_question()))
    results.append(("Local Mode Settings", test_complete_with_local_mode_settings()))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("="*60)
        print("\nYou can now safely use this on a GPU machine with real VLLM server.")
        print("Just change the base_url to point to your VLLM server.")
        return True
    else:
        print("✗ Some tests failed")
        print("="*60)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
