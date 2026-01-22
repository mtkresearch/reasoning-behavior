#!/usr/bin/env python3
"""
Test real vLLM server deployment.

This script tests a real vLLM server (not mock) to verify:
1. Server is accessible
2. Text completion works correctly
3. Model-specific templates are working

Usage:
    # Test local vLLM server
    python tests/e2e_vllm_text_completion/test_real_vllm.py

    # Test remote vLLM server
    python tests/e2e_vllm_text_completion/test_real_vllm.py --base-url http://gpu-server:8001/v1

    # Test with specific model type
    python tests/e2e_vllm_text_completion/test_real_vllm.py --model-type deepseek

    # Run with custom test question
    python tests/e2e_vllm_text_completion/test_real_vllm.py --question "What is 100 + 200?"
"""

import sys
import os
import argparse
import requests
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm_client import LLMClient, CompletionRequest


def check_server_health(base_url, timeout=10):
    """Check if vLLM server is healthy and accessible"""
    print(f"\n{'='*60}")
    print("Step 1: Server Health Check")
    print(f"{'='*60}")

    models_url = f"{base_url}/models"
    print(f"Checking server at: {models_url}")

    try:
        response = requests.get(models_url, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if not data.get('data') or len(data['data']) == 0:
            print("✗ Server returned no models")
            return False, None

        model_id = data['data'][0]['id']
        print(f"✓ Server is healthy")
        print(f"✓ Available model: {model_id}")
        return True, model_id

    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {base_url}")
        print("\nPlease ensure vLLM server is running:")
        print("  python -m vllm.entrypoints.openai.api_server --model <model-name> --port 8001")
        return False, None
    except requests.exceptions.Timeout:
        print(f"✗ Server timeout after {timeout}s")
        return False, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None


def test_basic_completion(base_url, model_type, question=None, reasoning=None):
    """Test basic text completion with real vLLM server"""
    print(f"\n{'='*60}")
    print("Step 2: Text Completion Test")
    print(f"{'='*60}")

    # Use default test question if not provided
    if question is None:
        question = "What is 2 + 2? Please show your calculation."
    if reasoning is None:
        reasoning = "To solve this, I need to add 2 and 2 together."

    print(f"\nModel type: {model_type}")
    print(f"Question: {question}")
    print(f"Reasoning: {reasoning}")

    # Initialize client
    client = LLMClient(mode="local", base_url=base_url)

    # Create completion request
    request = CompletionRequest(
        question=question,
        reasoning=reasoning,
        answer_prefix="The answer is ",
        model_type=model_type,
        temperature=0.7,
        max_tokens=200,
    )

    # Execute completion
    try:
        print("\nSending request to vLLM server...")
        start_time = time.time()
        result = client.complete(request)
        elapsed = time.time() - start_time

        print(f"\n✓ Completion successful! (took {elapsed:.2f}s)")
        print(f"\nResult: {result}")
        return True, result

    except Exception as e:
        print(f"\n✗ Completion failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_template_formatting(base_url, model_type):
    """Test that model-specific template is being applied correctly"""
    print(f"\n{'='*60}")
    print("Step 3: Template Formatting Test")
    print(f"{'='*60}")

    client = LLMClient(mode="local", base_url=base_url)

    # Enable DEBUG mode to see the formatted prompt
    import llm_client
    original_debug = llm_client.DEBUG
    llm_client.DEBUG = True

    try:
        print(f"\nTesting template formatting for model_type: {model_type}")
        print("(This will show the formatted prompt)")

        request = CompletionRequest(
            question="Test question",
            reasoning="Test reasoning",
            answer_prefix="Test answer: ",
            model_type=model_type,
            temperature=0.7,
            max_tokens=50,
        )

        # This will print the formatted prompt due to DEBUG=True
        result = client.complete(request)

        print(f"\n✓ Template applied successfully")
        return True

    except Exception as e:
        print(f"\n✗ Template formatting failed: {e}")
        return False

    finally:
        # Restore original DEBUG setting
        llm_client.DEBUG = original_debug


def run_comprehensive_test(base_url, model_type, question=None, reasoning=None):
    """Run comprehensive tests on real vLLM server"""
    print("\n" + "="*60)
    print("vLLM Server Comprehensive Test")
    print("="*60)
    print(f"\nBase URL: {base_url}")
    print(f"Model Type: {model_type}")

    results = []

    # Step 1: Health check
    healthy, model_id = check_server_health(base_url)
    results.append(("Server Health Check", healthy))

    if not healthy:
        print("\n" + "="*60)
        print("✗ Tests aborted - server is not healthy")
        print("="*60)
        return False

    # Step 2: Basic completion
    success, result = test_basic_completion(base_url, model_type, question, reasoning)
    results.append(("Basic Completion", success))

    # Step 3: Template formatting (optional, for debugging)
    # Uncomment to enable:
    # template_ok = test_template_formatting(base_url, model_type)
    # results.append(("Template Formatting", template_ok))

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
        print("\nYour vLLM server is working correctly!")
        print("You can now use it in your experiments.")
    else:
        print("✗ Some tests failed")
        print("="*60)
        print("\nPlease check:")
        print("1. vLLM server is running and accessible")
        print("2. Model type matches the deployed model")
        print("3. Server has sufficient resources (GPU memory, etc.)")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Test real vLLM server deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test local vLLM server with default settings
  python test_real_vllm.py

  # Test remote vLLM server
  python test_real_vllm.py --base-url http://192.168.1.100:8001/v1

  # Test with specific model type
  python test_real_vllm.py --model-type deepseek

  # Test with custom question
  python test_real_vllm.py --question "Solve: 123 + 456"
        """
    )

    parser.add_argument(
        '--base-url',
        default='http://localhost:8001/v1',
        help='vLLM server base URL (default: http://localhost:8001/v1)'
    )

    parser.add_argument(
        '--model-type',
        default='gpt-oss',
        choices=['gpt-oss', 'deepseek', 'deepseek-base', 'olmo'],
        help='Model type for template formatting (default: gpt-oss)'
    )

    parser.add_argument(
        '--question',
        help='Custom test question (optional)'
    )

    parser.add_argument(
        '--reasoning',
        help='Custom reasoning text (optional)'
    )

    args = parser.parse_args()

    # Run tests
    success = run_comprehensive_test(
        base_url=args.base_url,
        model_type=args.model_type,
        question=args.question,
        reasoning=args.reasoning
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
