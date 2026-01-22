#!/usr/bin/env python3
"""
Quick verification script to test if the mock server works.
This is a minimal test to verify the setup before running full tests.
"""

import requests
import json

def test_server():
    """Quick test to verify mock server is working"""
    base_url = "http://localhost:8001/v1"

    print("Testing Mock VLLM Server")
    print("=" * 50)

    # Test 1: Get models
    print("\n1. Testing GET /v1/models...")
    try:
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()
        data = response.json()
        print(f"   ✓ Status: {response.status_code}")
        print(f"   ✓ Model: {data['data'][0]['id']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: Post completion
    print("\n2. Testing POST /v1/completions...")
    try:
        payload = {
            "model": "mock-model",
            "prompt": "What is 2 + 2?",
            "temperature": 0.7,
            "max_tokens": 100
        }
        response = requests.post(
            f"{base_url}/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()
        print(f"   ✓ Status: {response.status_code}")
        print(f"   ✓ Completion: {data['choices'][0]['text']}")
        # Note: 'reasoning' is not part of standard vLLM completions API
        # print(f"   ✓ Reasoning: {data['choices'][0].get('reasoning', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("✓ Mock server is working correctly!")
    return True

if __name__ == '__main__':
    import sys
    success = test_server()
    sys.exit(0 if success else 1)
