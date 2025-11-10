#!/usr/bin/env python3
"""
Test script for inequality rule in mask_numbers_advance function
Run from project root: python tests/test_mask_inequality.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import mask_numbers_advance

def test_inequalities():
    """Test various inequality patterns"""

    test_cases = [
        # Basic inequalities
        ("n < 5", "n < 5", "Less than"),
        ("n > 10", "n > 10", "Greater than"),
        ("x ≤ 100", "x ≤ 100", "Less than or equal"),
        ("y ≥ 50", "y ≥ 50", "Greater than or equal"),

        # ASCII inequalities
        ("a <= 25", "a <= 25", "ASCII less than or equal"),
        ("b >= 75", "b >= 75", "ASCII greater than or equal"),

        # With spaces
        ("n  <  5", "n  <  5", "With extra spaces"),
        ("x  >=  100", "x  >=  100", "With extra spaces (>=)"),

        # Double inequalities
        ("1 < x < 10", "1 < x < 10", "Double inequality"),
        ("0 ≤ n ≤ 100", "0 ≤ n ≤ 100", "Double inequality with ≤"),
        ("5 <= x <= 15", "5 <= x <= 15", "Double inequality ASCII"),

        # Mixed with other content
        ("If n > 5, then", "If n > 5, then", "Inequality in sentence"),
        ("where 1 ≤ i ≤ n", "where 1 ≤ i ≤ n", "Inequality in description"),

        # Should still mask (no inequality nearby)
        ("n + 5", "n + █", "Addition - should mask"),
        ("10 * 2", "██ * █", "Multiplication - should mask"),
        ("x = 100", "x = ███", "Equality - should mask"),

        # Edge cases
        ("f(n) where n > 0", "f(n) where n > 0", "Function with inequality constraint"),
        ("for i in range(1, 10) where i < 5", "for i in range(█, ██) where i < 5",
         "Mixed: range should mask, inequality should not"),
    ]

    print("Testing inequality masking rules")
    print("=" * 80)

    passed = 0
    failed = 0

    for input_text, expected, description in test_cases:
        result = mask_numbers_advance(input_text, answer=None)
        status = "✓ PASS" if result == expected else "✗ FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status} - {description}")
        if result != expected:
            print(f"  Input:    '{input_text}'")
            print(f"  Expected: '{expected}'")
            print(f"  Got:      '{result}'")
        print()

    print("=" * 80)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"Success rate: {passed/len(test_cases)*100:.1f}%")

    return failed == 0


if __name__ == '__main__':
    success = test_inequalities()
    sys.exit(0 if success else 1)
