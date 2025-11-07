#!/usr/bin/env python3
"""
Test script for mask_numbers_all_advance function
Run from project root: python tests/test_mask_advance.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mask_numbers_experiment import mask_numbers_all_advance

# Test cases
test_cases = [
    # (input, expected_output, description)
    ("A12", "A12", "Variable index - should not mask"),
    ("x1", "x1", "Variable index - should not mask"),
    ("x_1", "x_1", "Subscript - should not mask"),
    ("a_2", "a_2", "Subscript - should not mask"),
    ("3x", "3x", "Coefficient - should not mask"),
    ("5a", "5a", "Coefficient - should not mask"),
    ("1st", "1st", "Ordinal - should not mask"),
    ("2nd", "2nd", "Ordinal - should not mask"),
    ("3rd", "3rd", "Ordinal - should not mask"),
    ("x^2", "x^█", "Exponent - should mask"),
    ("10^3", "██^█", "Power - should mask"),
    ("f(3)", "f(█)", "Function argument - should mask"),
    ("sin(30)", "sin(██)", "Function argument - should mask"),
    ("3x3", "█x█", "Multiplication - should mask (exception rule)"),
    ("2x5", "█x█", "Multiplication - should mask (exception rule)"),
    ("10x10", "██x██", "Multiplication - should mask (exception rule)"),
    ("1 + 2", "█ + █", "Addition - should mask"),
    ("3 * 5", "█ * █", "Multiplication - should mask"),
    ("10 - 7", "██ - █", "Subtraction - should mask"),
    ("1/2", "█/█", "Division - should mask"),
    ("x = 20", "x = ██", "Equation - should mask"),
    ("y > 5", "y > 5", "Inequality - should NOT mask"),
    ("n < 10", "n < 10", "Inequality - should NOT mask"),
    ("1 ≤ x ≤ 10", "1 ≤ x ≤ 10", "Double inequality - should NOT mask"),
    ("a >= 100", "a >= 100", "Inequality >= - should NOT mask"),
    ("42", "██", "Standalone number - should mask"),
    ("100", "███", "Standalone number - should mask"),
    ("AIME 2025", "AIME ████", "Year - should mask"),
    ("3x + 5y = 10", "3x + 5y = ██", "Mixed equation - coefficients kept, result masked"),
    ("x_1 + x_2 = 15", "x_1 + x_2 = ██", "Subscripts kept, result masked"),
    ("Let n = 100", "Let n = ███", "Assignment - should mask"),
    ("a1b2c3", "a1b2c3", "Mixed alphanumeric - should not mask"),
    ("Step 1:", "Step █:", "Step label - should mask (no special rule)"),
    ("2)", "█)", "List item - should mask"),
]

def run_tests():
    print("Testing mask_numbers_all_advance function (without answer parameter)\n")
    print("=" * 80)

    passed = 0
    failed = 0

    for input_text, expected, description in test_cases:
        result = mask_numbers_all_advance(input_text, answer=None)
        status = "✓ PASS" if result == expected else "✗ FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status}")
        print(f"  Description: {description}")
        print(f"  Input:       '{input_text}'")
        print(f"  Expected:    '{expected}'")
        print(f"  Got:         '{result}'")
        if result != expected:
            print(f"  ⚠ Mismatch!")
        print()

    print("=" * 80)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"Success rate: {passed/len(test_cases)*100:.1f}%")

    return passed, failed


def test_answer_hard_rule():
    """Test the hard rule: answer must always be masked"""
    print("\n\n")
    print("Testing HARD RULE: Answer must always be masked\n")
    print("=" * 80)

    # Test cases with answer parameter: (input, answer, expected, description)
    answer_test_cases = [
        ("x42", "42", "x██", "Answer=42 adjacent to letter - MUST mask"),
        ("42x", "42", "██x", "Answer=42 before letter - MUST mask"),
        ("n < 5", "5", "n < █", "Answer=5 in inequality - MUST mask"),
        ("1 ≤ x ≤ 100", "100", "1 ≤ x ≤ ███", "Answer=100 in inequality - MUST mask"),
        ("A100", "100", "A███", "Answer=100 as variable index - MUST mask"),
        ("x_5", "5", "x_█", "Answer=5 as subscript - MUST mask"),
        ("3x", "3", "█x", "Answer=3 as coefficient - MUST mask"),
        ("x^2 + 5x + 10", "10", "x^█ + 5x + ██", "Answer=10 in equation - MUST mask"),
        ("10 + 20 = 30", "30", "██ + ██ = ██", "Answer=30 in equation - MUST mask"),
        ("Answer is 42", "42", "Answer is ██", "Answer=42 in text - MUST mask"),
        # Non-answer numbers should follow normal rules
        ("x5 + 10", "10", "x5 + ██", "Answer=10, but 5 not answer (adjacent to x, not masked)"),
        ("n < 7 and m = 10", "10", "n < 7 and m = ██", "Answer=10, 7 in inequality not masked"),
    ]

    passed = 0
    failed = 0

    for input_text, answer, expected, description in answer_test_cases:
        result = mask_numbers_all_advance(input_text, answer=answer)
        status = "✓ PASS" if result == expected else "✗ FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status}")
        print(f"  Description: {description}")
        print(f"  Input:       '{input_text}'")
        print(f"  Answer:      '{answer}'")
        print(f"  Expected:    '{expected}'")
        print(f"  Got:         '{result}'")
        if result != expected:
            print(f"  ⚠ Mismatch!")
        print()

    print("=" * 80)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(answer_test_cases)} tests")
    print(f"Success rate: {passed/len(answer_test_cases)*100:.1f}%")

    return passed, failed


if __name__ == '__main__':
    p1, f1 = run_tests()
    p2, f2 = test_answer_hard_rule()

    total_passed = p1 + p2
    total_failed = f1 + f2
    total_tests = total_passed + total_failed

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total tests:     {total_tests}")
    print(f"Passed:          {total_passed}")
    print(f"Failed:          {total_failed}")
    print(f"Success rate:    {total_passed/total_tests*100:.1f}%")
    print("=" * 80)
