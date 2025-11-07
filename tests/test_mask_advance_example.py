#!/usr/bin/env python3
"""
Real-world example test for mask_numbers_all_advance function
Run from project root: python tests/test_mask_advance_example.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mask_numbers_experiment import mask_numbers_all_advance

# A realistic mathematical reasoning example
# Let's say the answer is 2400
ANSWER = "2400"

reasoning_example = """
Let's solve this step by step.

Step 1: Define variables
Let x1 be the first number and x2 be the second number.

Step 2: Set up equations
From the problem, we have:
x1 + x2 = 100
x1 - x2 = 20

Step 3: Solve the system
Adding the two equations:
2*x1 = 120
x1 = 60

Substituting back:
60 + x2 = 100
x2 = 40

Step 4: Verify
Check: 60 + 40 = 100 ✓
Check: 60 - 40 = 20 ✓

Step 5: Calculate the product
The product is 60 * 40 = 2400

For a 3x3 matrix with entries a_ij where i,j ∈ {1,2,3},
the determinant can be computed using the formula:
det(A) = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)

If we have f(x) = x^2 + 3x + 5, then:
f(10) = 10^2 + 3*10 + 5 = 100 + 30 + 5 = 135

Therefore, the answer is 2400.
"""

def main():
    print("=" * 80)
    print("ORIGINAL REASONING:")
    print("=" * 80)
    print(reasoning_example)

    print("\n" + "=" * 80)
    print(f"MASKED REASONING (all-advance mode with answer={ANSWER}):")
    print("=" * 80)
    masked = mask_numbers_all_advance(reasoning_example, answer=ANSWER)
    print(masked)

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print("\nRULE PRIORITY:")
    print(f"  1. HARD RULE: Answer ({ANSWER}) is ALWAYS masked - HIGHEST PRIORITY")
    print("  2. Number adjacent to letter/underscore → Don't mask")
    print("  3. Number in inequality → Don't mask")
    print("  4. Exception: digit x digit → Force mask")
    print("  5. Other numbers → Mask")

    print("\nWhat should be PRESERVED (not masked):")
    print("  ✓ Variable indices: x1, x2")
    print("  ✓ Subscripts: a_ij, i, j, a11, a22, a33, a23, a32, a12, a21, a31, a13")
    print("  ✓ Coefficients: 2*x1, 3*10, 3x")
    print("  ✓ Ordinals: (none in this example)")
    print("  ✓ Numbers in inequalities: {1,2,3} (in set notation)")

    print("\nWhat should be MASKED:")
    print(f"  ✓ ANSWER {ANSWER}: ALWAYS masked (hard rule)")
    print("  ✓ Computational values: 100, 20, 120, 60, 40, 135, etc.")
    print("  ✓ Exponents: x^2, 10^2")
    print("  ✓ Matrix dimensions: 3x3")
    print("  ✓ Function arguments: f(10)")
    print(f"  ✓ Note: {ANSWER} masked even though it's the final answer")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
