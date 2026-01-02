#!/usr/bin/env python3
"""
Quick test to verify Step 3+ can see Step 2
Run this directly: python quick_test.py
"""

import torch

print("\n" + "="*80)
print("QUICK TEST: Step 3+ Visibility Check")
print("="*80)

# Configuration
seq_len = 15
step1_end = 4
step2_end = 8
predictor_pos = step1_end + 1
step2_start = predictor_pos + 1

print(f"\nSetup:")
print(f"  Step 1: positions 0-{step1_end}")
print(f"  Predictor: position {predictor_pos}")
print(f"  Step 2: positions {step2_start}-{step2_end} (ISOLATED)")
print(f"  Step 3+: positions {step2_end+1}+")

# Build mask
mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
mask[step2_start:step2_end+1, :step2_start] = float('-inf')

print(f"\n{'='*80}")
print("TEST RESULTS:")
print("="*80)

# Test Step 2 isolation
step2_first_pos = step2_start
step2_can_see = [j for j in range(seq_len) if not torch.isinf(mask[step2_first_pos, j])]
print(f"\n1. Step 2 Isolation:")
print(f"   Position {step2_first_pos} (first Step 2 token) can see: {step2_can_see}")
if step2_can_see == [step2_first_pos]:
    print("   ✅ CORRECT: Can only see itself")
else:
    print(f"   ❌ WRONG: Should only see [{step2_first_pos}]")

# Test Step 3+ visibility
step3_first_pos = step2_end + 1
step3_can_see = [j for j in range(seq_len) if not torch.isinf(mask[step3_first_pos, j])]
print(f"\n2. Step 3+ Visibility:")
print(f"   Position {step3_first_pos} (first Step 3 token) can see: {step3_can_see}")

# Check if it can see Step 2
step2_positions = list(range(step2_start, step2_end+1))
can_see_step2 = all(pos in step3_can_see for pos in step2_positions)

if can_see_step2:
    print(f"   ✅ CORRECT: Can see Step 2 (positions {step2_positions})")
    print("   ✅ This means Step 3+ has NORMAL CAUSAL attention")
else:
    print(f"   ❌ WRONG: Cannot see Step 2 (positions {step2_positions})")
    print("   ❌ This would incorrectly block Step 2 from Step 3+")

# Summary
print(f"\n{'='*80}")
print("SUMMARY:")
print("="*80)

both_correct = (step2_can_see == [step2_first_pos]) and can_see_step2

if both_correct:
    print("✅ ALL TESTS PASSED!")
    print("   - Step 2 is properly isolated")
    print("   - Step 3+ can see everything (including Step 2)")
    print("\n🎉 Implementation is CORRECT!")
else:
    print("❌ TESTS FAILED!")
    print("   Check the attention masking logic")

print("="*80 + "\n")

