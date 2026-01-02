# Testing Guide for Step-JEPA Components 🧪

## Quick Testing Commands

### 1. **Interactive Testing Suite**
```bash
python test_components.py
```

This launches an interactive menu where you can:
- Test step extraction from raw data
- Visualize attention masks
- Test token-level boundary detection
- Test full pipeline
- Run all tests at once

### 2. **Verify Full Implementation**
```bash
python verify_implementation.py
```

Runs all automated tests (data format, trainer logic, loss computation).

---

## Component Tests

### Test 1: Step Extraction 📝

**What it tests**: Extraction of reasoning steps from `deepseek_response`

**Quick test**:
```python
python3 << 'EOF'
import json
from prepare_step_data import extract_all_steps

# Load example
with open('../datasets/gsm8k_synthetic_data.jsonl', 'r') as f:
    data = json.loads(f.readline())

# Extract steps
steps = extract_all_steps(data['deepseek_response'])
print(f"Extracted {len(steps)} steps")
for i, step in enumerate(steps[:2], 1):
    print(f"\nStep {i}: {step[:100]}...")
EOF
```

**What to check**:
- ✅ At least 2 steps extracted
- ✅ Each step > 20 chars
- ✅ Steps separated by `\n\n`
- ✅ No trivial steps (e.g., just "</think>")

---

### Test 2: Attention Masking 🎭

**What it tests**: Custom attention mask that isolates Step 2

**Visual test** (already run above):
```python
python3 -c "
import torch
seq_len = 20
step1_end = 5
step2_end = 10
predictor_pos = step1_end + 1
step2_start = predictor_pos + 1

# Build mask
mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
mask[step2_start:step2_end+1, :step2_start] = float('-inf')

# Check isolation
isolated = torch.all(torch.isinf(mask[step2_start:step2_end+1, :step2_start]))
print(f'Step 2 isolated: {isolated}')

# Show what Step 2 can see
for pos in range(step2_start, step2_end+1):
    can_see = [j for j in range(seq_len) if not torch.isinf(mask[pos, j])]
    print(f'Position {pos} can attend to: {can_see}')
"
```

**What to check**:
- ✅ Step 2 tokens **ONLY** see themselves
- ✅ Step 2 tokens see positions 7-10 (if step2_start=7, step2_end=10)
- ✅ Step 2 **cannot** see positions 0-6 (system, user, step1, predictor)
- ✅ Step 3+ has normal causal attention (can see everything **including** Step 2)

**Expected output**:
```
Step 2 isolated: True
Position 7 can attend to: [7]
Position 8 can attend to: [7, 8]
Position 9 can attend to: [7, 8, 9]
Position 10 can attend to: [7, 8, 9, 10]
```

---

### Test 3: Boundary Detection 🔍

**What it tests**: Finding `\n\n` separators at the token level

**Test with different separators**:
```python
python3 << 'EOF'
# Simulate tokenizer (simple char-level for demo)
text = "Step 1 content\n\nStep 2 content\n\nStep 3 content"
print("Text:", repr(text))

# Find separator positions
sep = "\n\n"
positions = []
idx = 0
while True:
    idx = text.find(sep, idx)
    if idx == -1:
        break
    positions.append(idx)
    idx += len(sep)

print(f"\nFound {len(positions)} separators at positions: {positions}")

if len(positions) >= 2:
    step1_end = positions[0]
    step2_end = positions[1]
    print(f"\n✅ Step 1 ends at char {step1_end}")
    print(f"✅ Step 2 ends at char {step2_end}")
    print(f"✅ Predictor would go at char {step1_end + 2}")
EOF
```

---

### Test 4: Full Pipeline 🔄

**What it tests**: End-to-end from raw data to training format

**Quick test**:
```bash
# Check one example
head -1 gsm8k_step_jepa.jsonl | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Messages:', len(d['messages']))
print('Roles:', [m['role'] for m in d['messages']])
print('Steps:', d['total_steps'])
print('Separators:', d['messages'][2]['content'].count('\n\n'))
print('Format: ✅ PASS' if len(d['messages']) == 3 else '❌ FAIL')
"
```

---

## Understanding the Attention Mask

### Visual Explanation

```
Sequence positions:
[0] [1] [2] [3] [4] [5]  [6]        [7] [8] [9] [10]  [11] [12] ...
 ^System   ^User  ^Step1  ^Predictor  ^---Step 2---^   ^--Step 3+--^

Attention rules:
┌────────────────────┬─────────────┬──────────────┐
│ Token Position     │ Can Attend  │ Why          │
├────────────────────┼─────────────┼──────────────┤
│ 0-5 (Sys/Q/S1)     │ 0 to self   │ Normal causal│
│ 6 (Predictor)      │ 0-6         │ Normal causal│
│ 7-10 (Step 2)      │ 7-10 only   │ ISOLATED!    │
│ 11+ (Step 3+)      │ 0 to self   │ Normal causal│
└────────────────────┴─────────────┴──────────────┘
```

### Why This Matters

**Step 2 isolation ensures**:
- Step 2 embedding doesn't "cheat" by seeing the question
- Forces model to learn step continuation patterns
- JEPA loss aligns "what comes after Step 1" with "Step 2 content"

---

## Common Issues & Solutions

### Issue 1: Step extraction returns < 2 steps
**Cause**: Example doesn't have enough `\n\n` separators
**Check**:
```bash
head -1 ../datasets/gsm8k_synthetic_data.jsonl | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Separators:', d['deepseek_response'].count('\n\n'))
"
```
**Solution**: Normal - some examples skipped during preprocessing

### Issue 2: Attention mask not isolating Step 2
**Test**:
```python
# Check if Step 2 can see position 0
mask[step2_start, 0]  # Should be -inf
```
**Solution**: Verify `step2_start` and boundary positions are correct

### Issue 3: Tokenizer can't find separators
**Cause**: Different tokenization of `\n\n`
**Test**:
```python
tokenizer.encode("\n\n", add_special_tokens=False)
# Should return consistent token IDs
```
**Solution**: Use the same tokenizer as training

---

## Interactive Menu Guide

When you run `python test_components.py`, you get:

```
1. Test step extraction from raw data
   → Shows steps extracted from deepseek_response
   → Good for understanding data preprocessing

2. Visualize attention mask
   → ASCII visualization of attention pattern
   → See Step 2 isolation clearly

3. Test token-level boundary detection
   → Requires tokenizer (may need HF auth)
   → Shows how \n\n is tokenized

4. Test full pipeline (processed data)
   → Verifies gsm8k_step_jepa.jsonl format
   → Quick sanity check

5. Run all tests
   → Runs tests 1-4 sequentially
   → Comprehensive check
```

---

## Expected Test Results

### ✅ All tests should show:

1. **Step Extraction**
   - 6,862 examples processed
   - Each has ≥ 2 steps
   - Separator count matches

2. **Attention Mask**
   - Step 2 isolated: True
   - Each Step 2 token only sees itself + previous Step 2 tokens
   - Visual shows 'X' blocking in Step 2 rows, columns 0-6

3. **Boundary Detection**
   - Finds ≥ 2 separator positions
   - Positions make sense (not at start/end)

4. **Full Pipeline**
   - Format: 3 messages (system, user, assistant)
   - Roles correct
   - Separators present

---

## Quick Diagnostics

### One-liner checks:

```bash
# Check data format
head -1 gsm8k_step_jepa.jsonl | python3 -m json.tool | grep -E "role|total_steps"

# Check separator count
head -1 gsm8k_step_jepa.jsonl | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['messages'][2]['content'].count('\n\n'))"

# Test mask isolation
python3 -c "import torch; m=torch.triu(torch.ones(10,10)*float('-inf'),1); m[7:10,:7]=float('-inf'); print('Isolated:', torch.all(torch.isinf(m[7:10,:7])))"

# Count total examples
wc -l gsm8k_step_jepa.jsonl
```

---

## Advanced Testing

### Test with Your Own Text

```python
python3 << 'EOF'
from prepare_step_data import extract_all_steps

# Your custom response
custom_response = """
First, I need to understand the problem.

Second, I'll break it down into parts.

Third, I'll solve each part.

Finally, I'll combine the results.
"""

steps = extract_all_steps(custom_response)
print(f"Extracted {len(steps)} steps:")
for i, s in enumerate(steps, 1):
    print(f"{i}. {s[:50]}...")
EOF
```

### Test Different Attention Patterns

```python
# Test if different step sizes work
python3 -c "
from test_components import visualize_attention_mask
visualize_attention_mask(seq_len=30, step1_end=8, step2_end=20)
"
```

---

## Debugging Checklist

Before training, verify:
- [ ] `python verify_implementation.py` → All pass ✅
- [ ] Step extraction works on your data
- [ ] Attention mask shows Step 2 isolation
- [ ] Processed data has correct format
- [ ] At least 2 separators per example

---

## Next Steps

1. **Run tests**: `python test_components.py`
2. **Verify all**: `python verify_implementation.py`
3. **If all pass**: `bash train.sh` 🚀

---

**Happy Testing! 🧪✨**

