"""
Interactive Testing for Step Extraction and Attention Masking

This script lets you test and visualize the core components of Step-JEPA:
1. Step extraction from deepseek_response
2. Attention mask creation
3. Token-level inspection
"""

import json
import torch
import sys
from pathlib import Path


# ============================================================================
# TEST 1: Step Extraction
# ============================================================================

def test_step_extraction(example_index=0):
    """Test step extraction on a specific example from the raw data"""
    
    print("="*80)
    print("TEST 1: Step Extraction from deepseek_response")
    print("="*80)
    
    # Load raw data
    raw_data_path = Path("../datasets/gsm8k_synthetic_data.jsonl")
    with open(raw_data_path, 'r') as f:
        lines = f.readlines()
        if example_index >= len(lines):
            print(f"❌ Example {example_index} out of range (max: {len(lines)-1})")
            return
        
        example = json.loads(lines[example_index])
    
    print(f"\n📝 Example {example_index}:")
    print(f"Question: {example['question'][:100]}...")
    print(f"\n{'─'*80}")
    print("Raw deepseek_response:")
    print("─"*80)
    print(example['deepseek_response'][:500])
    print("..." if len(example['deepseek_response']) > 500 else "")
    
    # Extract steps using our logic
    sys.path.insert(0, '.')
    from prepare_step_data import extract_all_steps
    
    steps = extract_all_steps(example['deepseek_response'])
    
    print(f"\n{'─'*80}")
    print(f"Extracted Steps: {len(steps)} total")
    print("─"*80)
    
    for i, step in enumerate(steps, 1):
        print(f"\n[Step {i}] ({len(step)} chars)")
        print("─"*40)
        print(step[:200] + ("..." if len(step) > 200 else ""))
    
    # Show separator detection
    print(f"\n{'─'*80}")
    print("Step Separators (\\n\\n) in response:")
    print("─"*80)
    sep_count = example['deepseek_response'].count('\n\n')
    print(f"Found {sep_count} double-newline separators")
    
    # Show what will be in training data
    concatenated = '\n\n'.join(steps)
    print(f"\n{'─'*80}")
    print("Final Training Format (first 300 chars):")
    print("─"*80)
    print(concatenated[:300] + "...")
    
    return steps


# ============================================================================
# TEST 2: Attention Mask Visualization
# ============================================================================

def visualize_attention_mask(seq_len=20, step1_end=5, step2_end=10):
    """
    Visualize the custom attention mask with Step 2 isolation
    
    Args:
        seq_len: Total sequence length
        step1_end: Position where Step 1 ends
        step2_end: Position where Step 2 ends
    """
    
    print("\n" + "="*80)
    print("TEST 2: Attention Mask Visualization")
    print("="*80)
    
    print(f"\nSequence Configuration:")
    print(f"  Total length: {seq_len}")
    print(f"  Step 1 ends at position: {step1_end}")
    print(f"  Predictor at position: {step1_end + 1}")
    print(f"  Step 2: positions {step1_end + 2} to {step2_end}")
    print(f"  Step 3+: positions {step2_end + 1} onwards")
    
    # Build the mask using Step-JEPA logic
    predictor_pos = step1_end + 1
    step2_start = predictor_pos + 1
    
    # Start with causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # Isolate Step 2 - can't see anything before it
    mask[step2_start:step2_end+1, :step2_start] = float('-inf')
    
    # Step 3+ has normal causal attention (can see everything including Step 2)
    # No additional masking needed
    
    # Visualize
    print(f"\n{'─'*80}")
    print("Attention Mask (rows=query, cols=key)")
    print("Legend: '·' = can attend (0), 'X' = blocked (-inf)")
    print("─"*80)
    
    # Column headers
    print("     ", end="")
    for j in range(seq_len):
        if j == step1_end:
            print("|S1", end="")
        elif j == predictor_pos:
            print("|PR", end="")
        elif j == step2_start:
            print("|S2", end="")
        elif j == step2_end + 1 and j < seq_len:
            print("|S3", end="")
        else:
            print(f"|{j:2d}", end="")
    print("|")
    
    print("     " + "─"*(seq_len*3 + 1))
    
    for i in range(seq_len):
        # Row label
        if i == step1_end:
            print("S1 ", end="")
        elif i == predictor_pos:
            print("PR ", end="")
        elif i == step2_start:
            print("S2 ", end="")
        elif i == step2_end + 1 and i < seq_len:
            print("S3 ", end="")
        else:
            print(f"{i:2d} ", end="")
        
        print("|", end="")
        for j in range(seq_len):
            if torch.isinf(mask[i, j]):
                # Color-code different regions
                if i >= step2_start and i <= step2_end and j < step2_start:
                    print(" X ", end="")  # Step 2 can't see before
                else:
                    print(" x ", end="")  # Normal causal blocking
            else:
                print(" · ", end="")
        print("|")
    
    # Analysis
    print(f"\n{'─'*80}")
    print("Attention Analysis:")
    print("─"*80)
    
    # Check Step 2 isolation
    step2_can_see_before = not torch.all(torch.isinf(mask[step2_start:step2_end+1, :step2_start]))
    
    if step2_can_see_before:
        print("❌ ISSUE: Step 2 can see tokens before it!")
    else:
        print("✅ Step 2 is properly isolated (can't see positions 0-{})".format(step2_start-1))
    
    # What can each region see?
    print(f"\nWhat each region can attend to:")
    print(f"  Positions 0-{step1_end} (System/User/Step1): Normal causal")
    print(f"  Position {predictor_pos} (Predictor): Can see 0-{predictor_pos}")
    print(f"  Positions {step2_start}-{step2_end} (Step2): Can ONLY see themselves (ISOLATED)")
    if step2_end + 1 < seq_len:
        print(f"  Positions {step2_end+1}+ (Step3+): Normal causal (can see EVERYTHING including Step2)")
    
    return mask


# ============================================================================
# TEST 3: Token-Level Boundary Detection
# ============================================================================

def test_boundary_detection_with_tokenizer(example_text=None):
    """
    Test how step boundaries are detected at the token level
    This requires downloading a tokenizer (can be skipped if gated)
    """
    
    print("\n" + "="*80)
    print("TEST 3: Token-Level Boundary Detection")
    print("="*80)
    
    try:
        from transformers import AutoTokenizer
        print("\nLoading tokenizer (this may require HuggingFace auth)...")
        
        # Try a non-gated model first
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
            print("✅ Using Qwen tokenizer")
        except:
            print("⚠️  Qwen not available, trying GPT2...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("✅ Using GPT2 tokenizer (demo purposes)")
        
    except Exception as e:
        print(f"❌ Could not load tokenizer: {e}")
        print("   Skipping token-level test (not critical for understanding)")
        return
    
    # Use example text or default
    if example_text is None:
        example_text = "First, I calculate 2+2=4.\n\nNext, I multiply by 3 to get 12.\n\nFinally, the answer is 12."
    
    print(f"\nTest text:")
    print("─"*80)
    print(repr(example_text))
    
    # Tokenize
    tokens = tokenizer.encode(example_text)
    print(f"\n{'─'*80}")
    print(f"Tokenization: {len(tokens)} tokens")
    print("─"*80)
    
    # Show tokens
    for i, token_id in enumerate(tokens[:30]):  # Show first 30
        token_str = tokenizer.decode([token_id])
        print(f"  {i:3d}: {token_id:6d} → {repr(token_str)}")
    if len(tokens) > 30:
        print(f"  ... ({len(tokens) - 30} more tokens)")
    
    # Find separator
    sep_text = "\n\n"
    sep_tokens = tokenizer.encode(sep_text, add_special_tokens=False)
    
    print(f"\n{'─'*80}")
    print(f"Separator '\\n\\n' tokenizes to: {sep_tokens}")
    print(f"  → {[tokenizer.decode([t]) for t in sep_tokens]}")
    print("─"*80)
    
    # Find occurrences
    print("\nSearching for separator occurrences...")
    positions = []
    for i in range(len(tokens) - len(sep_tokens) + 1):
        if tokens[i:i+len(sep_tokens)] == sep_tokens:
            positions.append(i + len(sep_tokens) - 1)
            print(f"  Found at position {i} (ends at {i + len(sep_tokens) - 1})")
    
    if len(positions) >= 2:
        print(f"\n✅ Found {len(positions)} separators")
        print(f"   Step 1 would end at: {positions[0]}")
        print(f"   Step 2 would end at: {positions[1]}")
        print(f"   Predictor would be inserted at: {positions[0] + 1}")
    else:
        print(f"\n⚠️  Only found {len(positions)} separators (need at least 2)")


# ============================================================================
# TEST 4: Full Pipeline Test
# ============================================================================

def test_full_pipeline(example_index=0):
    """
    Test the full pipeline: extract steps → format → check separators
    """
    
    print("\n" + "="*80)
    print("TEST 4: Full Pipeline Test")
    print("="*80)
    
    # Load processed data
    processed_data_path = Path("gsm8k_step_jepa.jsonl")
    if not processed_data_path.exists():
        print("❌ gsm8k_step_jepa.jsonl not found. Run prepare_step_data.py first.")
        return
    
    with open(processed_data_path, 'r') as f:
        lines = f.readlines()
        if example_index >= len(lines):
            print(f"❌ Example {example_index} out of range")
            return
        
        example = json.loads(lines[example_index])
    
    print(f"\n📝 Processed Example {example_index}:")
    print(f"Total steps: {example.get('total_steps', 'N/A')}")
    print(f"Messages: {len(example['messages'])}")
    
    # Check format
    assert len(example['messages']) == 3, "Should have 3 messages"
    assert example['messages'][0]['role'] == 'system'
    assert example['messages'][1]['role'] == 'user'
    assert example['messages'][2]['role'] == 'assistant'
    
    print("✅ Format correct: system, user, assistant")
    
    # Check separators
    assistant_content = example['messages'][2]['content']
    sep_count = assistant_content.count('\n\n')
    
    print(f"\nAssistant message:")
    print(f"  Length: {len(assistant_content)} chars")
    print(f"  Separators (\\n\\n): {sep_count}")
    print(f"  Total steps: {example.get('total_steps', sep_count + 1)}")
    
    if sep_count >= 1:
        print("✅ Has at least 1 separator (enough for Step 1 and Step 2)")
    else:
        print("❌ No separators found - this example won't work for Step-JEPA")
    
    # Show first 2 steps
    parts = assistant_content.split('\n\n')
    print(f"\n{'─'*80}")
    print("Step 1 (first part):")
    print("─"*80)
    print(parts[0][:200] + ("..." if len(parts[0]) > 200 else ""))
    
    if len(parts) > 1:
        print(f"\n{'─'*80}")
        print("Step 2 (second part):")
        print("─"*80)
        print(parts[1][:200] + ("..." if len(parts[1]) > 200 else ""))


# ============================================================================
# Interactive Menu
# ============================================================================

def main():
    """Interactive test menu"""
    
    print("\n" + "🔬 "*20)
    print("Step-JEPA Component Testing")
    print("🔬 "*20 + "\n")
    
    while True:
        print("\n" + "="*80)
        print("Choose a test:")
        print("="*80)
        print("1. Test step extraction from raw data")
        print("2. Visualize attention mask")
        print("3. Test token-level boundary detection")
        print("4. Test full pipeline (processed data)")
        print("5. Run all tests")
        print("0. Exit")
        print("="*80)
        
        choice = input("\nEnter choice (0-5): ").strip()
        
        if choice == '0':
            print("\n👋 Exiting. Happy training!")
            break
        
        elif choice == '1':
            idx = input("Example index (default 0): ").strip()
            idx = int(idx) if idx else 0
            test_step_extraction(idx)
        
        elif choice == '2':
            print("\nDefault: seq_len=20, step1_end=5, step2_end=10")
            custom = input("Use custom values? (y/n): ").strip().lower()
            
            if custom == 'y':
                seq_len = int(input("Sequence length: "))
                step1_end = int(input("Step 1 end position: "))
                step2_end = int(input("Step 2 end position: "))
                visualize_attention_mask(seq_len, step1_end, step2_end)
            else:
                visualize_attention_mask()
        
        elif choice == '3':
            use_custom = input("Use custom text? (y/n): ").strip().lower()
            if use_custom == 'y':
                print("Enter text (use \\n\\n for separators):")
                text = input().replace('\\n', '\n')
                test_boundary_detection_with_tokenizer(text)
            else:
                test_boundary_detection_with_tokenizer()
        
        elif choice == '4':
            idx = input("Example index (default 0): ").strip()
            idx = int(idx) if idx else 0
            test_full_pipeline(idx)
        
        elif choice == '5':
            print("\n🚀 Running all tests...\n")
            test_step_extraction(0)
            visualize_attention_mask()
            test_boundary_detection_with_tokenizer()
            test_full_pipeline(0)
            print("\n✅ All tests complete!")
        
        else:
            print("❌ Invalid choice")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

