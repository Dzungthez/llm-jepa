"""
Test K predictor tokens insertion for Step-JEPA
"""

import torch
import json


def visualize_step_jepa_with_k_predictors(K=2):
    """Show how K predictor tokens are inserted after Step 1"""
    
    # Sample data
    example = {
        "messages": [
            {"role": "system", "content": "Solve step by step"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "First, identify the numbers.\n\nSecond, add them together.\n\nFinal answer: 4"}
        ]
    }
    
    # Simulate tokenization
    tokens = [
        "[System]", "Solve", "step", "by", "step",
        "[User]", "What", "is", "2+2?",
        "[Assistant]", "First,", "identify", "the", "numbers.", "\n\n",  # Step 1
        # K predictor tokens will be inserted HERE
        "Second,", "add", "them", "together.", "\n\n",  # Step 2
        "Final", "answer:", "4"  # Step 3+
    ]
    
    step1_end = 14  # After "\n\n"
    step2_end = 19  # After second "\n\n"
    
    # Insert K predictor tokens
    predictors = [f"<|predictor_{i+1}|>" for i in range(K)]
    tokens_with_predictors = (
        tokens[:step1_end+1] + 
        predictors + 
        tokens[step1_end+1:]
    )
    
    # Adjust positions
    predictor_start = step1_end + 1
    predictor_end = predictor_start + K - 1
    step2_start = predictor_end + 1
    step2_end_adjusted = step2_end + K
    
    print("=" * 80)
    print(f"STEP-JEPA WITH K={K} PREDICTOR TOKENS")
    print("=" * 80)
    
    # Print sequence
    print("\n📝 Token Sequence:")
    print("-" * 80)
    for i, token in enumerate(tokens_with_predictors):
        marker = ""
        if i == step1_end:
            marker = "  ← Step 1 END"
        elif i == predictor_start:
            marker = "  ← Predictor START"
        elif i == predictor_end:
            marker = f"  ← Predictor END (extract embedding from here)"
        elif i == step2_start:
            marker = "  ← Step 2 START (ISOLATED)"
        elif i == step2_end_adjusted:
            marker = "  ← Step 2 END (extract embedding from here)"
        
        if i in range(predictor_start, predictor_end + 1):
            print(f"  [{i:2d}] {token:20s} 🎯{marker}")
        elif i in range(step2_start, step2_end_adjusted + 1):
            print(f"  [{i:2d}] {token:20s} 🔒{marker}")
        else:
            print(f"  [{i:2d}] {token:20s} {marker}")
    
    # Show attention mask
    print("\n" + "=" * 80)
    print("🔍 Attention Mask Pattern:")
    print("=" * 80)
    
    seq_len = len(tokens_with_predictors)
    
    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # Isolate Step 2
    mask[step2_start:step2_end_adjusted+1, :step2_start] = float('-inf')
    
    # Visualize key regions
    print(f"\n{'':20s}", end="")
    for i in range(seq_len):
        if i == step1_end:
            print("S1E", end=" ")
        elif i in range(predictor_start, predictor_end + 1):
            print(f"P{i-predictor_start+1:1d}", end="  ")
        elif i == step2_start:
            print("S2S", end=" ")
        elif i == step2_end_adjusted:
            print("S2E", end=" ")
        elif i > step2_end_adjusted:
            print("S3", end="  ")
        else:
            print(f"{i:2d}", end="  ")
    print()
    
    for i in range(seq_len):
        # Row label
        if i == step1_end:
            print(f"{'Step1 End':20s}", end="")
        elif i in range(predictor_start, predictor_end + 1):
            print(f"Predictor {i-predictor_start+1:1d}{' ':11s}", end="")
        elif i in range(step2_start, step2_end_adjusted + 1):
            print(f"{'Step2 (ISOLATED)':20s}", end="")
        elif i > step2_end_adjusted:
            print(f"{'Step3+':20s}", end="")
        else:
            print(f"{i:20d}", end="")
        
        # Show mask values
        for j in range(seq_len):
            val = mask[i, j].item()
            if val == float('-inf'):
                print("⛔ ", end=" ")
            else:
                print("✓  ", end=" ")
        print()
    
    print("\n" + "=" * 80)
    print("📊 Key Embeddings:")
    print("=" * 80)
    print(f"  View 1 (Step 1 → Step 2): Extract from position {predictor_end} (last predictor)")
    print(f"  View 2 (Step 2):          Extract from position {step2_end_adjusted} (Step 2 end)")
    print(f"\n  JEPA Loss = align(embedding[{predictor_end}], embedding[{step2_end_adjusted}])")
    
    print("\n" + "=" * 80)
    print("✨ Attention Pattern Summary:")
    print("=" * 80)
    print(f"  1. Positions [0:{step1_end}]: System + User + Step 1 (normal causal)")
    print(f"  2. Positions [{predictor_start}:{predictor_end}]: K={K} predictor tokens (normal causal)")
    print(f"  3. Positions [{step2_start}:{step2_end_adjusted}]: Step 2 (ISOLATED - only see themselves)")
    print(f"  4. Positions [{step2_end_adjusted+1}:]: Step 3+ (normal causal, see everything)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n\n")
    visualize_step_jepa_with_k_predictors(K=1)
    
    print("\n\n")
    visualize_step_jepa_with_k_predictors(K=2)
    
    print("\n\n")
    visualize_step_jepa_with_k_predictors(K=3)

