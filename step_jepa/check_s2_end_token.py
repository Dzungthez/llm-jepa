#!/usr/bin/env python3
"""
Check what token is at s2_end position
"""

from transformers import AutoTokenizer

# Test with a common model (DeepSeek)
model_name = "deepseek-ai/deepseek-math-7b-base"
print(f"Loading tokenizer: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except:
    # Fallback to a common model
    print("Trying fallback model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"✓ Loaded: {model_name}\n")

# Test what \n\n tokenizes to
sep_text = "\n\n"
sep_tokens = tokenizer.encode(sep_text, add_special_tokens=False)

print("=" * 80)
print("SEPARATOR TOKENIZATION")
print("=" * 80)
print(f"Text: {repr(sep_text)}")
print(f"Token IDs: {sep_tokens}")
print(f"Number of tokens: {len(sep_tokens)}")
print()

# Decode each token
print("Individual tokens:")
for i, token_id in enumerate(sep_tokens):
    decoded = tokenizer.decode([token_id])
    print(f"  Token {i}: ID={token_id}, decoded={repr(decoded)}")

print()
print("=" * 80)
print("WHAT IS h[s2_end]?")
print("=" * 80)

# Simulate a step sequence
example_text = "First step.\n\nSecond step.\n\nThird step."
tokens = tokenizer.encode(example_text, add_special_tokens=False)

print(f"\nExample text: {repr(example_text)}")
print(f"Total tokens: {len(tokens)}")
print()

# Find separators
positions = []
for i in range(len(tokens) - len(sep_tokens) + 1):
    if tokens[i:i+len(sep_tokens)] == sep_tokens:
        end_pos = i + len(sep_tokens) - 1
        positions.append(end_pos)
        print(f"Found separator at position {i}, ends at position {end_pos}")
        
        # Show what token is at the end position
        if len(sep_tokens) == 1:
            token_at_end = tokens[end_pos]
            decoded_at_end = tokenizer.decode([token_at_end])
            print(f"  → Token at position {end_pos}: ID={token_at_end}, decoded={repr(decoded_at_end)}")
        else:
            print(f"  → Separator spans {len(sep_tokens)} tokens")
            print(f"  → Last separator token at position {end_pos}:")
            token_at_end = tokens[end_pos]
            decoded_at_end = tokenizer.decode([token_at_end])
            print(f"     ID={token_at_end}, decoded={repr(decoded_at_end)}")

if len(positions) >= 2:
    step1_end = positions[0]
    step2_end = positions[1]
    
    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(f"step1_end = {step1_end}")
    print(f"step2_end = {step2_end}")
    print()
    
    if len(sep_tokens) == 1:
        print(f"✓ h[s2_end] = h[{step2_end}] corresponds to the token: {repr(tokenizer.decode([tokens[step2_end]]))}")
        print(f"✓ This IS the '\\n\\n' token (single token)")
    else:
        print(f"✓ h[s2_end] = h[{step2_end}] corresponds to the LAST token of '\\n\\n'")
        print(f"✓ Token value: {repr(tokenizer.decode([tokens[step2_end]]))}")
        print(f"✓ The separator '\\n\\n' spans {len(sep_tokens)} tokens")
        print(f"✓ Positions: {step2_end - len(sep_tokens) + 1} to {step2_end}")
    
    print()
    print("🔍 Key Insight:")
    print(f"   - In code: step2_end = i + len(sep_tokens) - 1")
    print(f"   - This gives the position of the LAST token in the separator")
    print(f"   - So h[s2_end] is the hidden state at the END of '\\n\\n'")
    
    if len(sep_tokens) > 1:
        print(f"   - Since '\\n\\n' tokenizes to {len(sep_tokens)} tokens,")
        print(f"     h[s2_end] is the last of those tokens")
        print(f"   - It represents the separator, but may not literally be '\\n\\n' as a single token")
    else:
        print(f"   - Since '\\n\\n' is a single token, h[s2_end] IS that token")

