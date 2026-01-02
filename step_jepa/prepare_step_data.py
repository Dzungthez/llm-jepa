"""
Prepare step-wise JEPA training data from gsm8k_synthetic_data.jsonl

NEW ARCHITECTURE:
- Question in user message
- ALL steps in ONE assistant message (Step1\n\nStep2\n\nStep3...)
- <|predictor|> token inserted after Step1
- Custom attention: Step2 is isolated during JEPA forward pass
- Embeddings extracted at: predictor token (View1) and Step2 end (View2)
"""

import json
import re
import argparse
from pathlib import Path


def extract_all_steps(deepseek_response):
    """
    Extract ALL steps from deepseek_response, keeping them in order.
    
    Steps are separated by \n\n and we look for patterns like:
    - **Step 1:** ...
    - **Step 2:** ...
    - Thinking paragraphs separated by \n\n
    
    Returns: list of steps (strings), or empty list if can't extract
    """
    # Split by double newline first
    parts = deepseek_response.split('\n\n')
    
    steps = []
    current_step = []
    
    # Pattern to detect step markers
    step_pattern = re.compile(r'^\*\*Step\s+\d+:?\*\*|^\d+\.|^Step\s+\d+:', re.IGNORECASE)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Skip think tags
        if part.startswith('</think>') or part.startswith('<think>'):
            continue
            
        # Check if this part starts with a step marker
        if step_pattern.match(part):
            # Save previous step if it exists and has content
            if current_step:
                step_text = '\n\n'.join(current_step)
                # Filter out trivial steps (too short)
                if len(step_text) > 20:
                    steps.append(step_text)
            current_step = [part]
        else:
            # This is a continuation of current step or new thinking text
            if current_step:
                current_step.append(part)
            elif len(part) > 30:  # Substantial thinking text
                steps.append(part)
    
    # Don't forget the last step
    if current_step:
        step_text = '\n\n'.join(current_step)
        if len(step_text) > 20:
            steps.append(step_text)
    
    return steps


def create_step_jepa_training_example(question, answer, ground_truth, deepseek_response):
    """
    Create training example for Step-JEPA with proper structure:
    
    Format:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Step1\\n\\nStep2\\n\\nStep3..."}
      ]
    }
    
    The trainer will:
    1. Insert <|predictor|> after Step1's \\n\\n
    2. Do 2 forward passes:
       - Pass 1: Normal NTP loss
       - Pass 2: Custom attention (Step2 isolated), extract embeddings
    3. JEPA loss aligns: embedding at predictor token <-> embedding at Step2 end
    
    Returns: training example dict or None if can't extract enough steps
    """
    steps = extract_all_steps(deepseek_response)
    
    # We need at least 2 steps for JEPA
    if len(steps) < 2:
        return None
    
    # Concatenate all steps with \n\n separator
    full_response = '\n\n'.join(steps)
    
    # Create the training example
    training_example = {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "total_steps": len(steps),
        "messages": [
            {
                "role": "system",
                "content": "Please solve the problem step by step (separate steps with double newlines), but keep it short and put your final answer (do not include any other text or units) within \\boxed{}."
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": full_response  # All steps in one message
            }
        ]
    }
    
    return training_example


def process_dataset(input_file, output_file, max_examples=None):
    """
    Process the input JSONL file and create step-wise JEPA training data.
    
    For each problem:
    - Extract ALL steps from deepseek_response
    - Create ONE training example with:
      * User: Question
      * Assistant: All steps concatenated (Step1\n\nStep2\n\nStep3...)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Processing {input_file}...")
    print("Extracting ALL steps from each problem's deepseek_response...")
    print("Format: Question → Step1\\n\\nStep2\\n\\nStep3...")
    
    all_examples = []
    skipped = 0
    processed = 0
    
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if max_examples and processed >= max_examples:
                break
                
            try:
                data = json.loads(line.strip())
                
                question = data.get('question', '')
                answer = data.get('answer', '')
                ground_truth = data.get('ground_truth', '')
                deepseek_response = data.get('deepseek_response', '')
                
                # Create Step-JEPA training example
                example = create_step_jepa_training_example(
                    question, answer, ground_truth, deepseek_response
                )
                
                if example:
                    all_examples.append(example)
                    processed += 1
                else:
                    skipped += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"\nStatistics:")
    print(f"  Total input problems: {line_num}")
    print(f"  Successfully processed: {processed}")
    print(f"  Skipped (< 2 steps or error): {skipped}")
    print(f"  Success rate: {processed/line_num*100:.1f}%")
    
    # Write output
    print(f"\nWriting to {output_file}...")
    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✓ Done! Created {len(all_examples)} Step-JEPA training examples")
    
    # Show a sample
    if all_examples:
        print("\n" + "="*80)
        print("Sample Step-JEPA training example:")
        print("="*80)
        sample = all_examples[0]
        print(f"Question: {sample['messages'][1]['content'][:100]}...")
        print(f"Total steps: {sample['total_steps']}")
        print(f"\nAssistant response (first 300 chars):")
        print(f"  {sample['messages'][2]['content'][:300]}...")
        print("\n" + "="*80)
        print("\nNOTE: Custom trainer will:")
        print("  1. Insert <|predictor|> after Step1")
        print("  2. Extract embedding at predictor token (View 1)")
        print("  3. Extract embedding at Step2 end (View 2)")  
        print("  4. Align View1 and View2 with JEPA loss")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Prepare Step-JEPA training data from gsm8k_synthetic_data.jsonl
        
        This extracts the first TWO steps from each problem's deepseek_response
        and creates training pairs compatible with finetune.py:
        - View 1 (User): First step
        - View 2 (Assistant): Second step
        
        The JEPA loss will align the embeddings of these two views.
        """
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="../datasets/gsm8k_synthetic_data.jsonl",
        help="Input JSONL file with deepseek_response field"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./gsm8k_step_jepa.jsonl",
        help="Output JSONL file with Step-JEPA training data"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    process_dataset(args.input_file, args.output_file, args.max_examples)


if __name__ == "__main__":
    main()

