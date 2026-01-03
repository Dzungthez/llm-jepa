#!/usr/bin/env python3
"""
Evaluate Step-JEPA trained model on GSM8K test set.

Usage:
    python evaluate_step_jepa.py \
        --model_path ./checkpoints_adapted \
        --test_file ../datasets/gsm8k_test.jsonl \
        --output_file ./evaluation_results.jsonl
"""

import argparse
import json
import re
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format (used in DeepSeek system prompt)"""
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).strip()
        return normalize_answer(answer)
    return None


def extract_hash_answer(text):
    """Extract answer from #### format (GSM8K standard format)"""
    pattern = r'\n#### (.+)$'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).strip()
        return normalize_answer(answer)
    return None


def extract_final_number(text):
    """Try to extract the last number from the text as a fallback"""
    # Look for numbers at the end of text
    numbers = re.findall(r'[-+]?(?:\d*\.*\d+)', text)
    if numbers:
        return normalize_answer(numbers[-1])
    return None


def normalize_answer(answer):
    """Normalize answer for comparison"""
    if answer is None:
        return None
    
    # Remove common text patterns
    answer = answer.replace('$', '').replace(',', '').strip()
    
    # Try to convert to number and normalize
    try:
        num = float(answer)
        # If it's a whole number, return as int
        if num.is_integer():
            return str(int(num))
        else:
            # Round to reasonable precision
            return f"{num:.10g}"
    except (ValueError, TypeError):
        # If not a number, return cleaned string
        return answer.strip()


def extract_answer_from_generated(generated_text):
    """Extract answer from generated text - try multiple formats"""
    # Try boxed format first (from DeepSeek system prompt)
    answer = extract_boxed_answer(generated_text)
    if answer is not None:
        return answer
    
    # Try #### format (GSM8K standard)
    answer = extract_hash_answer(generated_text)
    if answer is not None:
        return answer
    
    # Fallback: try to extract last number
    answer = extract_final_number(generated_text)
    return answer


def eval_gsm8k(generated, ground_truth):
    """
    Evaluate GSM8K answer.
    
    Args:
        generated: Generated response text
        ground_truth: Ground truth in GSM8K format (with ####)
    
    Returns:
        (is_correct, gt_answer, gen_answer)
    """
    # Extract ground truth answer
    gt_answer = extract_hash_answer(ground_truth)
    
    # Extract generated answer
    gen_answer = extract_answer_from_generated(generated)
    
    # Compare
    is_correct = (gt_answer is not None and 
                  gen_answer is not None and 
                  gt_answer == gen_answer)
    
    return is_correct, gt_answer, gen_answer


def load_model(model_path, use_lora=True, device="cuda"):
    """Load the trained model"""
    model_path = Path(model_path)
    
    print(f"Loading model from: {model_path}")
    
    # Check if this is a LoRA checkpoint or full model
    if use_lora and (model_path / "adapter_config.json").exists():
        # Load LoRA model
        print("  Detected LoRA checkpoint")
        
        # Read adapter config to get base model
        with open(model_path / "adapter_config.json") as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
        
        if base_model_name is None:
            raise ValueError("Could not find base_model_name_or_path in adapter_config.json")
        
        print(f"  Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"  Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
        
    else:
        # Load full fine-tuned model
        print("  Loading full model")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model.eval()
    print(f"  Model loaded successfully on {device}")
    
    return model, tokenizer


def load_test_data(test_file):
    """Load test dataset"""
    print(f"Loading test data from: {test_file}")
    
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            test_data.append(example)
    
    print(f"  Loaded {len(test_data)} test examples")
    return test_data


def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.0):
    """Generate response for a given prompt"""
    # Format the conversation
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (skip the input)
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text.strip()


def evaluate_model(model, tokenizer, test_data, output_file, 
                   max_new_tokens=512, temperature=0.0, max_examples=None,
                   use_training_system_prompt=True):
    """Evaluate model on test data"""
    
    # System prompt used during training (from prepare_step_data.py)
    TRAINING_SYSTEM_PROMPT = "Please solve the problem step by step (separate steps with double newlines), but keep it short and put your final answer (do not include any other text or units) within \\boxed{}."
    
    if max_examples is not None:
        test_data = test_data[:max_examples]
        print(f"Evaluating on first {max_examples} examples")
    
    results = []
    correct_count = 0
    total_count = 0
    
    print(f"\nEvaluating {len(test_data)} examples...")
    if use_training_system_prompt:
        print(f"✓ Using training system prompt for consistency")
    else:
        print(f"⚠️  Using original test system prompts (may differ from training)")
    print("=" * 80)
    
    with open(output_file, 'w') as f:
        for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                # Get messages from test example
                messages = example["messages"]
                
                # Extract ground truth from the last assistant message
                ground_truth = messages[-1]["content"]
                
                # Prepare input messages (system + user only)
                if use_training_system_prompt:
                    # Replace system prompt with training prompt
                    input_messages = [
                        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                        messages[1]  # User message
                    ]
                else:
                    # Use original test prompts
                    input_messages = messages[:-1]  # Exclude assistant's answer
                
                # Generate response
                generated_response = generate_response(
                    model, tokenizer, input_messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                # Evaluate
                is_correct, gt_answer, gen_answer = eval_gsm8k(generated_response, ground_truth)
                
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                # Compute accuracy so far
                accuracy = correct_count / total_count * 100
                
                # Create result entry
                result = {
                    "index": idx,
                    "question": messages[1]["content"],  # User message
                    "ground_truth": ground_truth,
                    "generated_response": generated_response,
                    "gt_answer": gt_answer,
                    "gen_answer": gen_answer,
                    "correct": is_correct,
                    "accuracy_so_far": accuracy
                }
                results.append(result)
                
                # Write to file
                f.write(json.dumps(result) + '\n')
                f.flush()
                
                # Print progress every 10 examples
                if (idx + 1) % 10 == 0:
                    print(f"\nAfter {idx + 1} examples: {correct_count}/{total_count} correct ({accuracy:.2f}%)")
                
            except Exception as e:
                print(f"\n❌ Error at index {idx}: {e}")
                result = {
                    "index": idx,
                    "question": example["messages"][1]["content"],
                    "error": str(e),
                    "correct": False
                }
                results.append(result)
                f.write(json.dumps(result) + '\n')
                f.flush()
    
    # Final statistics
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total examples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Accuracy: {correct_count / total_count * 100:.2f}%")
    print(f"Results saved to: {output_file}")
    print("=" * 80)
    
    return results, correct_count, total_count


def main():
    parser = argparse.ArgumentParser(description="Evaluate Step-JEPA model on GSM8K test set")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Whether the model uses LoRA (default: True)")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA (use full fine-tuned model)")
    
    # Data arguments
    parser.add_argument("--test_file", type=str, 
                        default="../datasets/gsm8k_test.jsonl",
                        help="Path to test file")
    parser.add_argument("--output_file", type=str,
                        default="./evaluation_results.jsonl",
                        help="Path to save evaluation results")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate (for testing)")
    parser.add_argument("--use_test_prompts", action="store_true",
                        help="Use original test system prompts instead of training prompt")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Handle LoRA flag
    use_lora = args.use_lora and not args.no_lora
    
    print("=" * 80)
    print("STEP-JEPA MODEL EVALUATION")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output file: {args.output_file}")
    print(f"Use LoRA: {use_lora}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"System prompt: {'Original test' if args.use_test_prompts else 'Training (consistent)'}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load model
    model, tokenizer = load_model(args.model_path, use_lora=use_lora, device=args.device)
    
    # Load test data
    test_data = load_test_data(args.test_file)
    
    # Evaluate
    results, correct_count, total_count = evaluate_model(
        model, tokenizer, test_data, args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_examples=args.max_examples,
        use_training_system_prompt=not args.use_test_prompts
    )
    
    # Save summary
    summary = {
        "model_path": args.model_path,
        "test_file": args.test_file,
        "total_examples": total_count,
        "correct": correct_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    
    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()

