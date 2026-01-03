# Evaluation Guide for Step-JEPA

This guide explains how to evaluate your trained Step-JEPA model on the GSM8K test set.

---

## Quick Start

### Simple Evaluation

```bash
bash evaluate.sh
```

This will:
- Load the model from `./checkpoints_adapted`
- Evaluate on `../datasets/gsm8k_test.jsonl` (1,319 examples)
- Save results to `./evaluation_results.jsonl`
- Display accuracy metrics

---

## Detailed Usage

### Using the Python Script

```bash
python evaluate_step_jepa.py \
  --model_path ./checkpoints_adapted \
  --test_file ../datasets/gsm8k_test.jsonl \
  --output_file ./evaluation_results.jsonl \
  --max_new_tokens 512 \
  --temperature 0.0
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to trained model checkpoint |
| `--test_file` | `../datasets/gsm8k_test.jsonl` | Path to test data |
| `--output_file` | `./evaluation_results.jsonl` | Where to save results |
| `--max_new_tokens` | 512 | Maximum tokens to generate per example |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `--max_examples` | None | Limit number of examples (for testing) |
| `--use_test_prompts` | False | Use original test prompts (default: use training prompt) |
| `--use_lora` | True | Whether model uses LoRA |
| `--no_lora` | False | Disable LoRA (use full model) |
| `--device` | cuda | Device to use (cuda/cpu) |

**⚠️ Important**: By default, the evaluation uses the **training system prompt** for consistency:
```
"Please solve the problem step by step (separate steps with double newlines), 
but keep it short and put your final answer (do not include any other text or 
units) within \boxed{}."
```

The original test file has a different prompt (`"Answer the math question, show steps."`). 
Using the training prompt ensures the model sees the same instructions it was trained on.

---

## Output Files

### 1. Detailed Results (`evaluation_results.jsonl`)

Each line is a JSON object with:

```json
{
  "index": 0,
  "question": "Janet's ducks lay 16 eggs per day...",
  "ground_truth": "Janet sells 16 - 3 - 4 = 9...\n#### 18",
  "generated_response": "First, calculate eggs remaining...\n\\boxed{18}",
  "gt_answer": "18",
  "gen_answer": "18",
  "correct": true,
  "accuracy_so_far": 100.0
}
```

**Fields:**
- `index`: Example index
- `question`: The math problem
- `ground_truth`: Full GSM8K answer with reasoning
- `generated_response`: Model's generated response
- `gt_answer`: Extracted ground truth answer
- `gen_answer`: Extracted generated answer
- `correct`: Whether the answer is correct
- `accuracy_so_far`: Running accuracy

### 2. Summary (`evaluation_results_summary.json`)

```json
{
  "model_path": "./checkpoints_adapted",
  "test_file": "../datasets/gsm8k_test.jsonl",
  "total_examples": 1319,
  "correct": 856,
  "accuracy": 0.649,
  "max_new_tokens": 512,
  "temperature": 0.0
}
```

---

## Answer Extraction Formats

The evaluation script handles multiple answer formats:

### 1. Boxed Format (from DeepSeek system prompt)

```
The answer is \boxed{42}.
```

Extracted as: `42`

### 2. Hash Format (GSM8K standard)

```
Therefore, the answer is 42.
#### 42
```

Extracted as: `42`

### 3. Fallback (last number in text)

If neither format is found, extracts the last number in the response.

---

## Examples

### Quick Test on First 100 Examples

```bash
python evaluate_step_jepa.py \
  --model_path ./checkpoints_adapted \
  --test_file ../datasets/gsm8k_test.jsonl \
  --max_examples 100 \
  --output_file ./quick_eval.jsonl
```

### Evaluate with Sampling

```bash
python evaluate_step_jepa.py \
  --model_path ./checkpoints_adapted \
  --temperature 0.7 \
  --output_file ./eval_sampling.jsonl
```

### Evaluate Full Fine-Tuned Model (no LoRA)

```bash
python evaluate_step_jepa.py \
  --model_path ./checkpoints_full \
  --no_lora \
  --output_file ./eval_full_model.jsonl
```

### Using Original Test Prompts

If you want to use the original test system prompts instead:

```bash
python evaluate_step_jepa.py \
  --model_path ./checkpoints_adapted \
  --use_test_prompts \
  --output_file ./eval_original_prompts.jsonl
```

**Note**: This may give different results because the model was trained with a different prompt.

---

## Analyzing Results

### Using Python

```python
import json

# Load results
results = []
with open('evaluation_results.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))

# Calculate accuracy
correct = sum(1 for r in results if r['correct'])
total = len(results)
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")

# Find errors
errors = [r for r in results if not r['correct']]
print(f"\nFound {len(errors)} errors")

# Analyze first error
if errors:
    err = errors[0]
    print(f"\nQuestion: {err['question']}")
    print(f"Ground truth: {err['gt_answer']}")
    print(f"Generated: {err['gen_answer']}")
    print(f"Full response: {err['generated_response']}")
```

### Using Command Line

```bash
# Count correct answers
grep -c '"correct": true' evaluation_results.jsonl

# Count total examples
wc -l evaluation_results.jsonl

# View summary
cat evaluation_results_summary.json
```

---

## Comparing Multiple Models

Evaluate different checkpoints and compare:

```bash
# Baseline (no JEPA)
python evaluate_step_jepa.py \
  --model_path ./checkpoints_baseline \
  --output_file ./eval_baseline.jsonl

# Step-JEPA K=1
python evaluate_step_jepa.py \
  --model_path ./checkpoints_k1 \
  --output_file ./eval_k1.jsonl

# Step-JEPA K=2
python evaluate_step_jepa.py \
  --model_path ./checkpoints_k2 \
  --output_file ./eval_k2.jsonl

# Step-JEPA K=3
python evaluate_step_jepa.py \
  --model_path ./checkpoints_k3 \
  --output_file ./eval_k3.jsonl
```

Then compare summaries:

```python
import json

models = ['baseline', 'k1', 'k2', 'k3']
for model in models:
    with open(f'eval_{model}_summary.json') as f:
        summary = json.load(f)
        print(f"{model:15s}: {summary['accuracy']*100:.2f}%")
```

---

## Expected Performance

Based on similar work and the GSM8K benchmark:

| Model Type | Expected Accuracy |
|------------|-------------------|
| Base model (no fine-tuning) | 20-40% |
| Regular fine-tuning | 50-70% |
| Step-JEPA | 55-75% (potentially better reasoning) |

**Note**: Actual performance depends on:
- Base model size and quality
- Training hyperparameters (λ, K, etc.)
- Number of training epochs
- Training data quality

---

## Troubleshooting

### Error: Model directory not found

```
❌ Error: Model directory not found: ./checkpoints_adapted
```

**Solution**: Train a model first using `train_adapted.sh` or specify correct path.

### Error: Out of memory

```
torch.cuda.OutOfMemoryError
```

**Solutions**:
1. Reduce batch size (evaluation uses batch size 1 by default)
2. Reduce `max_new_tokens`
3. Use CPU: `--device cpu`
4. Use a smaller base model

### Error: Test file not found

```
❌ Error: Test file not found: ../datasets/gsm8k_test.jsonl
```

**Solution**: Download the test file or adjust the path.

### Answer extraction fails

If the model generates answers in unexpected formats:

1. Check generated responses in `evaluation_results.jsonl`
2. Modify `extract_answer_from_generated()` in `evaluate_step_jepa.py`
3. Add custom extraction logic for your model's format

---

## Advanced: Custom Evaluation Metrics

You can extend the evaluation script to compute additional metrics:

```python
# Add to evaluate_step_jepa.py

def compute_metrics(results):
    """Compute additional metrics"""
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    # Basic accuracy
    accuracy = correct_count / total_count
    
    # Average response length
    avg_length = sum(len(r['generated_response']) 
                     for r in results) / total_count
    
    # Error analysis
    errors_by_type = {}
    for r in results:
        if not r['correct']:
            # Categorize errors
            if r['gen_answer'] is None:
                error_type = 'no_answer_extracted'
            elif r['gt_answer'] is None:
                error_type = 'bad_ground_truth'
            else:
                error_type = 'wrong_answer'
            
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
    
    return {
        'accuracy': accuracy,
        'avg_response_length': avg_length,
        'errors_by_type': errors_by_type
    }
```

---

## Integration with Training

### Evaluate During Training

Add evaluation to your training script:

```bash
# Train
bash train_adapted.sh

# Evaluate
bash evaluate.sh

# Compare
cat evaluation_results_summary.json
```

### Automated Evaluation Loop

```bash
#!/bin/bash

for K in 1 2 3; do
  for lbd in 0.1 0.5 2.0; do
    echo "Training K=$K, λ=$lbd"
    
    python finetune_step_jepa_adapted.py \
      --step_jepa --predictors $K --lbd $lbd \
      --output_dir ./outputs/K${K}_lbd${lbd}
    
    echo "Evaluating K=$K, λ=$lbd"
    
    python evaluate_step_jepa.py \
      --model_path ./outputs/K${K}_lbd${lbd} \
      --output_file ./eval_K${K}_lbd${lbd}.jsonl
  done
done

# Aggregate results
python analyze_all_evals.py
```

---

## Tips for Better Evaluation

1. **Use greedy decoding** (`temperature=0.0`) for consistent results
2. **Evaluate multiple checkpoints** to find the best one
3. **Inspect errors** manually to understand failure modes
4. **Compare with baseline** (regular fine-tuning without JEPA)
5. **Save all outputs** for later analysis

---

## Next Steps

After evaluation:

1. **Analyze errors**: Why did the model fail?
2. **Compare with baseline**: Does Step-JEPA improve reasoning?
3. **Ablation studies**: Test different K and λ values
4. **Error analysis**: Categorize failure modes
5. **Iterate**: Adjust training based on evaluation insights

---

**Ready to evaluate!** 🎯

Run `bash evaluate.sh` to get started!

