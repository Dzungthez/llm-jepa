# Full Options Guide for Step-JEPA Adapted

This document explains all available options in the adapted version, matching the original `finetune.py`.

## 🎛️ Training Modes

### 1. **Regular Mode** (No JEPA)
Standard language model training without any JEPA loss.

```bash
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --regular \
  --lora
```

### 2. **Original JEPA Mode** (Text ↔ Code)
Uses the original LLM-JEPA approach: align "Text" and "Code" views with K predictor tokens.

```bash
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --predictors 2 \
  --lbd 0.5 \
  --lora
```

**Note:** Without `--step_jepa`, this uses the original Text/Code view alignment from the paper.

### 3. **Step-JEPA Mode** (Step 1 ↔ Step 2)
Our novel approach: align consecutive reasoning steps with K predictor tokens after Step 1.

```bash
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --step_jepa \
  --predictors 2 \
  --lbd 0.1 \
  --lora
```

**Note:** With `--step_jepa`, the K predictors are inserted **after Step 1** (not after user message).

---

## 📊 Key Parameters

### JEPA Loss Weight (`--lbd`)
Controls the strength of JEPA regularization.

```bash
--lbd 0.1    # Light regularization (default)
--lbd 0.5    # Medium regularization
--lbd 2.0    # Strong regularization
```

**Recommendation:** Start with `0.1`, increase if you want stronger step alignment.

### LM Loss Weight (`--gamma`)
Controls the language modeling loss weight.

```bash
--gamma 1.0  # Default (equal weight)
--gamma 0.5  # Less emphasis on LM
--gamma 2.0  # More emphasis on LM
```

**Recommendation:** Keep at `1.0` unless you have specific reasons.

### Predictor Tokens (`--predictors`)
Number of special `<|predictor_N|>` tokens inserted after the first view.

**For Original JEPA (Text ↔ Code):**
```bash
--predictors 0  # No predictor tokens (direct alignment)
--predictors 1  # One predictor token after user message
--predictors 2  # Two predictor tokens (recommended, paper default)
```

**For Step-JEPA (Step 1 ↔ Step 2):**
```bash
--step_jepa --predictors 1  # One predictor after Step 1
--step_jepa --predictors 2  # Two predictors after Step 1 (recommended)
--step_jepa --predictors 3  # Three predictors (more capacity)
```

**Key Points:**
- Predictors are inserted **after Step 1** in Step-JEPA mode
- Embedding is extracted from the **last predictor token** (`<|predictor_K|>`)
- More predictors = more capacity but higher memory cost
- Paper default: K=2

**From LLM-JEPA paper:** K=2 provides good balance between capacity and efficiency.

### Last Token Offset (`--last_token`)
Which token to use for embedding extraction (counted from end).

```bash
--last_token -1  # Last token (default for most models)
--last_token -2  # Second to last (for Gemma, Llama)
--last_token -3  # Third to last (for Qwen)
--last_token -4  # Fourth to last (for OpenELM)
```

**Model-specific recommendations:**
- Llama: `-2`
- Gemma: `-2`
- Qwen: `-3`
- OpenELM: `-4`
- OLMo, DeepSeek: `-1`

### Random Seed (`--seed` or `--finetune_seed`)
For reproducibility across experiments.

```bash
--finetune_seed 42   # Default
--finetune_seed 82   # Alternative seed (from run.sh)
```

**From `run.sh`:** Original experiments used seeds: `82, 23, 37, 84, 4`

---

## 🔧 Loss Types

### Cosine Similarity (Default)
```bash
# No flag needed - this is the default
```
Loss: `1 - cosine_similarity(view1, view2)`

### L2 Norm
```bash
--jepa_l2
```
Loss: `||view1 - view2||_2`

### Mean Squared Error
```bash
--jepa_mse
```
Loss: `mean((view1 - view2)^2)`

### InfoNCE
```bash
--infonce
```
Contrastive loss with temperature.

---

## 💾 LoRA Options

```bash
--lora                 # Enable LoRA fine-tuning
--lora_rank 16         # LoRA rank (default: 16)
--lora_rank 32         # Higher rank = more parameters
```

**Recommendation:** Use LoRA for faster training and lower memory usage.

---

## 🎓 Complete Examples

### Example 1: Step-JEPA (Recommended)
```bash
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --output_dir ./checkpoints_step_jepa \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --step_jepa \
  --lbd 0.1 \
  --gamma 1.0 \
  --last_token -2 \
  --lora \
  --lora_rank 16 \
  --finetune_seed 42
```

### Example 2: Original JEPA with Predictors
```bash
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --output_dir ./checkpoints_jepa \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --predictors 2 \
  --lbd 0.5 \
  --gamma 1.0 \
  --last_token -2 \
  --lora \
  --finetune_seed 42
```

### Example 3: Regular Training (Baseline)
```bash
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --output_dir ./checkpoints_regular \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --regular \
  --lora \
  --finetune_seed 42
```

### Example 4: Hyperparameter Sweep (Like run.sh)
```bash
#!/bin/bash
for seed in 82 23 37 84 4; do
  for lbd in 0.1 0.5 2.0; do
    python finetune_step_jepa_adapted.py \
      --train_file ./gsm8k_step_jepa.jsonl \
      --output_dir ./checkpoints_sweep_seed${seed}_lbd${lbd} \
      --step_jepa \
      --lbd $lbd \
      --finetune_seed $seed \
      --lora
  done
done
```

### Example 5: Different Loss Types
```bash
# L2 loss
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --step_jepa --lbd 0.1 --jepa_l2 --lora

# MSE loss
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --step_jepa --lbd 0.1 --jepa_mse --lora

# InfoNCE loss
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --step_jepa --lbd 0.1 --infonce --lora
```

---

## 🔍 Debugging Options

```bash
--debug 0   # No debug output
--debug 1   # Basic debug info
--debug 5   # Show losses during training (default)
--debug 7   # Show attention masks and inputs
```

---

## 📋 Quick Reference Table

| Parameter | Default | Purpose | Used In |
|-----------|---------|---------|---------|
| `--regular` | False | Disable JEPA | All modes |
| `--step_jepa` | False | Enable Step-JEPA | Step-JEPA |
| `--predictors` | 0 | Number of predictor tokens | Original JEPA |
| `--lbd` | 0.1 | JEPA loss weight (λ) | JEPA modes |
| `--gamma` | 1.0 | LM loss weight (γ) | JEPA modes |
| `--last_token` | -1 | Token for embedding | JEPA modes |
| `--jepa_l2` | False | Use L2 loss | JEPA modes |
| `--jepa_mse` | False | Use MSE loss | JEPA modes |
| `--infonce` | False | Use InfoNCE loss | JEPA modes |
| `--jepa_ratio` | -1.0 | Random JEPA dropout | JEPA modes |
| `--lora` | False | Enable LoRA | All modes |
| `--lora_rank` | 16 | LoRA rank | When LoRA enabled |
| `--finetune_seed` | 42 | Random seed | All modes |
| `--debug` | 0 → 5 | Debug level | All modes |

---

## 🎯 Usage Scripts

We provide three training scripts:

### 1. **`train_adapted.sh`** (Simple)
Basic Step-JEPA training with default options.

```bash
bash train_adapted.sh
```

### 2. **`train_adapted_full.sh`** (Comprehensive)
Shows all options with easy configuration section.

```bash
bash train_adapted_full.sh
```

### 3. **Custom Script**
Create your own based on your needs!

---

## 💡 Tips

1. **Start simple**: Use `train_adapted.sh` first
2. **Compare modes**: Run regular, JEPA, and Step-JEPA to compare
3. **Tune λ**: Try `0.1, 0.5, 1.0, 2.0` to find best JEPA strength
4. **Multiple seeds**: Use 3-5 seeds for robust results
5. **Loss type**: Start with cosine (default), try L2 if needed
6. **Model-specific**: Adjust `last_token` based on your model

---

## 🔗 Relationship to Original finetune.py

This adapted version supports **all** the key options from `finetune.py`:

✅ `--regular` - Regular training without JEPA  
✅ `--predictors` - Original JEPA with predictor tokens  
✅ `--lbd` - JEPA loss weight  
✅ `--gamma` - LM loss weight  
✅ `--last_token` - Embedding extraction position  
✅ `--jepa_l2` - L2 norm loss  
✅ `--jepa_mse` - MSE loss  
✅ `--infonce` - InfoNCE loss  
✅ `--jepa_ratio` - Random JEPA dropout  
✅ `--finetune_seed` - Random seed  
✅ `--lora` - LoRA fine-tuning  

**Plus** our new option:
✨ `--step_jepa` - Step-wise JEPA mode

---

## 📊 Comparison Matrix

| Feature | `train.sh` (Standalone) | `train_adapted.sh` (Simple) | `train_adapted_full.sh` (Full) |
|---------|-------------------------|----------------------------|-------------------------------|
| Step-JEPA | ✅ | ✅ | ✅ |
| Original JEPA | ❌ | ❌ | ✅ |
| Regular mode | ❌ | ❌ | ✅ |
| Predictor tokens | ❌ | ❌ | ✅ |
| Multiple loss types | ✅ | ❌ | ✅ |
| Custom seeds | ✅ | ❌ | ✅ |
| Last token offset | ❌ | ❌ | ✅ |
| JEPA ratio | ❌ | ❌ | ✅ |

**Recommendation:** Use `train_adapted_full.sh` for maximum flexibility!

---

**Ready to experiment with all options!** 🚀

