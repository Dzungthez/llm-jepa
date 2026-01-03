# K Predictor Tokens for Step-JEPA

## Overview

Step-JEPA now supports **K predictor tokens** inserted after Step 1, matching the original LLM-JEPA paper's approach where K tokens are placed after the "Text" view to predict the "Code" view.

## Token Sequence Structure

### With K=2 Predictor Tokens

```
[System] [User] [Step1] <|predictor_1|> <|predictor_2|> [Step2] [Step3+]
                   ↑              ↑                           ↑
                   └──────────────┴───────────────────────────┘
                   Extract embed from last predictor (<|predictor_2|>)
                                                       Extract embed from Step2 end
```

### General Pattern (K tokens)

```
Sequence = [System] [User] [Step1] <|predictor_1|> ... <|predictor_K|> [Step2] [Step3+]
                                              ↑                           ↑
                                              └───────JEPA LOSS───────────┘
```

## Attention Masking

The attention mask ensures Step 2 is **isolated** while predictor tokens follow normal causal attention:

| Token Region | Can See |
|--------------|---------|
| System, User, Step 1 | Normal causal (each token sees previous tokens) |
| `<|predictor_1|>` to `<|predictor_K|>` | Normal causal (see System, User, Step 1, and previous predictors) |
| Step 2 tokens | **ISOLATED** - only see themselves |
| Step 3+ tokens | Normal causal (see everything, including Step 2) |

## JEPA Loss Computation

1. **View 1 (Predictor → Step 2)**: Extract embedding from the **last predictor token** (`<|predictor_K|>`)
2. **View 2 (Step 2)**: Extract embedding from the **last token of Step 2** (typically `\n\n`)
3. **JEPA Loss**: Align these two embeddings using cosine similarity, L2, MSE, or InfoNCE

```python
# Pseudocode
predictor_pos = step1_end + K  # Last predictor token
step2_pos = step2_end + K      # Adjusted for inserted predictors

view1_embed = hidden_states[predictor_pos]  # From last predictor
view2_embed = hidden_states[step2_pos]      # From Step 2 end

jepa_loss = 1 - cosine_similarity(view1_embed, view2_embed)
```

## Command-Line Usage

### Basic Step-JEPA with K=2

```bash
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --step_jepa \
  --predictors 2 \
  --lbd 0.1
```

### Hyperparameter Sweep

```bash
for K in 1 2 3; do
  for lbd in 0.1 0.5 2.0; do
    python finetune_step_jepa_adapted.py \
      --train_file ./gsm8k_step_jepa.jsonl \
      --step_jepa \
      --predictors $K \
      --lbd $lbd \
      --output_dir ./outputs/K${K}_lbd${lbd}
  done
done
```

## Comparison with Original JEPA

| Aspect | Original JEPA | Step-JEPA |
|--------|---------------|-----------|
| **View 1** | User message | Step 1 |
| **View 2** | Assistant response | Step 2 |
| **Predictor location** | After user message | After Step 1 |
| **K predictors** | After user | After Step 1 |
| **View 2 isolation** | Assistant isolated from user | Step 2 isolated from Step 1 |
| **Activation** | `--predictors K` (no --step_jepa) | `--step_jepa --predictors K` |

## Testing

Run the visualization script to see how K predictor tokens are inserted:

```bash
cd /Users/dungnh/coding-papers/llm-jepa/step_jepa
python test_k_predictors.py
```

This will show:
- Token sequences with K=1, 2, 3
- Attention mask patterns
- Embedding extraction positions

## Key Implementation Details

### Token Addition

```python
# Add K predictor tokens to tokenizer
for i in range(K):
    token = f"<|predictor_{i+1}|>"
    tokenizer.add_special_tokens({"additional_special_tokens": [token]})
```

### Sequence Modification

```python
# Insert K predictors after Step 1
new_sequence = (
    original_seq[:step1_end+1] +
    [f"<|predictor_{i+1}|>" for i in range(K)] +
    original_seq[step1_end+1:]
)
```

### Position Adjustment

```python
# Adjust positions for JEPA embedding extraction
predictor_pos = step1_end + K          # Last predictor
step2_end_adjusted = step2_end + K     # Step 2 end (adjusted)
```

## Recommended Values

Based on the original LLM-JEPA paper experiments:

- **K=0**: No predictor tokens (direct alignment, similar to SimCLR)
- **K=1**: Single predictor (lightweight, faster training)
- **K=2**: Two predictors (paper's default, good balance)
- **K=3+**: More capacity but higher memory/compute cost

## Ablation Studies

You can test the impact of K by running:

```bash
bash train_adapted_full.sh
```

And modifying the `PREDICTORS` variable to sweep over different K values.

## Architecture Diagram

```
                    Forward Pass
                    
Input:    [Sys][User][S1] <|p_1|> <|p_2|> [S2] [S3+]
                 ↓       ↓       ↓      ↓    ↓    ↓
Attention: Normal    Normal  Normal  Normal Isolated Normal
                                            (JEPA)
                           ↓                  ↓
Hidden:             h[p_2]            h[S2_end]
                           ↓                  ↓
                           └────JEPA Loss────┘
                           
                     cosine_sim(h[p_2], h[S2_end])
```

## Why K Predictor Tokens?

From the LLM-JEPA paper:
1. **Capacity**: K tokens provide more representational capacity to predict the next view
2. **Flexibility**: The model learns to use K tokens optimally for prediction
3. **Regularization**: Forces the model to compress Step 1 information into K tokens before predicting Step 2
4. **Empirical**: Paper shows K=2 works well across datasets

## Notes

- The predictor tokens are **not** used for next-token prediction (LM loss)
- They only participate in JEPA embedding extraction
- Labels remain unchanged (we don't predict the predictor tokens themselves)
- The attention mask ensures Step 2 remains isolated regardless of K

