# Step-JEPA Architecture with K Predictor Tokens

## Overview

Step-JEPA extends the LLM-JEPA architecture to consecutive reasoning steps instead of Text/Code views.

---

## Sequence Structure (K=2 example)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOKEN SEQUENCE                                      │
└─────────────────────────────────────────────────────────────────────────────┘

[System: "Solve step by step"]
[User: "What is 2+2?"]
[Step 1: "First, identify the numbers.\n\n"]
<|predictor_1|>
<|predictor_2|>
[Step 2: "Second, add them together.\n\n"]
[Step 3+: "Final answer: 4"]

```

---

## Attention Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ATTENTION MASK PATTERN                                │
└─────────────────────────────────────────────────────────────────────────────┘

Region          │ Can See                               │ Type
────────────────┼───────────────────────────────────────┼─────────────────
System          │ Only itself                           │ Causal
User            │ System, self                          │ Causal
Step 1          │ System, User, self                    │ Causal
<|predictor_1|> │ System, User, Step 1, self            │ Causal
<|predictor_2|> │ System, User, Step 1, predictors      │ Causal
Step 2          │ ONLY Step 2 tokens (ISOLATED)         │ 🔒 ISOLATED
Step 3+         │ Everything (including Step 2)         │ Causal

```

---

## Forward Pass Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FORWARD PASS                                       │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT:
  [Sys][User][S1] <|p_1|> <|p_2|> [S2] [S3+]

   │
   ├──────────────────────────────────────────────────────────────────────────
   │  PASS 1: Language Model (with JEPA mask)
   ├──────────────────────────────────────────────────────────────────────────
   │
   ▼

ATTENTION:
  [Sys]: Normal causal
  [User]: Normal causal
  [S1]: Normal causal
  <|p_1|>: Normal causal (sees Sys, User, S1)
  <|p_2|>: Normal causal (sees Sys, User, S1, p_1)
  [S2]: ISOLATED (only sees S2 tokens)
  [S3+]: Normal causal (sees everything)

   │
   ▼

HIDDEN STATES:
  h_sys, h_user, h_s1, h_p1, h_p2, h_s2, h_s3
                         ↑           ↑
                         │           │
                         │           └─────────────────┐
                         │                             │
   ┌─────────────────────┘                             │
   │                                                    │
   │                                                    │
   ▼                                                    ▼
Extract h[p_2]                                  Extract h[s2_end]
(last predictor)                                (Step 2 last token)
   │                                                    │
   │                                                    │
   └─────────────────────────────────────────────────> │
                        JEPA LOSS                       │
          align(h[p_2], h[s2_end])                     │
                                                        │
   ┌────────────────────────────────────────────────────┘
   │
   │  PASS 1 (continued): Next Token Prediction
   │
   ▼

LM LOSS:
  Cross-entropy on next token prediction
  (using normal labels, ignoring predictor tokens)

   │
   ▼

TOTAL LOSS:
  γ × LM_loss + λ × JEPA_loss

```

---

## JEPA Loss Computation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        JEPA EMBEDDING ALIGNMENT                             │
└─────────────────────────────────────────────────────────────────────────────┘

View 1: Step 1 → Step 2 (predictor embedding)
  ┌───────────────────────────────────────────┐
  │ Embedding = h[last_predictor_position]    │
  │           = h[step1_end + K]               │
  │                                            │
  │ This embedding should PREDICT Step 2      │
  └───────────────────────────────────────────┘

View 2: Step 2 (direct embedding)
  ┌───────────────────────────────────────────┐
  │ Embedding = h[step2_end_position]         │
  │           = h[step2_end + K]  (adjusted)  │
  │                                            │
  │ This embedding represents actual Step 2   │
  └───────────────────────────────────────────┘

JEPA Loss:
  ┌───────────────────────────────────────────┐
  │ loss = 1 - cosine_similarity(view1, view2)│
  │                                            │
  │ OR: L2 norm, MSE, InfoNCE                 │
  └───────────────────────────────────────────┘

```

---

## Comparison: Original JEPA vs Step-JEPA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ORIGINAL JEPA (Text ↔ Code)                            │
└─────────────────────────────────────────────────────────────────────────────┘

[System]
[User: "What is 2+2?"]  ← View 1 (Text)
<|predictor_1|>
<|predictor_2|>
[Assistant: "2+2 = 4"]  ← View 2 (Code) - ISOLATED

JEPA: Align user question embedding with code solution embedding


┌─────────────────────────────────────────────────────────────────────────────┐
│                     STEP-JEPA (Step 1 ↔ Step 2)                            │
└─────────────────────────────────────────────────────────────────────────────┘

[System]
[User: "What is 2+2?"]
[Step 1: "First, identify..."]  ← View 1 (Step 1)
<|predictor_1|>
<|predictor_2|>
[Step 2: "Second, add..."]      ← View 2 (Step 2) - ISOLATED
[Step 3+: "Final answer: 4"]

JEPA: Align Step 1 embedding with Step 2 embedding

```

---

## Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| **K** | K | 2 | Number of predictor tokens after Step 1 |
| **λ** | λ | 0.1 | JEPA loss weight |
| **γ** | γ | 1.0 | LM loss weight |
| **last_token** | - | -1 | Token offset for embedding extraction |

---

## Why K Predictor Tokens?

1. **Capacity**: K tokens provide more representational space to encode Step 1 information
2. **Flexibility**: Model learns to optimally use K tokens for prediction
3. **Regularization**: Forces information compression before prediction
4. **Empirical**: Paper shows K=2 works well

---

## Implementation Notes

- **Single forward pass**: Both LM loss and JEPA loss computed in one forward pass
- **Efficient masking**: Custom attention mask isolates Step 2
- **Dynamic K**: K can be any value (1, 2, 3, ...) via `--predictors K`
- **Compatible**: Works with LoRA, different loss types, etc.

---

## Training Command

```bash
# Step-JEPA with K=2 predictor tokens
python finetune_step_jepa_adapted.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --step_jepa \
  --predictors 2 \
  --lbd 0.1 \
  --gamma 1.0 \
  --last_token -1 \
  --lora
```

---

## Visualization

Run `python test_k_predictors.py` to see:
- Token sequences for K=1, 2, 3
- Attention mask patterns
- Embedding extraction positions

---

**Ready to train Step-JEPA!** 🚀

