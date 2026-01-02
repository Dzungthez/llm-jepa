# Step-JEPA: Step-wise Joint Embedding Predictive Architecture

This implements Step-JEPA, a novel adaptation of LLM-JEPA for reasoning step alignment.

## 🎯 Key Idea

Instead of aligning "Text" and "Code" views (as in LLM-JEPA paper), Step-JEPA aligns **consecutive reasoning steps**:
- **View 1**: First step of reasoning (e.g., "First, I note that...")
- **View 2**: Second step of reasoning (e.g., "In May, she sold half...")

This teaches the model to create consistent representations across the reasoning process.

## 🏗️ Architecture

### Data Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Natalia sold clips to 48 friends..."},
    {"role": "assistant", "content": "Step1\n\nStep2\n\nStep3..."}
  ],
  "total_steps": 7
}
```

### Training Mechanism

**Two Forward Passes:**

1. **Forward Pass 1: Normal NTP Loss**
   - Standard causal language modeling
   - All tokens attend normally
   - Loss computed on assistant response

2. **Forward Pass 2: JEPA Embeddings with Custom Attention**
   - Insert `<|predictor|>` token after Step 1
   - Custom attention mask isolates Step 2:
     ```
     [System, User, Step1, <|predictor|>] → Normal causal attention
     [Step2 tokens]                        → Isolated (only attend to themselves)
     [Step3+ tokens]                       → Normal causal attention
     ```
   - Extract embeddings:
     * **View 1**: Hidden state at `<|predictor|>` token
     * **View 2**: Hidden state at end of Step 2 (`\n\n` token)
   - Compute JEPA loss (cosine similarity, L2, or MSE)

**Total Loss:**
```
L_total = γ × L_LM + λ × L_JEPA
```

### Attention Masking Details

The custom attention mask ensures Step 2 is **completely isolated** during JEPA embedding extraction:

```
Position:  [0...pred]  [pred+1...step2_end]  [step2_end+1...seq_len]
Tokens:    [Sys|Q|S1]  [Step 2]              [Step 3+]
Attention: Normal      Isolated              Normal
           causal      (only self)           causal (includes S2)
```

This prevents information leakage while maintaining causality for other parts of the sequence.

## 📊 Dataset

### Input Data
- Source: `gsm8k_synthetic_data.jsonl`
- Field used: `deepseek_response` (detailed step-by-step solutions)
- Examples: 6,862 math problems

### Preprocessing
```bash
python prepare_step_data.py
```

This extracts all reasoning steps from `deepseek_response` and creates training examples.

### Output Format
- File: `gsm8k_step_jepa.jsonl`
- Format: JSONL with 3 messages (system, user, assistant)
- Success rate: 100% (all 6,862 examples have ≥2 steps)

## 🚀 Training

### Quick Start
```bash
bash train.sh
```

### Custom Training
```bash
python train_step_jepa.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./checkpoints \
  --lora \
  --lora_rank 16 \
  --lbd 0.1 \
  --gamma 1.0 \
  --jepa_loss_type cosine \
  --num_epochs 3 \
  --batch_size 4 \
  --grad_accum 4
```

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lbd` | JEPA loss weight (λ) | 0.1 |
| `--gamma` | LM loss weight (γ) | 1.0 |
| `--jepa_loss_type` | JEPA loss function | `cosine` |
| `--lora` | Use LoRA fine-tuning | False |
| `--lora_rank` | LoRA rank | 16 |

### JEPA Loss Types
- **cosine**: `1 - cosine_similarity(view1, view2)`
- **l2**: L2 norm of difference
- **mse**: Mean squared error

## 🔬 Implementation Details

### Custom Trainer (`StepJEPATrainer`)
- Extends HuggingFace `Trainer`
- Automatically finds step boundaries in tokenized sequences
- Inserts `<|predictor|>` token dynamically
- Builds custom attention masks per example
- Extracts embeddings at correct positions
- Computes combined LM + JEPA loss

### Key Methods
- `_find_step_boundaries()`: Locates `\n\n` separators
- `_insert_predictor_token()`: Adds special token after Step 1
- `_build_step2_isolation_mask()`: Creates custom attention mask
- `compute_loss()`: Orchestrates two forward passes and loss computation

## 📁 Files

```
step_jepa/
├── README.md                    # This file
├── prepare_step_data.py         # Data preprocessing
├── step_jepa_trainer.py         # Custom trainer implementation
├── train_step_jepa.py           # Training script
├── train.sh                     # Quick start script
├── gsm8k_step_jepa.jsonl       # Processed training data
└── check_format.py              # Data format verification
```

## 🎓 Theoretical Background

### Why Step-JEPA?

Traditional language modeling learns to predict the next token, but doesn't explicitly align representations across reasoning steps. Step-JEPA addresses this by:

1. **Preventing collapse**: JEPA loss ensures different steps maintain similar but distinct representations
2. **Encouraging consistency**: Consecutive steps should build on each other
3. **Improving reasoning**: Better step alignment → more coherent reasoning chains

### Comparison to LLM-JEPA

| Aspect | LLM-JEPA | Step-JEPA |
|--------|----------|-----------|
| View 1 | Text (question) | First reasoning step |
| View 2 | Code (solution) | Second reasoning step |
| Goal | Align natural language ↔ formal solution | Align consecutive reasoning steps |
| Application | General code generation | Mathematical reasoning |

## 🔍 Verification

Check data format:
```bash
python check_format.py
```

Expected output:
```
Format check:
  Total messages: 3
  Roles: ['system', 'user', 'assistant']
  
User message: Natalia sold clips to 48 of her friends...

Assistant message (first 400 chars):
First, I note that Natalia sold 48 clips in April.

In May, she sold half as many clips as she did in April...

Total steps: 7
Assistant content has 6 step separators (\n\n)
```

## 📈 Expected Results

Step-JEPA should improve:
- **Step coherence**: Consecutive steps are more logically connected
- **Solution quality**: Better overall reasoning chains
- **Generalization**: More robust to different problem types

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- PEFT (for LoRA)
- Datasets
- CUDA-capable GPU (training)

## 🐛 Debugging

Set `--debug 1` to see loss values during training:
```bash
python train_step_jepa.py --debug 1 ...
```

Output example:
```
LM loss: 2.3451, JEPA loss: 0.1234, Total: 2.3574
```

## 📚 Citation

If you use Step-JEPA, please cite both:
1. The LLM-JEPA paper (original JEPA for LLMs)
2. This implementation

## 🤝 Contributing

This is a research implementation. Feel free to:
- Experiment with different loss functions
- Try other model architectures
- Test on different datasets (not just GSM8K)
- Adjust attention masking strategies

## ⚠️ Known Limitations

1. **Step detection**: Currently relies on `\n\n` separators - may fail on different formats
2. **Computational cost**: Two forward passes per batch (2x compute vs standard fine-tuning)
3. **Memory usage**: Stores hidden states for embedding extraction
4. **Dataset specific**: Designed for GSM8K synthetic data format

## 🚧 Future Work

- [ ] Dynamic step detection (handle various formats)
- [ ] Multi-step JEPA (align 3+ steps simultaneously)
- [ ] Cross-dataset evaluation
- [ ] Attention visualization tools
- [ ] Memory-efficient implementation (gradient checkpointing for JEPA pass)

---

**Let's change the world, one reasoning step at a time!** 🌍✨

