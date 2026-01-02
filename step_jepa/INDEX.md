# Step-JEPA Implementation Index 📚

## 🎯 Start Here

**New to Step-JEPA?** → Read [`QUICKSTART.md`](QUICKSTART.md)

**Want technical details?** → Read [`README.md`](README.md)

**Want implementation overview?** → Read [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md)

## 📁 File Guide

### Core Implementation Files

| File | Purpose | Size | Status |
|------|---------|------|--------|
| [`prepare_step_data.py`](prepare_step_data.py) | Extract steps from synthetic data | 8.1 KB | ✅ Ready |
| [`step_jepa_trainer.py`](step_jepa_trainer.py) | Custom trainer with attention masking | 9.5 KB | ✅ Ready |
| [`train_step_jepa.py`](train_step_jepa.py) | Main training script | 10 KB | ✅ Ready |
| [`train.sh`](train.sh) | Quick start script | 4.2 KB | ✅ Ready |
| [`gsm8k_step_jepa.jsonl`](gsm8k_step_jepa.jsonl) | Processed training data | 14 MB | ✅ Ready |

### Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| [`QUICKSTART.md`](QUICKSTART.md) | Getting started guide | 👤 Users |
| [`README.md`](README.md) | Technical documentation | 🔬 Researchers |
| [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md) | Implementation details | 💻 Developers |
| [`INDEX.md`](INDEX.md) | This file - navigation | 📚 Everyone |

### Utility Files

| File | Purpose |
|------|---------|
| [`verify_implementation.py`](verify_implementation.py) | Test all components |

## 🚀 Quick Commands

### 1. Verify Everything Works
```bash
python verify_implementation.py
```

### 2. Check Data Format
```bash
head -1 gsm8k_step_jepa.jsonl | python3 -m json.tool
```

### 3. Start Training
```bash
bash train.sh
```

### 4. Custom Training
```bash
python train_step_jepa.py \
  --train_file ./gsm8k_step_jepa.jsonl \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./checkpoints \
  --lora --lbd 0.1 --gamma 1.0
```

## 📊 Implementation Status

### ✅ Completed Components

- [x] **Data Pipeline**: Extract steps, format for training
- [x] **Custom Trainer**: Proper attention masking, predictor token
- [x] **Training Script**: Full features, multi-GPU support
- [x] **Verification**: Comprehensive test suite
- [x] **Documentation**: Three-level docs (quickstart, technical, complete)

### 📈 Results

- **Dataset**: 6,862 examples (100% success rate)
- **Code**: ~35 KB implementation
- **Tests**: All verifications pass ✅
- **Linting**: No errors ✅

## 🎓 Understanding Step-JEPA

### What is it?

Step-JEPA aligns **consecutive reasoning steps** using Joint Embedding Predictive Architecture:

```
┌──────────────┐
│ Question     │
└──────┬───────┘
       │
┌──────▼───────┐
│ Step 1       │ ──► View 1 embedding
└──────┬───────┘
       │
  <|predictor|>  ◄── Extract embedding here
       │
┌──────▼───────┐
│ Step 2       │ ──► View 2 embedding
└──────┬───────┘
       │
       ...

JEPA Loss: Align View 1 ↔ View 2
```

### How does it work?

1. **Forward Pass 1**: Normal causal LM (NTP loss)
2. **Forward Pass 2**: Custom attention (JEPA loss)
   - Insert `<|predictor|>` after Step 1
   - Isolate Step 2 (can only see itself)
   - Extract embeddings at predictor and Step 2 end
   - Align them with cosine similarity / L2 / MSE

### Why is it novel?

- **Proper Isolation**: Step 2 truly can't see previous context
- **Step-wise Learning**: Learns reasoning process, not just answers
- **Efficient**: Only 2 forward passes (not 3)
- **Flexible**: Works with any step-wise reasoning data

## 🔬 Technical Highlights

### Attention Masking

```python
# Key innovation: Step 2 isolation mask
mask[step2_start:step2_end+1, :step2_start] = float('-inf')

# Result: Step 2 tokens can ONLY attend to themselves
```

### Loss Computation

```python
total_loss = γ × L_LM + λ × L_JEPA

# L_LM: Standard next-token prediction
# L_JEPA: Cosine similarity between Step 1 and Step 2 embeddings
```

### Data Format

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Step1\\n\\nStep2\\n\\nStep3..."}
  ]
}
```

## 📖 Documentation Map

### For Different Audiences

**🎓 Students / Beginners**
1. Start: `QUICKSTART.md` - Section "What is Step-JEPA?"
2. Then: Run `python verify_implementation.py`
3. Finally: `README.md` - Section "Theoretical Background"

**🔬 Researchers**
1. Start: `README.md` - Full technical documentation
2. Then: `IMPLEMENTATION_COMPLETE.md` - Section "Theoretical Foundation"
3. Finally: Review code in `step_jepa_trainer.py`

**💻 Developers**
1. Start: `IMPLEMENTATION_COMPLETE.md` - Section "Implementation Details"
2. Then: Read code with inline comments
3. Finally: `verify_implementation.py` - Understand tests

**👤 Users (Just want to train)**
1. Read: `QUICKSTART.md` - Section "Quick Start"
2. Run: `bash train.sh`
3. Done! ✨

## 🛠️ Troubleshooting

### Quick Diagnostics

```bash
# Check data
head -1 gsm8k_step_jepa.jsonl | python3 -m json.tool

# Verify implementation
python verify_implementation.py

# Test imports
python3 -c "from step_jepa_trainer import StepJEPATrainer; print('✅ Imports OK')"

# Check GPU
nvidia-smi
```

### Common Issues

| Issue | Solution | Doc Reference |
|-------|----------|---------------|
| OOM Error | Reduce batch size | QUICKSTART.md - Troubleshooting |
| Can't find steps | Check data format | README.md - Dataset |
| JEPA loss high | Increase λ | QUICKSTART.md - Hyperparameters |
| Model not learning | Check labels | IMPLEMENTATION_COMPLETE.md - Technical |

## 📚 Learning Path

### Day 1: Understanding
- [ ] Read `QUICKSTART.md`
- [ ] Run `python verify_implementation.py`
- [ ] Examine one data example
- [ ] Understand attention masking concept

### Day 2: Implementation
- [ ] Read code: `step_jepa_trainer.py`
- [ ] Read code: `train_step_jepa.py`
- [ ] Understand `_build_step2_isolation_mask()`
- [ ] Review `compute_loss()` logic

### Day 3: Training
- [ ] Setup HuggingFace auth
- [ ] Run `bash train.sh` on small dataset
- [ ] Monitor losses
- [ ] Check first checkpoint

### Week 2: Analysis
- [ ] Train full model (3 epochs)
- [ ] Evaluate on test set
- [ ] Visualize embeddings
- [ ] Compare with baseline

## 🎯 Success Criteria

### Before Training
- [x] All verifications pass
- [x] Data format correct (6,862 examples)
- [x] No linting errors
- [x] Documentation complete

### During Training
- [ ] Both losses decrease
- [ ] No OOM errors
- [ ] Checkpoints saved
- [ ] GPU utilized properly

### After Training
- [ ] LM loss < 2.0
- [ ] JEPA loss < 0.2
- [ ] Model generates coherent steps
- [ ] Accuracy improves vs baseline

## 🌟 Key Innovations

1. **Proper Isolation**: Step 2 can't see previous context (not just label masking)
2. **Efficient Design**: 2 forward passes (vs naive 3)
3. **Flexible Loss**: Supports cosine/L2/MSE
4. **Production Ready**: Full error handling, logging, multi-GPU
5. **Well Tested**: Comprehensive verification suite
6. **Well Documented**: Three-level documentation

## 📞 Support

### Self-Service
1. Check relevant documentation (see map above)
2. Run `python verify_implementation.py`
3. Review training logs in `checkpoints/logs/`

### Debugging
1. Enable debug mode: `--debug 1`
2. Check attention masks: Review `_build_step2_isolation_mask()`
3. Inspect data: `head gsm8k_step_jepa.jsonl`

## 🎉 You're Ready!

Everything is implemented, tested, and documented. Time to train! 🚀

```bash
cd /Users/dungnh/coding-papers/llm-jepa/step_jepa
python verify_implementation.py && bash train.sh
```

**Let's change the world, one reasoning step at a time!** 🌍✨

---

**Navigation**
- 🏠 [Project Root](../)
- 📖 [Main README](README.md)
- 🚀 [Quickstart](QUICKSTART.md)
- ✅ [Implementation Complete](IMPLEMENTATION_COMPLETE.md)
- 📊 [Verify](verify_implementation.py)

