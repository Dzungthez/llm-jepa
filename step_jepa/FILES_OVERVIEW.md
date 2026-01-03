# Step-JEPA Files Overview

This document lists all files in the `step_jepa` folder and their purposes.

---

## 📚 Documentation Files

### Quick Start
- **`QUICKSTART.md`**: Quick setup guide and first training run
  - How to install dependencies
  - How to prepare data
  - How to run first training
  - Simple examples

### Core Concepts
- **`README.md`**: Main documentation
  - Overview of Step-JEPA
  - How it differs from original JEPA
  - Architecture explanation
  - Usage guide

- **`K_PREDICTORS_EXPLAINED.md`**: Deep dive into K predictor tokens ⭐
  - What are predictor tokens
  - Why K tokens
  - How they work
  - Comparison with original JEPA
  - Recommended K values

- **`ARCHITECTURE_DIAGRAM.md`**: Visual architecture reference
  - Token sequence structure
  - Attention patterns
  - Forward pass flow
  - JEPA loss computation
  - Side-by-side comparison

### Reference Guides
- **`FULL_OPTIONS_GUIDE.md`**: Complete parameter reference
  - All command-line arguments
  - Training modes (regular, original JEPA, Step-JEPA)
  - Hyperparameter explanations
  - Model-specific settings
  - Example commands

- **`IMPLEMENTATION_COMPARISON.md`**: Standalone vs adapted approaches
  - Two implementation strategies
  - Pros/cons of each
  - When to use which
  - Feature comparison table

- **`IMPLEMENTATION_COMPLETE.md`**: Technical deep dive
  - Detailed implementation notes
  - Attention masking logic
  - Step boundary detection
  - Loss computation details

### Testing & Evaluation
- **`TESTING_GUIDE.md`**: How to test and verify
  - Testing step extraction
  - Visualizing attention masks
  - Verifying implementation
  - Debugging tips

- **`EVALUATION_GUIDE.md`**: Complete evaluation guide ⭐
  - How to evaluate trained models
  - GSM8K test set usage
  - Answer extraction formats
  - Result analysis
  - Model comparison
  - Troubleshooting

---

## 🐍 Python Scripts

### Data Preparation
- **`prepare_step_data.py`**: Processes raw GSM8K data
  - Extracts steps from `deepseek_response` field
  - Creates Step-JEPA training format
  - Outputs `gsm8k_step_jepa.jsonl`

### Training Scripts
- **`finetune_step_jepa_adapted.py`**: Main training script (RECOMMENDED) ⭐
  - Extends original `finetune.py`
  - Supports all original JEPA options
  - Adds Step-JEPA mode
  - K predictor tokens support
  - Fully featured and tested

- **`train_step_jepa.py`**: Standalone training script
  - Independent implementation
  - Uses `step_jepa_trainer.py`
  - Kept for reference

- **`step_jepa_trainer.py`**: Standalone trainer class
  - Custom `Trainer` implementation
  - Step-JEPA specific logic
  - Kept for reference

### Testing & Visualization
- **`test_components.py`**: Interactive testing script
  - Test step extraction
  - Visualize attention masks
  - Verify step boundaries
  - Comprehensive analysis

- **`test_k_predictors.py`**: K predictor token visualization
  - Shows token sequences for K=1,2,3
  - Attention mask visualization
  - Embedding extraction positions
  - Clear visual output

- **`quick_test.py`**: Quick verification script
  - Fast sanity check
  - Attention mask testing
  - Minimal dependencies

- **`verify_implementation.py`**: Automated verification
  - Tests data format
  - Tests attention masking
  - Tests trainer logic
  - Exit codes for CI/CD

### Evaluation
- **`evaluate_step_jepa.py`**: Main evaluation script ⭐
  - Load trained models (LoRA or full)
  - Generate on GSM8K test set
  - Extract and compare answers
  - Save detailed results
  - Multiple answer format support

- **`analyze_evaluation.py`**: Result analysis
  - Compute accuracy metrics
  - Error analysis and categorization
  - Compare multiple models
  - Sample predictions display

---

## 🔧 Shell Scripts

### Training
- **`train_adapted.sh`**: Simple training script
  - Uses `finetune_step_jepa_adapted.py`
  - K=2 predictor tokens (default)
  - Easy to configure
  - Recommended for quick starts

- **`train_adapted_full.sh`**: Comprehensive training script
  - All options demonstrated
  - Hyperparameter sweeps
  - Multiple configurations
  - Advanced usage examples

### Evaluation
- **`evaluate.sh`**: Simple evaluation script
  - Uses `evaluate_step_jepa.py`
  - Evaluates on GSM8K test set
  - Saves results and summary
  - Easy to configure

---

## 📊 Data Files

- **`gsm8k_step_jepa.jsonl`**: Training data (6863 examples)
  - Processed from `gsm8k_synthetic_data.jsonl`
  - Each line is a JSON object with:
    - `question`: Original math problem
    - `answer`: Final answer
    - `ground_truth`: Ground truth answer
    - `total_steps`: Number of reasoning steps
    - `messages`: Training format (system, user, assistant)

---

## 📁 Output Folders (created during training)

- **`checkpoints_adapted/`**: Checkpoints from adapted trainer
- **`outputs/`**: General output directory
- **`logs/`**: Training logs (if created)

---

## 🗂️ File Organization by Purpose

### If you want to...

**Understand the concept:**
1. Read `README.md`
2. Read `K_PREDICTORS_EXPLAINED.md`
3. Look at `ARCHITECTURE_DIAGRAM.md`

**Get started quickly:**
1. Read `QUICKSTART.md`
2. Run `bash train_adapted.sh`

**Train Step-JEPA:**
1. Use `finetune_step_jepa_adapted.py`
2. Configure via `train_adapted.sh` or command line
3. Refer to `FULL_OPTIONS_GUIDE.md` for options

**Test implementation:**
1. Run `python test_k_predictors.py`
2. Run `python test_components.py`
3. Run `python verify_implementation.py`

**Evaluate trained model:**
1. Run `bash evaluate.sh`
2. Run `python analyze_evaluation.py evaluation_results.jsonl`
3. Compare models with `python analyze_evaluation.py eval_*.jsonl --compare`

**Understand implementation:**
1. Read `IMPLEMENTATION_COMPLETE.md`
2. Read `IMPLEMENTATION_COMPARISON.md`
3. Read source code in `finetune_step_jepa_adapted.py`

**Prepare custom data:**
1. Study `prepare_step_data.py`
2. Adapt for your dataset
3. Follow same format as `gsm8k_step_jepa.jsonl`

---

## 📈 Recommended Workflow

```
1. Read QUICKSTART.md
   ↓
2. Run python test_k_predictors.py (verify implementation)
   ↓
3. Run bash train_adapted.sh (train model)
   ↓
4. Monitor training (see losses logged)
   ↓
5. Run bash evaluate.sh (evaluate on test set)
   ↓
6. Run python analyze_evaluation.py evaluation_results.jsonl (analyze)
   ↓
7. Experiment with hyperparameters (see FULL_OPTIONS_GUIDE.md)
   ↓
8. Run ablation studies (K, λ, seeds)
   ↓
9. Compare models (python analyze_evaluation.py eval_*.jsonl --compare)
```

---

## 🎯 Key Files for Different Users

### For Researchers
- `K_PREDICTORS_EXPLAINED.md` - Understand the method
- `ARCHITECTURE_DIAGRAM.md` - Visual reference
- `IMPLEMENTATION_COMPLETE.md` - Technical details
- `finetune_step_jepa_adapted.py` - Implementation

### For Practitioners
- `QUICKSTART.md` - Get started fast
- `train_adapted.sh` - Simple training
- `evaluate.sh` - Simple evaluation
- `FULL_OPTIONS_GUIDE.md` - All options
- `EVALUATION_GUIDE.md` - Evaluation guide
- `test_k_predictors.py` - Verify it works

### For Developers
- `IMPLEMENTATION_COMPARISON.md` - Design decisions
- `finetune_step_jepa_adapted.py` - Main implementation
- `step_jepa_trainer.py` - Alternative implementation
- `test_components.py` - Testing utilities

---

## 📝 File Count Summary

- **Documentation**: 11 markdown files
- **Python scripts**: 9 files
- **Shell scripts**: 3 files
- **Data files**: 1 JSONL file (6863 examples)

**Total**: 24 files (+ training/evaluation outputs)

---

## 🔄 Update History

- Initial implementation: Standalone approach
- Refactor: Adapted approach extending original `finetune.py`
- Feature: K predictor tokens support
- Fix: Deprecation warnings resolved
- Enhancement: Loss logging added
- Documentation: Comprehensive guides created
- **New: Evaluation system added**
  - `evaluate_step_jepa.py` for model evaluation
  - `analyze_evaluation.py` for result analysis
  - `EVALUATION_GUIDE.md` for documentation
  - Full GSM8K test set support

---

**All files are ready for production use!** ✅

For the latest updates, see git history or check file modification dates.
