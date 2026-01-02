#!/bin/bash

# Training script for GSM8K with JEPA on single GPU
# Usage: bash train_gsm8k_single_gpu.sh

set -e  # Exit on error

# ===========================================
# Configuration
# ===========================================

# Model settings
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR="./fine-tuned-gsm8k-pred3"

# Dataset
TRAIN_FILE="datasets/gsm8k_train.jsonl"
EVAL_FILE="datasets/gsm8k_test.jsonl"

# Training hyperparameters
NUM_EPOCHS=4
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_LENGTH=512

# JEPA settings
PREDICTORS=3          # Number of predictor tokens
LBD=0.5              # Lambda for JEPA loss (try 0.5 or 2.0)
GAMMA=1.0            # Gamma for LM loss
LAST_TOKEN=-2        # -1 for Llama-3.2, -2 for most models

# Seeds
FINETUNE_SEED=42

# Training mode: "jepa" or "regular"
MODE="jepa"

# Optional flags (uncomment to enable)
# USE_LORA="--lora --lora_rank=16"
USE_LORA=""

# Advanced JEPA options (uncomment to enable)
# JEPA_OPTIONS="--jepa_l2"           # Use L2 norm instead of cosine
# JEPA_OPTIONS="--jepa_mse"          # Use MSE instead of cosine
# JEPA_OPTIONS="--infonce"           # Use InfoNCE contrastive loss
# JEPA_OPTIONS="--additive_mask"     # Use efficient additive mask (2x speedup)
# JEPA_OPTIONS="--jepa_ratio=0.5"    # Random JEPA dropout (0.5 = 50% batches use JEPA)
JEPA_OPTIONS=""

# ===========================================
# Verify files exist
# ===========================================

echo "=========================================="
echo "GSM8K Training Script (Single GPU)"
echo "=========================================="

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Training file not found: $TRAIN_FILE"
    echo "Current directory: $(pwd)"
    exit 1
fi

if [ ! -f "$EVAL_FILE" ]; then
    echo "WARNING: Eval file not found: $EVAL_FILE"
    echo "Training without evaluation."
    EVAL_FILE=""
fi

# ===========================================
# Build command
# ===========================================

CMD="python finetune.py \
  --train_file=$TRAIN_FILE \
  --output_dir=$OUTPUT_DIR \
  --model_name=$MODEL_NAME \
  --num_epochs=$NUM_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --batch_size=$BATCH_SIZE \
  --grad_accum=$GRAD_ACCUM \
  --max_length=$MAX_LENGTH \
  --finetune_seed=$FINETUNE_SEED"

# Add eval file if it exists
if [ -n "$EVAL_FILE" ]; then
    CMD="$CMD --eval_file=$EVAL_FILE"
fi

# Add JEPA-specific options
if [ "$MODE" == "jepa" ]; then
    echo "Training mode: JEPA"
    CMD="$CMD \
  --predictors=$PREDICTORS \
  --lbd=$LBD \
  --gamma=$GAMMA \
  --last_token=$LAST_TOKEN \
  $JEPA_OPTIONS"
else
    echo "Training mode: Regular SFT"
    CMD="$CMD --regular"
fi

# Add LoRA if enabled
if [ -n "$USE_LORA" ]; then
    CMD="$CMD $USE_LORA"
    echo "LoRA: Enabled"
else
    echo "LoRA: Disabled (full fine-tuning)"
fi

# ===========================================
# Print configuration
# ===========================================

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Max length: $MAX_LENGTH"
echo ""

if [ "$MODE" == "jepa" ]; then
    echo "JEPA Settings:"
    echo "  Predictor tokens: $PREDICTORS"
    echo "  Lambda (JEPA weight): $LBD"
    echo "  Gamma (LM weight): $GAMMA"
    echo "  Last token index: $LAST_TOKEN"
    echo "  Extra options: $JEPA_OPTIONS"
    echo ""
fi

echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo ""

# ===========================================
# Run training
# ===========================================

echo "Starting training..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="

