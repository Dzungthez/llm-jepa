#!/bin/bash

# Train Step-JEPA using adapted finetune.py (Option 1)
# This version extends the existing RepresentationTrainer

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Step-JEPA Training (Adapted Version)"
echo "=========================================="
echo ""
echo "This uses finetune.py's infrastructure:"
echo "  - Extends RepresentationTrainer class"
echo "  - Uses existing JEPA logic"
echo "  - Adds Step-JEPA specific masking"
echo "  - Supports K predictor tokens after Step 1"
echo ""

# Configuration
TRAIN_FILE="./gsm8k_step_jepa.jsonl"
OUTPUT_DIR="./checkpoints_adapted"
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

# Hyperparameters
NUM_EPOCHS=3
LEARNING_RATE=2e-5
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_LENGTH=1024

# Step-JEPA parameters
LBD=0.1        # JEPA loss weight
GAMMA=1.0      # LM loss weight
PREDICTORS=2   # K predictor tokens after Step 1

# LoRA settings
USE_LORA=1
LORA_RANK=16

# Build command
CMD="python finetune_step_jepa_adapted.py \
  --train_file=$TRAIN_FILE \
  --output_dir=$OUTPUT_DIR \
  --model_name=$MODEL_NAME \
  --num_epochs=$NUM_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --batch_size=$BATCH_SIZE \
  --grad_accum=$GRAD_ACCUM \
  --max_length=$MAX_LENGTH \
  --step_jepa \
  --predictors=$PREDICTORS \
  --lbd=$LBD \
  --gamma=$GAMMA"

# Note: debug=5 is set by default in the script for loss logging
# Override with --debug=N if you want different behavior

if [ -n "$USE_LORA" ]; then
    CMD="$CMD --lora --lora_rank=$LORA_RANK"
fi

echo "Command:"
echo "$CMD"
echo ""
echo "K predictor tokens: $PREDICTORS"
echo ""

# Execute
$CMD

