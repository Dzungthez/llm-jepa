#!/bin/bash

# Train Step-JEPA model using custom trainer
# Proper architecture: Step2 isolated with custom attention mask

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Step-JEPA Training Script"
echo "=========================================="
echo ""
echo "Architecture:"
echo "  - User: Question"
echo "  - Assistant: Step1\\n\\nStep2\\n\\nStep3..."
echo "  - <|predictor|> inserted after Step1"
echo "  - Step2 isolated during JEPA forward pass"
echo "  - JEPA aligns: predictor embedding <-> Step2 end embedding"
echo ""

# Default configuration
TRAIN_FILE="step_jepa/gsm8k_step_jepa.jsonl"
OUTPUT_DIR="step_jepa/checkpoints"
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-5
MAX_LENGTH=1024

# Step-JEPA specific parameters
PREDICTORS=1    # Add <|predictor_1|> token after Step 1
LBD=0.1        # Lambda for JEPA loss weight
GAMMA=1.0       # Gamma for LM loss weight
LAST_TOKEN=-1   # Index of last token

# Parse command line arguments
USE_LORA="--lora"
LORA_RANK=16

while [[ $# -gt 0 ]]; do
    case $1 in
        --lbd)
            LBD="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --predictors)
            PREDICTORS="$2"
            shift 2
            ;;
        --no-lora)
            USE_LORA=""
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--lbd LAMBDA] [--gamma GAMMA] [--predictors N] [--no-lora] [--model MODEL] [--epochs N]"
            exit 1
            ;;
    esac
done

# Check if data file exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Training file not found: $TRAIN_FILE"
    echo ""
    echo "Please run first:"
    echo "  cd step_jepa"
    echo "  python prepare_step_data.py"
    exit 1
fi

# Print configuration
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Training file: $TRAIN_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo ""
echo "Step-JEPA Parameters:"
echo "  Predictor tokens: $PREDICTORS"
echo "  Lambda (JEPA weight): $LBD"
echo "  Gamma (LM weight): $GAMMA"
echo "  Last token index: $LAST_TOKEN"
if [ -n "$USE_LORA" ]; then
    echo "  LoRA: Enabled (rank=$LORA_RANK)"
else
    echo "  LoRA: Disabled"
fi
echo ""

# Build command
CMD="python train_step_jepa.py \
  --train_file=$TRAIN_FILE \
  --output_dir=$OUTPUT_DIR \
  --model_name=$MODEL_NAME \
  --num_epochs=$NUM_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --batch_size=$BATCH_SIZE \
  --grad_accum=$GRAD_ACCUM \
  --max_length=$MAX_LENGTH \
  --lbd=$LBD \
  --gamma=$GAMMA"

if [ -n "$USE_LORA" ]; then
    CMD="$CMD --lora --lora_rank=$LORA_RANK"
fi

# Detect multi-GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "Detected $GPU_COUNT GPUs. Use torchrun for multi-GPU training:"
        echo ""
        echo "  torchrun --nproc_per_node=$GPU_COUNT finetune.py \\"
        echo "    --train_file=$TRAIN_FILE \\"
        echo "    --output_dir=$OUTPUT_DIR \\"
        echo "    --predictors=$PREDICTORS \\"
        echo "    --lbd=$LBD $USE_LORA"
        echo ""
        read -p "Run with single GPU instead? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Multi-GPU training command shown above. Exiting."
            exit 0
        fi
    fi
fi

echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""
echo "Command: $CMD"
echo ""

# Run training
eval $CMD

echo ""
echo "=========================================="
echo "✅ Training completed!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "The model now has aligned representations for consecutive reasoning steps!"
echo ""

