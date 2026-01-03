#!/bin/bash

# Evaluate Step-JEPA model on GSM8K test set

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Step-JEPA Model Evaluation"
echo "=========================================="
echo ""

# Configuration
MODEL_PATH="./checkpoints_adapted"  # Path to trained model
TEST_FILE="../datasets/gsm8k_test.jsonl"
OUTPUT_FILE="./evaluation_results.jsonl"
MAX_NEW_TOKENS=512
TEMPERATURE=0.0  # Greedy decoding for evaluation

# Optional: limit number of examples for quick testing
# MAX_EXAMPLES=100
MAX_EXAMPLES=""

echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model directory not found: $MODEL_PATH"
    echo "   Please train a model first using train_adapted.sh"
    exit 1
fi

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

# Build command
CMD="python evaluate_step_jepa.py \
  --model_path $MODEL_PATH \
  --test_file $TEST_FILE \
  --output_file $OUTPUT_FILE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE"

# Add max_examples if set
if [ -n "$MAX_EXAMPLES" ]; then
    CMD="$CMD --max_examples $MAX_EXAMPLES"
    echo "Note: Evaluating only first $MAX_EXAMPLES examples (for testing)"
    echo ""
fi

echo "Running evaluation..."
echo ""

# Execute
$CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_FILE"
echo "  - ${OUTPUT_FILE/.jsonl/_summary.json}"
echo ""

