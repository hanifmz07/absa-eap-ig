#!/bin/bash
# Define the log file names for clarity
LOG_BASE_NAME="eval"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).err"

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

echo "========================================================" > "$STDOUT_LOG"
echo "Starting Evaluation script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
  SEEDS=(42 123 2024 31415 777)
  TEST_JSON="hotel_dataset/indo/hotel_aste_test_augmented.json"

  for SEED in "${SEEDS[@]}"
  do
    MODEL_DIR="outputs/models/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants"
    OUTPUT_DIR="outputs/evals/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants"
    
    for MODEL_PATH in "$MODEL_DIR"/*; do
      MODEL_NAME=$(basename "$MODEL_PATH")
      EVAL_RESULT_DIR="$OUTPUT_DIR/$MODEL_NAME"

      # Only run if model folder exists and hasn't been evaluated yet
      if [ -d "$MODEL_PATH" ] && [ ! -d "$EVAL_RESULT_DIR" ]; then
        echo "-----------------------------------------------------------"
        echo "Evaluating model: $MODEL_NAME (Seed: $SEED)"
        echo "-----------------------------------------------------------"
        
        python run_eval.py \
          --test_json_path "$TEST_JSON" \
          --model_path "$MODEL_PATH" \
          --output_dir "$EVAL_RESULT_DIR" \
          --batch_size 5 \
          --save_predictions

      else
        echo "-----------------------------------------------------------"
        echo "Skipping $MODEL_NAME — either not trained yet or already evaluated."
        echo "-----------------------------------------------------------"
      fi
    done
  done

  echo ""
  echo "========================================================"
  echo "All evaluations completed at: $(date)"
  echo "========================================================"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"
