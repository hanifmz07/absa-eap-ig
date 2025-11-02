#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=7

# Extract language parameters from the command line arguments
# Usage: ./sft5.sh <language> <dataset_folder>
LANGUAGE="$1"
if [ -z "$LANGUAGE" ]; then
    echo "Error: No language specified."
    exit 1
fi
# Validate the language argument
if [[ "$LANGUAGE" != "indo" && "$LANGUAGE" != "eng" && "$LANGUAGE" != "sunda" ]]; then
    echo "Error: Invalid language specified. Use 'indo', 'eng', or 'sunda'."
    exit 1
fi

DATASET_FOLDER="$2"
# Validate the dataset folder argument
if [ -z "$DATASET_FOLDER" ]; then
    echo "Error: Dataset folder must be specified. Name a folder located in the hotel_dataset/{lang} directory."
    exit 1 # Exit with a non-zero status to indicate an error
fi

BATCH_SIZE="$3"
# Validate the batch size argument
if [ -z "$BATCH_SIZE" ]; then
    echo "Error: Batch size must be specified."
    exit 1 # Exit with a non-zero status to indicate an error
fi

TOPK_CIRCUIT="$4"
if [ -z "$TOPK_CIRCUIT" ]; then
    echo "Error: TOPK_CIRCUIT must be specified for eval_circuit.sh."
    exit 1 # Exit with a non-zero status to indicate an error
fi

# Seeds for the SFT process
SEED=777

# Define the log file names for clarity
LOG_BASE_NAME="eval"
LOG_DIR="logs"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}/${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}/${PID}_$(date).err"
# Create necessary directories for logs
mkdir -p "${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}"

echo "========================================================" > "$STDOUT_LOG"
echo "Starting Evaluation script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
  TEST_JSON="hotel_dataset/${LANGUAGE}/${DATASET_FOLDER}/hotel_aste_test_augmented.json"
  TOPKS=($TOPK_CIRCUIT)
  for TOPK_CIRCUIT in "${TOPKS[@]}"; do
    
    MODEL_DIR="outputs/modelsbestv3.7.3/eap/${DATASET_FOLDER}/circuit-${LANGUAGE}_finetune-${LANGUAGE}/seed_$SEED/aos_sequence_variants/topk_${TOPK_CIRCUIT}"
    OUTPUT_DIR="outputs/evalsbestv3.7.3/eap/${DATASET_FOLDER}/circuit-${LANGUAGE}_finetune-${LANGUAGE}/seed_$SEED/aos_sequence_variants/topk_${TOPK_CIRCUIT}"
    
    for MODEL_PATH in "$MODEL_DIR"/*; do
      MODEL_NAME=$(basename "$MODEL_PATH")
      EVAL_RESULT_DIR="$OUTPUT_DIR/$MODEL_NAME"

      # Only run if model folder exists and hasn't been evaluated yet
      if [ -d "$MODEL_PATH" ] && [ ! -d "$EVAL_RESULT_DIR" ]; then
        echo "-----------------------------------------------------------"
        echo "Evaluating model: $MODEL_NAME (Seed: $SEED) (TopK: $TOPK_CIRCUIT)"
        echo "-----------------------------------------------------------"
        
        python run_eval.py \
          --test_json_path "$TEST_JSON" \
          --model_path "$MODEL_PATH" \
          --output_dir "$EVAL_RESULT_DIR" \
          --batch_size $BATCH_SIZE \
          --prompt_type "mvp" \
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
