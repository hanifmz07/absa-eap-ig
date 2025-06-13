#!/bin/bash
#SBATCH --job-name=evals_loop
#SBATCH --output=logs/eval_loop_%j.out
#SBATCH --error=logs/eval_loop_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00

echo "Starting evaluation loop..."

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
      echo "Evaluating model: $MODEL_NAME (Seed: $SEED)"
      
      python run_eval.py \
        --test_json_path "$TEST_JSON" \
        --model_path "$MODEL_PATH" \
        --output_dir "$EVAL_RESULT_DIR" \
        --batch_size 5 \
        --save_predictions

    else
      echo "Skipping $MODEL_NAME — either not trained yet or already evaluated."
    fi
  done
done

echo "All evaluations completed."
