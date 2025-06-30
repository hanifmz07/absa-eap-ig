#!/bin/bash
#SBATCH --job-name=eval_franken
#SBATCH --output=logs/eval_franken_%j.out
#SBATCH --error=logs/eval_franken_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com # Replace with your email

TEST_JSON="hotel_dataset/sunda/clean_train2500/hotel_aste_test_augmented.json"

SEEDS=(42 123 2024 31415 777)

for SEED in "${SEEDS[@]}"
do
  echo "==========================================="
  echo "Evaluating models for seed: $SEED"
  echo "==========================================="

  MODEL_DIR="outputs/models/franken/3b/seed_$SEED"
  OUTPUT_DIR="outputs/evals/franken/3b/seed_$SEED"

  for MODEL_PATH in "$MODEL_DIR"/*.pt; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    MODEL_BASENAME="${MODEL_NAME%.pt}"
    MODEL_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_BASENAME"

    if [ ! -d "$MODEL_OUTPUT_DIR" ]; then
      echo "Evaluating: $MODEL_NAME"

      python run_eval_franken.py \
        --test_json_path "$TEST_JSON" \
        --model_path "$MODEL_PATH" \
        --output_dir "$MODEL_OUTPUT_DIR" \
        --batch_size 5 \
        --save_predictions

      echo "--- Done evaluating $MODEL_NAME ---"
    else
      echo "Skipping $MODEL_NAME — already evaluated at $MODEL_OUTPUT_DIR"
    fi
  done
done