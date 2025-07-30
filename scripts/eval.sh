#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@example.com  # Replace with your email

echo "Starting evaluation loop..."

SEEDS=(9584 123 2024 31415 777)
TEST_JSON="hotel_dataset/indo/hotel_aste_test_augmented.json"

for SEED in "${SEEDS[@]}"
do
  echo "==========================================="
  echo "Evaluating models for seed: $SEED"
  echo "==========================================="

  MODEL_DIR="outputs/models/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants"
  OUTPUT_DIR="outputs/evals/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants"

  for MODEL_PATH in "$MODEL_DIR"/*; do
    MODEL_NAME=$(basename "$MODEL_PATH")

    if [ -d "$MODEL_PATH" ] && [ ! -d "$OUTPUT_DIR/$MODEL_NAME" ]; then
      echo "Evaluating: $MODEL_NAME"

      python run_eval.py \
        --test_json_path "$TEST_JSON" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 5 \
        --save_predictions

      echo "--- Done evaluating $MODEL_NAME ---"
    else
      echo "Skipping $MODEL_NAME — already evaluated or not trained yet."
    fi
  done

  echo -e "\nFinished all evaluations for seed: $SEED\n"
  echo "\n----------------------------------------------------------------\n"
done

echo "All evaluations completed."
