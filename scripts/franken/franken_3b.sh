#!/bin/bash
#SBATCH --job-name=finetune_franken_3b
#SBATCH --output=logs/finetune_franken_3b_%j.out
#SBATCH --error=logs/finetune_franken_3b_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com

SEEDS=(42 123 2024 31415 777)

BASE_MODEL="Qwen/Qwen2.5-0.5B"
DATA_PATH="hotel_dataset/indo/clean_train2500/hotel_aste_train_augmented_noreasoning.json"

for SEED in "${SEEDS[@]}"; do
  echo "======================"
  echo "Fine-tuning seed: $SEED"
  echo "======================"

  MODEL_DIR_3A="outputs/models/franken/3a/seed_${SEED}"
  MODEL_DIR_3B="outputs/models/franken/3b/seed_${SEED}"

  mkdir -p "$MODEL_DIR_3B"

  for WEIGHT_PATH in "$MODEL_DIR_3A"/*.pt; do
    WEIGHT_NAME=$(basename "$WEIGHT_PATH")
    SAVE_PATH="$MODEL_DIR_3B/$WEIGHT_NAME"

    echo "→ Fine-tuning: $WEIGHT_NAME"

    python run_franken_3b.py \
      --base_model_name "$BASE_MODEL" \
      --franken_weights_path "$WEIGHT_PATH" \
      --json_data_path "$DATA_PATH" \
      --device cuda \
      --num_epochs 20 \
      --batch_size 16 \
      --lr 1e-4 \
      --freeze_embedding \
      --seed $SEED \
      --save_path "$SAVE_PATH"
  done
done
