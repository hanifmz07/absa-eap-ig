#!/bin/bash
#SBATCH --job-name=sft_full
#SBATCH --output=logs/sft_full_%j.out
#SBATCH --error=logs/sft_full_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@example.com  # change to your real email

echo "Running ABSA SFT"

SEEDS=(9584 123 2024 31415 777)

for SEED in "${SEEDS[@]}"
do
    echo "==========================================="
    echo "Running full SFT with seed: $SEED"
    echo "==========================================="

    python run_sft.py \
      --train_json_path "hotel_dataset/indo/hotel_aste_train_augmented_noreasoning.json" \
      --model_name "Qwen/Qwen2.5-0.5B" \
      --output_dir "outputs/models/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants" \
      --num_epochs 20 \
      --batch_size 16 \
      --lr 1e-4 \
      --seed $SEED \
      --train_full_model

    echo "\n----------------------------------------------------------------\n"
done
