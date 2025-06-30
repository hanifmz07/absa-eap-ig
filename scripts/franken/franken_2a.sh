#!/bin/bash
#SBATCH --job-name=franken_2a
#SBATCH --output=logs/franken_2a_%j.out
#SBATCH --error=logs/franken_2a_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com

SEEDS=(42 123 2024 31415 777)

for SEED in "${SEEDS[@]}"
do
    echo "==========================================="
    echo "Running instruction tuning for seed $SEED"
    echo "==========================================="

    python run_franken_2a.py \
        --model_name "Qwen/Qwen2.5-0.5B" \
        --json_path "hotel_dataset/indo/clean_train2500/hotel_aste_train_augmented_noreasoning.json" \
        --output_dir "outputs/models/franken/2a/seed_${SEED}" \
        --batch_size 16 \
        --lr 1e-4 \
        --num_epochs 20 \
        --device "cuda" \
        --seed "$SEED"

    if [ $? -eq 0 ]; then
        echo "[DONE] Instruction Tuning - Seed $SEED"
    else
        echo "[FAILED] Instruction Tuning - Seed $SEED"
    fi

    echo "\n----------------------------------------------------------------\n"
done
