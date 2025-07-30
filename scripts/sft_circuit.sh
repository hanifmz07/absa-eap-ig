#!/bin/bash
#SBATCH --job-name=sft_circuit
#SBATCH --output=logs/sft_circuit_%j.out
#SBATCH --error=logs/sft_circuit_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@example.com  # Change to your email

echo "Running ABSA SFT with Circuit-Based Training"

SEEDS=(9584 123 2024 31415 777)
TOPKS=(1000 2000 5000)

for SEED in "${SEEDS[@]}"
do
    echo "==========================================="
    echo "Running circuit-based SFT for seed: $SEED"
    echo "==========================================="

    for TOPK in "${TOPKS[@]}"
    do
        echo "Top-k: $TOPK"

        python run_sft.py \
          --train_json_path "hotel_dataset/indo/hotel_aste_train_augmented_noreasoning.json" \
          --model_name "Qwen/Qwen2.5-0.5B" \
          --output_dir "outputs/models/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants" \
          --num_epochs 20 \
          --batch_size 16 \
          --lr 1e-4 \
          --seed $SEED \
          --circuit_csv_path "outputs/multitokens/seed_$SEED/aos_circuit_topk-${TOPK}.csv"

        echo -e "\n--- Finished seed $SEED | topk $TOPK ---\n"
    done

    echo "\n----------------------------------------------------------------\n"
done
