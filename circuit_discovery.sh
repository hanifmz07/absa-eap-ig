#!/bin/bash
#SBATCH --job-name=circuit_discovery
#SBATCH --output=logs/circuit_discovery_%j.out
#SBATCH --error=logs/circuit_discovery_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00


SEEDS=(42 123 2024 31415 777)

for SEED in "${SEEDS[@]}"
do
    echo "Running with seed $SEED"
    python run_eap_multitokens.py \
        --base_model "Qwen/Qwen2.5-0.5B" \
        --finetuned_model "outputs/models/eap/indo/seed_$SEED/2025-06-03 09:56:42.802285_tflens_hotel_aste_train_augmented_noreasoning_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20" \
        --dataset "eap_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_eap_dataset_AOS_sequence_variants.csv" \
        --batch_size 5 \
        --ig_steps 5 \
        --topks 1000 2000 5000 \
        --output_dir "outputs/multitokens/seed_$SEED" \
        --device "cuda" \
        --element "aos"
done

