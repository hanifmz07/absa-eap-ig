#!/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --output=logs/create_dataset_%j.out
#SBATCH --error=logs/create_dataset_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00

SEEDS=(42 123 2024 31415 777)

for SEED in "${SEEDS[@]}"
do
    echo "🚀 Running dataset generation for seed $SEED..."

    python run_create_dataset.py \
        --finetuned_model "outputs/models/eap/indo/seed_$SEED/2025-06-03 09:56:42.802285_tflens_hotel_aste_train_augmented_noreasoning_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20" \
        --dataset_path "hotel_dataset/indo/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_correctedv2.csv" \
        --filtered_data_path "hotel_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_filtered_AOS.csv" \
        --full_aos_path "hotel_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_full_AOS.csv" \
        --sequence_variants_path "hotel_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_AOS_sequence_variants.csv" \
        --eap_output_path "eap_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_eap_dataset_AOS_sequence_variants.csv"

done
