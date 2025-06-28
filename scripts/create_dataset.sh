#!/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --output=logs/create_dataset_%j.out
#SBATCH --error=logs/create_dataset_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com

SEEDS=(42 123 2024 31415 777)

for SEED in "${SEEDS[@]}"
do
    echo "================================================================"
    echo "Starting dataset generation for seed $SEED"
    echo "================================================================"

    MODEL_PARENT_DIR="outputs/models/eap/indo/seed_${SEED}/aos_sequence_variants"

    # Get the latest full finetuned model directory with full training data
    FINETUNED_MODEL=$(find "$MODEL_PARENT_DIR" -mindepth 1 -maxdepth 1 -type d \
        ! -name "*topk*" ! -name "*_n[0-9]*" \
        -exec test -f "{}/model_config.pkl" \; -print | sort -r | head -n 1)

    if [ -z "$FINETUNED_MODEL" ]; then
        echo "No valid model found for seed $SEED"
        echo "\n----------------------------------------------------------------\n"
        continue
    fi

    echo "Using model: $FINETUNED_MODEL"

    python run_create_dataset.py \
        --finetuned_model "$FINETUNED_MODEL" \
        --dataset_path "hotel_dataset/filled_counterfacts/formatted_indo_counterfacts.csv" \
        --filtered_data_path "hotel_dataset/indo/seed_${SEED}/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_filtered_AOS.csv" \
        --full_aos_path "hotel_dataset/indo/seed_${SEED}/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_full_AOS.csv" \
        --sequence_variants_path "hotel_dataset/indo/seed_${SEED}/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_AOS_sequence_variants.csv" \
        --eap_output_path "eap_dataset/indo/seed_${SEED}/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_eap_dataset_AOS_sequence_variants.csv"

    if [ $? -eq 0 ]; then
        echo "Dataset creation finished for seed $SEED"
    else
        echo "Dataset creation failed for seed $SEED"
    fi

    echo "\n----------------------------------------------------------------\n"
done
