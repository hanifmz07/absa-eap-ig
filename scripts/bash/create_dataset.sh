#!/bin/bash
# Define the log file names for clarity
LOG_BASE_NAME="create_dataset"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).err"

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

echo "========================================================" > "$STDOUT_LOG"
echo "Starting Create Dataset script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
    SEEDS=(42 123 2024 31415 777)

    for SEED in "${SEEDS[@]}"
    do
        echo ""
        echo "--------------------------------------------------------"
        echo "🚀 Running dataset generation for seed $SEED..."
        echo "--------------------------------------------------------"

        python run_create_dataset.py \
            --finetuned_model outputs/models/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants/*_tflens_hotel_aste_train_augmented_noreasoning_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20 \
            --dataset_path "hotel_dataset/counterfacts/hotel_aste_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_counterfactual.csv" \
            --filtered_data_path "hotel_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_filtered_AOS.csv" \
            --full_aos_path "hotel_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_full_AOS.csv" \
            --sequence_variants_path "hotel_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_counterfactual_AOS_sequence_variants.csv" \
            --eap_output_path "eap_dataset/indo/seed_$SEED/hotel_aste_model-Qwen2.5-0.0001_bs-16_epochs-20_eap_dataset_AOS_sequence_variants.csv"

    done

    echo ""
    echo "========================================================"
    echo "All seeds completed at: $(date)"
    echo "========================================================"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"