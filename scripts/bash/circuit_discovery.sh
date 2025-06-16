#!/bin/bash
# Define the log file names for clarity
LOG_BASE_NAME="circuit_discovery"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).err"

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

echo "========================================================" > "$STDOUT_LOG"
echo "Starting Circuit Discovery script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
    SEEDS=(42 123 2024 31415 777)

    for SEED in "${SEEDS[@]}"
    do
        echo ""
        echo "--------------------------------------------------------"
        echo "Running circuit discover with seed $SEED"
        echo "--------------------------------------------------------"
        
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
} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"
