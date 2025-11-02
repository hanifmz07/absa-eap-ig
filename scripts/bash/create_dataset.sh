#!/bin/bash
# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=7

# Extract language parameters from the command line arguments
# Usage: ./create_dataset.sh <language> <dataset_folder>
LANGUAGE="$1"
if [ -z "$LANGUAGE" ]; then
    echo "Error: No language specified."
    exit 1
fi
# Validate the language argument
if [[ "$LANGUAGE" != "indo" && "$LANGUAGE" != "eng" && "$LANGUAGE" != "sunda" ]]; then
    echo "Error: Invalid language specified. Use 'indo', 'eng', or 'sunda'."
    exit 1
fi

DATASET_FOLDER="$2"
# Validate the dataset folder argument
if [ -z "$DATASET_FOLDER" ]; then
    echo "Error: Dataset folder must be specified. Name a folder located in the hotel_dataset/{lang} directory."
    exit 1 # Exit with a non-zero status to indicate an error
fi


# Define the log file names for clarity
LOG_BASE_NAME="create_dataset"
LOG_DIR="logs"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/${PID}_$(date).err"
# Create necessary directories for logs
mkdir -p "${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}"

echo "========================================================" > "$STDOUT_LOG"
echo "Starting Create Dataset script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
    SEEDS=(9584 123 2024 31415 777)
    for SEED in "${SEEDS[@]}"
    do
        echo ""
        echo "--------------------------------------------------------"
        echo "🚀 Running dataset generation for seed $SEED..."
        echo "--------------------------------------------------------"
        MODEL_PATH=$(echo "outputs/modelsbest_adamw/eap/${DATASET_FOLDER}/circuit-${LANGUAGE}_finetune-${LANGUAGE}/seed_$SEED/aos_sequence_variants/full_sft/"*"_tflens_hotel_aste_train_augmented_noreasoning_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20")
        
        python run_create_dataset.py \
            --finetuned_model "$MODEL_PATH" \
            --dataset_path "hotel_dataset/counterfactsv3.7.4/${DATASET_FOLDER}/${LANGUAGE}_counterfacts.csv" \
            --filtered_data_path "eap_dataset/filtered_datav3.7.4/${LANGUAGE}/${DATASET_FOLDER}/seed_$SEED/tflens_Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_counterfactual_filtered_AOS.csv" \
            --full_aos_path "eap_dataset/full_aosv3.7.4/${LANGUAGE}/${DATASET_FOLDER}/seed_$SEED/tflens_Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_counterfactual_full_AOS.csv" \
            --sequence_variants_path "eap_dataset/sequence_variantsv3.7.4/${LANGUAGE}/${DATASET_FOLDER}/seed_$SEED/tflens_Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_counterfactual_AOS_sequence_variants.csv" \
            --eap_output_path "eap_dataset/eap_outputv3.7.4/${LANGUAGE}/${DATASET_FOLDER}/seed_$SEED/tflens_Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_eap_dataset_AOS_sequence_variants.csv" \
            --method "mvp"

    done

    echo ""
    echo "========================================================"
    echo "All seeds completed at: $(date)"
    echo "========================================================"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"