#!/bin/bash
# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=3

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

SEED=31415
# Define the log file names for clarity
LOG_BASE_NAME="circuit_discovery"
LOG_DIR="logs"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}/${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}/${PID}_$(date).err"
# Create necessary directories for logs
mkdir -p "${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}"

echo "========================================================" > "$STDOUT_LOG"
echo "Starting Circuit Discovery script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
    echo ""
    echo "--------------------------------------------------------"
    echo "Running circuit discover with seed $SEED"
    echo "--------------------------------------------------------"

    MODEL_PATH=$(echo "outputs/models/eap/${DATASET_FOLDER}/circuit-${LANGUAGE}_finetune-${LANGUAGE}/seed_$SEED/aos_sequence_variants/"*"_tflens_hotel_aste_train_augmented_noreasoning_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20")
    
    python run_eap_multitokens.py \
        --base_model "Qwen/Qwen2.5-0.5B" \
        --finetuned_model "$MODEL_PATH" \
        --dataset "eap_dataset/eap_output/${LANGUAGE}/${DATASET_FOLDER}/seed_$SEED/tflens_Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_eap_dataset_AOS_sequence_variants.csv" \
        --batch_size 5 \
        --ig_steps 5 \
        --topks 1000 2000 5000 \
        --output_dir "outputs/multitokens/${DATASET_FOLDER}/${LANGUAGE}/seed_$SEED" \
        --device "cuda" \
        --element "aos"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"
