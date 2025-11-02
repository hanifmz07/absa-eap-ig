#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=6

# Extract language parameters from the command line arguments
# Usage: ./sft_circuit1.sh <language> <dataset_folder>
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

TOPK_CIRCUIT="$3"
# Validate the CIRCUIT_CSV_PATH argument if TOPK_CIRCUIT is
if [ -z "$TOPK_CIRCUIT" ]; then
    echo "Error: CIRCUIT_CSV_PATH must be specified for sft_circuit.sh."
    exit 1 # Exit with a non-zero status to indicate an error
fi

# Seeds for the SFT process
SEED=777

# Define the log file names for clarity
LOG_BASE_NAME="sft_circuit"
LOG_DIR="logs"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}/${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}/${PID}_$(date).err"
# Create necessary directories for logs
mkdir -p "${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_FOLDER}/${LANGUAGE}/seed_${SEED}"

# --- Step 1: Initialize Log Files ---
# Clear previous logs and add a timestamp to mark the start of this run.
# The '>' operator truncates the file to zero before writing.
echo "========================================================" > "$STDOUT_LOG"
echo "Starting Targeted SFT script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

# We do the same for the error log, in case of any setup errors.
>"$STDERR_LOG"

# --- Step 2: Group the entire process and apply redirection ---
# The curly braces { ... } group all commands inside.
# The redirection at the end applies to everything within the braces,
# including all the 'echo' statements and every python run.
# We use 'tee -a' to append to our initialized logs.
{
    echo "Running ABSA Targeted SFT"
    
    TOPKS=(5000)
    for TOPK_CIRCUIT in "${TOPKS[@]}"
    do
        echo ""
        echo "--------------------------------------------------------"
        echo "Running targeted sft with seed: $SEED with topk: $TOPK_CIRCUIT"
        echo "--------------------------------------------------------"

        # The output of this python command will now be correctly
        # redirected along with everything else in the loop.
        python run_sft.py \
            --train_json_path "hotel_dataset/${LANGUAGE}/${DATASET_FOLDER}/hotel_aste_train_augmented_noreasoning.json" \
            --model_name "Qwen/Qwen2.5-0.5B" \
            --output_dir "outputs/modelsbestv3.7.4/eap/${DATASET_FOLDER}/circuit-${LANGUAGE}_finetune-${LANGUAGE}/seed_$SEED/aos_sequence_variants/topk_${TOPK_CIRCUIT}" \
            --num_epochs 20 \
            --batch_size 16 \
            --lr 1e-4 \
            --seed $SEED \
            --circuit_csv_path "outputs/multitokensv3.7.4/${DATASET_FOLDER}/${LANGUAGE}/seed_$SEED/aos_circuit_topk-${TOPK_CIRCUIT}.csv" \
            --save_mode "best" \
            --optimizer "AdamW" \
            --val_json_path "hotel_dataset/${LANGUAGE}/${DATASET_FOLDER}/hotel_aste_test_augmented.json" \
            --val_batch_size 16
    done
    echo ""
    echo "========================================================"
    echo "All seeds completed at: $(date)"
    echo "========================================================"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"