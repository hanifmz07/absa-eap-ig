#!/bin/bash

# Define the log file names for clarity
LOG_BASE_NAME="sft_full"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).err"

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=3

# Extract language parameters from the command line arguments
# Usage: ./sft5.sh <language>
LANGUAGE="$1"
if [ -z "$LANGUAGE" ]; then
    echo "Usage: $0 <language>"
    exit 1
fi
# Validate the language argument
if [[ "$LANGUAGE" != "indo" && "$LANGUAGE" != "eng" && "$LANGUAGE" != "sunda" ]]; then
    echo "Error: Invalid language specified. Use 'indo', 'eng', or 'sunda'."
    exit 1
fi


# --- Step 1: Initialize Log Files ---
# Clear previous logs and add a timestamp to mark the start of this run.
# The '>' operator truncates the file to zero before writing.
echo "========================================================" > "$STDOUT_LOG"
echo "Starting SFT script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

# We do the same for the error log, in case of any setup errors.
>"$STDERR_LOG"

# --- Step 2: Group the entire process and apply redirection ---
# The curly braces { ... } group all commands inside.
# The redirection at the end applies to everything within the braces,
# including all the 'echo' statements and every python run.
# We use 'tee -a' to append to our initialized logs.
{
    echo "Running ABSA SFT"

    SEEDS=(31415)

    for SEED in "${SEEDS[@]}"
    do
        echo ""
        echo "--------------------------------------------------------"
        echo "Running full sft with seed: $SEED"
        echo "--------------------------------------------------------"

        # The output of this python command will now be correctly
        # redirected along with everything else in the loop.
        python run_sft.py \
            --train_json_path "hotel_dataset/${LANGUAGE}/hotel_aste_train_augmented_noreasoning.json" \
            --model_name "Qwen/Qwen2.5-0.5B" \
            --output_dir "outputs/models/eap/circuit-${LANGUAGE}_finetune-${LANGUAGE}/seed_$SEED/aos_sequence_variants" \
            --num_epochs 20 \
            --batch_size 16 \
            --lr 1e-4 \
            --seed $SEED \
            --train_full_model

    done

    echo ""
    echo "========================================================"
    echo "All seeds completed at: $(date)"
    echo "========================================================"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"