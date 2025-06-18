#!/bin/bash

# Define the log file names for clarity
LOG_BASE_NAME="sft_circuit"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}_${PID}_$(date).err"

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

echo "========================================================" > "$STDOUT_LOG"
echo "Starting SFT Circuit script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
  SEEDS=(42)
  SAMPLE_SIZES=(100 500 1000 5000 7500 10000 15000)

  for SEED in "${SEEDS[@]}"
  do
    for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"
    do
      echo "Running with sample size: $SAMPLE_SIZE and seed: $SEED"

      # Only run train_full_model if sample_size is not 15000
      if [ "$SAMPLE_SIZE" -ne 15000 ]; then
        python run_sft.py \
          --train_json_path "hotel_dataset/indo/hotel_aste_train_augmented_noreasoning.json" \
          --model_name "Qwen/Qwen2.5-0.5B" \
          --output_dir "outputs/models/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants" \
          --num_epochs 20 \
          --batch_size 16 \
          --lr 1e-4 \
          --seed $SEED \
          --train_full_model \
          --sample_size $SAMPLE_SIZE
      fi

      # Run with circuit-based training
      for TOPK in 1000 2000 5000
      do
        python run_sft.py \
          --train_json_path "hotel_dataset/indo/hotel_aste_train_augmented_noreasoning.json" \
          --model_name "Qwen/Qwen2.5-0.5B" \
          --output_dir "outputs/models/eap/circuit-indo_finetune-indo/seed_$SEED/aos_sequence_variants" \
          --num_epochs 20 \
          --batch_size 16 \
          --lr 1e-4 \
          --seed $SEED \
          --circuit_csv_path ""outputs/multitokens/seed_$SEED"/aos_circuit_topk-${TOPK}.csv" \
          --sample_size $SAMPLE_SIZE
      done
    done
  done
} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"