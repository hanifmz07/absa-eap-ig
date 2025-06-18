#!/bin/bash
#SBATCH --job-name=circuit_discovery
#SBATCH --output=logs/circuit_discovery_%j.out
#SBATCH --error=logs/circuit_discovery_%j.err
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com  # SBATCH-level mail for failed job start

SEEDS=(42 123 2024 31415 777)

for SEED in "${SEEDS[@]}"
do
    echo "==========================================="
    echo "Running circuit discovery for seed $SEED"
    echo "==========================================="

    MODEL_PARENT_DIR="outputs/models/eap/indo/seed_${SEED}/aos_sequence_variants"

    # Get the latest full finetuned model directory with full training data
    FINETUNED_MODEL=$(find "$MODEL_PARENT_DIR" -maxdepth 1 -type d \
        ! -name "*topk*" ! -name "*_n*" \
        | sort -r | head -n 1)

    if [ -z "$FINETUNED_MODEL" ]; then
        echo "No valid full model directory found for seed $SEED. Skipping..."
        echo "\n----------------------------------------------------------------\n"
        continue
    fi

    echo "Using model: $FINETUNED_MODEL"

    python run_eap_multitokens.py \
        --base_model "Qwen/Qwen2.5-0.5B" \
        --finetuned_model "$FINETUNED_MODEL" \
        --dataset "eap_dataset/indo/seed_${SEED}/hotel_aste_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_eap_dataset_AOS_sequence_variants.csv" \
        --batch_size 5 \
        --ig_steps 5 \
        --topks 1000 2000 5000 \
        --output_dir "outputs/multitokens/seed_${SEED}" \
        --device "cuda" \
        --element "aos"

    if [ $? -eq 0 ]; then
        echo "[DONE] Circuit Discovery - Seed $SEED" 
    else
        echo "[FAILED] Circuit Discovery - Seed $SEED"
    fi

    echo "\n----------------------------------------------------------------\n"
done
