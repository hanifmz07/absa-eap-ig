#!/bin/bash

# ======== Job Configuration ========
#SBATCH --job-name=absa_eap
#SBATCH --output=logs/%x_%j.out       # Save stdout to logs/ folder with job name and ID
#SBATCH --error=logs/%x_%j.err        # Save stderr to logs/ folder
#SBATCH --time=12:00:00               # Max run time
#SBATCH --partition=gpu               # Choose the appropriate partition (gpu/cpu)
#SBATCH --gres=gpu:3                  # Number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ======== Environment Setup ========
echo "Starting job on $(hostname)"

# ======== Run Script ========
echo "Running ABSA EAP-IG Evaluation on Aspect"

python run_eap_multitokens.py \
  --base_model "Qwen/Qwen2.5-0.5B" \
  --finetuned_model "models/fine_tuned_model/" \
  --dataset "eap_dataset/eap_dataset_aspect_multitokens.csv" \
  --output_dir "outputs/multitokens" \
  --batch_size 16 \
  --ig_steps 5 \
  --topks 100 200 500 1000 2000 5000 10000 20000 30000 40000 \
  --device "cuda" \
  --element "aspect" \
  --log_file "multitokens_faithfulness_log.csv"

echo "Running ABSA EAP-IG Evaluation on Opinion"

python run_eap_multitokens.py \
  --base-model "Qwen/Qwen2.5-0.5B" \
  --finetuned_model "models/fine_tuned_model/" \
  --dataset "eap_dataset/eap_dataset_opinion_multitokens.csv" \
  --output_dir "outputs/multitokens" \
  --batch_size 16 \
  --ig_steps 5 \
  --topks 100 200 500 1000 2000 5000 10000 20000 30000 40000 \
  --device "cuda" \
  --element "opinion" \
  --log_file "multitokens_faithfulness_log.csv"

echo "Running ABSA EAP-IG Evaluation on Sentiment"

python run_eap_multitokens.py \
  --base-model "Qwen/Qwen2.5-0.5B" \
  --finetuned_model "models/fine_tuned_model/" \
  --base_model "Qwen/Qwen2.5-0.5B" \
  --dataset "eap_dataset/eap_dataset_sentiment_multitokens.csv" \
  --output_dir "outputs/multitokens" \
  --batch_size 16 \
  --ig-steps 5 \
  --topks 100 200 500 1000 2000 5000 10000 20000 30000 40000 \
  --device "cuda" \
  --element "sentiment" \
  --log_file "multitokens_faithfulness_log.csv"

echo "Job completed"