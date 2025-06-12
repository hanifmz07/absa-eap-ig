# ======== Environment Setup ========
echo "Starting job on $(hostname)"

# ======== Run Script ========
echo "Running ABSA SFT"

# Uncomment the following line to use a specific GPU
# export CUDA_VISIBLE_DEVICES=4

python run_sft.py \
  --train_json_path "hotel_dataset/hotel_aste_train_augmented_noreasoning.json" \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --output_dir "outputs/models" \
  --num_epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --seed 42 \
  --circuit_csv_path "outputs/multitokens/v3/complete_circuit_topk-20000.csv"
  # --train_full_model \
  # --sample_size