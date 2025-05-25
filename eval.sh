# ======== Environment Setup ========
echo "Starting job on $(hostname)"

# ======== Run Script ========
echo "Running ABSA Inference"

# Uncomment the following line to use a specific GPU
# export CUDA_VISIBLE_DEVICES=4

python run_eval.py \
  --test_json_path "hotel_dataset/hotel_aste_test_augmented.json" \
  --model_path "outputs/models/2025-05-22 05:15:40.522365_tflens_hotel_aste_train_augmented_noreasoning_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_topk-20000" \
  --output_dir "outputs/evals/batch_size_1" \
  --batch_size 1 \
  --save_predictions