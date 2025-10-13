from src import load_finetuned_model, load_finetuned_model_lens_from_dir
from src.utils import postprocess_absa_outputs, calculate_metrics
import torch
import json
from tqdm import tqdm
import os, re
import argparse

def main(args):
    # === Load Model ===
    model = load_finetuned_model_lens_from_dir(args.model_path)

    # === Setup Device ===
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    model.eval()

    # === Load Dataset ===
    with open(args.test_json_path, 'r') as f:
        test_data = json.load(f)

    # Get the input prompts, labels, sentence IDs, and task elements
    prompts = [instance['input'] for instance in test_data]
    labels = [instance['target'] for instance in test_data]
    sentence_ids = [instance['sentence_id'] for instance in test_data]
    tasks = [instance['task_elements'] for instance in test_data]
    element_orders = [instance['element_order'] for instance in test_data]

    # --- Batching Modification ---
    batch_size = args.batch_size # You can adjust this based on your GPU memory
    outputs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating outputs"):
        batch_prompts = prompts[i:i + batch_size]

        # Generate outputs for the batch
        batch_outputs_text = model.generate(
            input=batch_prompts,
            max_new_tokens=300,
            stop_at_eos=True,
            do_sample=False,
            return_type="str",
            verbose=False,
            padding_side="left"
        )
        
        if isinstance(batch_outputs_text, str):
            batch_outputs_text = [batch_outputs_text]

        outputs.extend(batch_outputs_text)

    # Cut off the prompts from the outputs
    outputs = [output[len(prompts[idx]):].strip() for idx, output in enumerate(outputs)]

    # Postprocess outputs and calculate metrics
    # Temporary storing for debugging
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, args.model_path.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "raw_inference_results.json")
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)
 
    inference_results = []

    per_task = {}
    for prompt, pred, label, si, t, element_order in zip(prompts, outputs, labels, sentence_ids, tasks, element_orders):
        
        if args.prompt_type == "mvp":
            # Postprocess the prediction for MvP
            # Split the target and prediction into lists
            target_split = label.split(" [SSEP] ")
            pred_split = pred.split(" [SSEP] ")
            # Strip whitespace
            target_split = [l.strip() for l in target_split]
            pred_split = [l.strip() for l in pred_split]
        elif args.prompt_type == "gas":
            # Split the target and prediction into lists (GAS and LegoABSA)
            target_split = label.split(';')
            pred_split = pred.split(';')
            target_split = [l.strip() for l in target_split]
            pred_split = [l.strip() for l in pred_split]

        # Store inference results
        inf_dict = {}
        inf_dict["sentence_id"] = si
        inf_dict["task_elements"] = t
        inf_dict["element_order"] = element_order
        inf_dict["input"] = prompt
        inf_dict["target"] = label
        inf_dict["prediction"] = pred
        inf_dict["target_list"] = target_split
        inf_dict["prediction_list"] = pred_split

        inference_results.append(inf_dict)

        # Store predictions and targets per task for metric calculation
        if element_order not in per_task.keys():
            per_task[element_order] = {"predictions": [], "targets":[]}
        per_task[element_order]["predictions"].append(pred_split)
        per_task[element_order]["targets"].append(target_split)


    result_metrics = {}
    for task, v in per_task.items():
        predictions = v["predictions"]
        targets = v["targets"]
        result_metrics.update(
            calculate_metrics(predictions, targets, task)
    )
    scaled_result_metrics = {key: value * 100 for key, value in result_metrics.items()}
    
    print("Evaluation results:", scaled_result_metrics)

    # Save the metric results
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, args.model_path.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(scaled_result_metrics, f, indent=4)
    print(f"Evaluation results saved to {output_file}")

    # Save inference results if specified
    if args.save_predictions:
        inference_output_file = os.path.join(output_dir, "inference_results.json")
        with open(inference_output_file, "w") as f:
            json.dump(inference_results, f, indent=4)
        print(f"Inference results saved to {inference_output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate models for ABSA task")
    parser.add_argument("--test_json_path", type=str, required=True, help="Path to the test JSON dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained model name")
    parser.add_argument("--prompt_type", type=str, required=True, help="Prompt type for the model, either MvP or GAS.", choices=["mvp", "gas"])
    parser.add_argument("--output_dir", type=str, default=f"./outputs/evals", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--save_predictions", action="store_true", help="Save inference results to a JSON file")

    args = parser.parse_args()
    main(args)
