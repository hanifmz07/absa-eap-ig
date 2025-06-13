import argparse
import pandas as pd
from functools import partial
import os, csv

from eap.graph import Graph
from eap.evaluate import evaluate_baseline_multitoken, evaluate_graph_multitoken
from eap.attribute import attribute
from src.metric import logit_diff
from src.utils import load_finetuned_model, load_finetuned_model_lens_from_dir, edge_merging


def log_results_csv(log_path, element, metric, total_edges, baseline_score, top_k, score, faithfulness):
    file_exists = os.path.isfile(log_path)
    edge_percentage = top_k/total_edges
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["element", "metric", "total_edges", "baseline_score", "top_k", "edge_percentage", "circuit_score", "faithfulness", ])
        writer.writerow([element, metric, total_edges, f"{baseline_score:.4f}", top_k, f"{edge_percentage:.1%}", f"{score:.4f}", f"{faithfulness:.4%}", ])

def main(args):
    print("Loading fine-tuned model...")
    # model = load_finetuned_model(args.base_model, args.finetuned_model)
    print(f"Loading model from {args.finetuned_model}")
    model = load_finetuned_model_lens_from_dir(args.finetuned_model, args.device)
    model.to(args.device)
    model.cfg.device = args.device
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    print("Loading dataset...")
    df = pd.read_csv(args.dataset)

    print("Initializing graph from model...")
    graph = Graph.from_model(model)

    print("Evaluating baseline performance (no intervention)...")
    baseline = evaluate_baseline_multitoken(
        model=model,
        df=df,
        metrics=[logit_diff],
        run_corrupted=False,
        batch_size=args.batch_size,
        quiet=False
    )
    print(f"Baseline logit_diff = {baseline:.4f} \n")

    print("⚡ Running attribution with EAP-IG...")
    attribute(
        model=model,
        graph=graph,
        dataloader=df,
        metric=partial(logit_diff, loss=False, mean=True),
        method="EAP-IG-inputs",
        ig_steps=args.ig_steps,
        is_absa=True,
        batch_size=args.batch_size
    )
    print("Attribution scores computed.\n")

    n_edges = graph.real_edge_mask.sum().item()
    print(f"Total real edges: {n_edges}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, args.log_file) or os.path.join(args.output_dir, "faithfulness_log.csv")

    for top_k in args.topks:
        print(f"\nEvaluating circuit with top-k = {top_k} edges...")
        graph.reset()
        graph.apply_topn(top_k, True)

        results = evaluate_graph_multitoken(
            model=model,
            graph=graph,
            df=df,
            metrics=[partial(logit_diff, mean=True, loss=False)],
            batch_size=args.batch_size
        )

        faithfulness = results / baseline
        print(f"Top-{top_k} ({top_k/n_edges:.2%}) logit_diff = {results:.4f} → faithfulness = {faithfulness:.1%}")
        print(f"Included nodes: {graph.count_included_nodes()}, edges: {graph.count_included_edges()}")

        output_path = f"{args.output_dir}/{args.element}_circuit_topk-{top_k}.pt"
        graph.to_pt(output_path)
        print(f"Saved circuit to {output_path}")

        log_results_csv(log_file, args.element, "logit_diff", n_edges, baseline, top_k, results, faithfulness)

        if args.element == "aos":
            complete_edges = edge_merging(graph_paths=f"{args.output_dir}/{args.element}_circuit_topk-{top_k}.pt")
            complete_edges.to_csv(f"{args.output_dir}/{args.element}_circuit_topk-{top_k}.csv", index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EAP-IG circuit discovery for ABSA task")

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base Hugging Face model name")
    parser.add_argument("--finetuned_model", type=str, default="models/fine_tuned_model/", help="Path to fine-tuned model")
    parser.add_argument("--dataset", type=str, default="eap_dataset/eap_dataset_aspect_multitokens.csv", help="Path to dataset CSV")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--ig_steps", type=int, default=5, help="Number of IG steps for EAP-IG")
    parser.add_argument("--topks", type=int, nargs="+", default=[100, 200, 500, 1000, 2000, 5000, 10000, 20000], help="Top-k values to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save circuits")
    parser.add_argument("--device", type=str, default="mps", help="Device to run model on: 'cuda', 'mps', or 'cpu'")
    parser.add_argument("--element", type=str, default="aspect", choices=["aspect", "opinion", "sentiment", "aos"], help="Which ABSA element to attribute")
    parser.add_argument("--log_file", type=str, default=None, help="CSV log file path (default: <output_dir>/faithfulness_log.csv)")

    args = parser.parse_args()
    main(args)
