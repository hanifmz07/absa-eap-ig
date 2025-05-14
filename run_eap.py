import argparse
import ast
import os
from functools import partial
from random import random
from typing import Optional

import pandas as pd
import transformers
from torch.utils.data import Dataset, DataLoader
import torch
from typing_extensions import Tuple

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from src.utils import build_eap_dataset
from src import load_finetuned_model, filter_correct_data


def safe_parse(raw):
    try:
        return ast.literal_eval(raw)[0]  # unbox the list-of-list
        # return ast.literal_eval(raw)[0][0]  # unbox the list-of-list
    except Exception as e:
        raise ValueError(f"Failed to parse: {raw}\n{e}")


class EAPDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clean = row["clean"]
        corrupted = row["corrupted"]
        correct_idx = safe_parse(row["correct_idx"])
        incorrect_idx = safe_parse(row["incorrect_idx"])
        return clean, corrupted, [correct_idx, incorrect_idx]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)

def collate_EAP(batch):
    clean, corrupted, labels = zip(*batch)

    correct_idx_batch = [torch.tensor(l[0], dtype=torch.long) for l in labels]
    incorrect_idx_batch = [torch.tensor(l[1], dtype=torch.long) for l in labels]

    return list(clean), list(corrupted), (correct_idx_batch, incorrect_idx_batch)


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor, labels: Optional[Tuple[torch.Tensor, torch.Tensor]]):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    if labels is None:
        logits_batch = logits[idx, input_length - 1]
    else: # if multi-token labels, then need to return List[logits of shape (n_tokens, vocab)]
        logits_batch = []
        for b in range(batch_size):
            l = logits[b, input_length[b]-len(labels[0][b]): input_length[b]]
            logits_batch.append(l)
    return logits_batch


def prob_diff_multitoken(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean=True,
    loss=False
):
    logits = get_logit_positions(logits, input_length, labels)
    # logits is a list of tensors, can't batch process
    # probs = torch.softmax(logits, dim=-1)
    probs = [torch.softmax(l, dim=-1) for l in logits]

    # labels is a tuple (bs) of torch tensors (# objs)
    results = []
    for b, label in enumerate(labels[0]):
        # (i.e. query box contains multiple objs, label would be [#objs])
        result = probs[b][range(len(label)), label].prod()
        results.append(result)
    results = torch.stack(results)

    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results



def filter_dataset():
    base_model_name = "Qwen/Qwen2.5-0.5B"
    fine_tuned_model_path = "models/fine_tuned_model/"
    model = load_finetuned_model(base_model_name, fine_tuned_model_path)
    model.to("mps")
    model.eval()

    # first filter by success (this needs to be done for A, O, S individually)
    # dataset_path = "hotel_dataset/hotel_aste_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_counterfactual_corrected.csv"
    # filtered_data_path = "hotel_dataset/hotel_aste_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_counterfactual_filtered.csv"
    # test_data = pd.read_csv(dataset_path)
    #
    # filtered_data = filter_correct_data(
    #     model, test_data,
    #     "original_sentence", "original_triplet",
    #     save_path=filtered_data_path)

    # then filter by length and format to eap ig dataset format
    filtered_data = pd.read_csv(
        "hotel_dataset/hotel_aste_model-Qwen2.5-0.5B_lr-0.0001_bs-16_epochs-20_counterfactual_filtered.csv")

    eap_df = build_eap_dataset(
        model=model,
        df=filtered_data,
        sentence_col="original_sentence",
        triplet_col="original_triplet",
        corrupted_col="counterfact1_modified",
        corrupted_triplet_col="counterfact_triplet1_modified",
        suffix="[A]",
        idx=0,  # 0 for aspect, 1 for opinion, 2 for sentiment,
        filer_same_length_counterfactuals=True
    )
    eap_df.to_csv("eap_dataset/eap_dataset_aspect.csv", index=False)


def main():
    # load model
    base_model_name = "Qwen/Qwen2.5-0.5B"
    fine_tuned_model_path = "models/fine_tuned_model/"
    model = load_finetuned_model(base_model_name, fine_tuned_model_path)
    model.to("mps")
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    # load dataset
    ds = EAPDataset("eap_dataset/eap_dataset_aspect.csv")
    dataloader = ds.to_dataloader(batch_size=10)

    baseline = evaluate_baseline(model, dataloader, partial(prob_diff_multitoken, loss=False, mean=False)).mean().item()
    print(f"Original performance is prob_diff={baseline}")

    # Instantiate a graph with a model
    g = Graph.from_model(model)

    attribute(
        model,
        g,
        dataloader,
        partial(prob_diff_multitoken, loss=True, mean=True),
        method='EAP-IG-inputs',
        ig_steps=5,
    )

    os.makedirs("outputs", exist_ok=True)
    n_edges = g.real_edge_mask.sum().item()  # total 171K edges for qwen2.5-0.5B
    for topk in [100, 200, 500, 1000, 2000, 5000, 10000]:
        g.apply_topn(topk, True)
        results = evaluate_graph(model, g, dataloader, partial(prob_diff_multitoken, loss=False, mean=False)).mean().item()

        print(f"with top-k = {topk} ({topk/n_edges:.1%}), the circuit's performance is {results}, faithfulness={results/baseline:.1%}")
        g.to_pt(f'outputs/aspect_circuit_topk-{topk}.pt')

        if topk <= 500:
            gz = g.to_graphviz(f'outputs/aspect_circuit_topk-{topk}.png')
        print(f"included nodes: {g.count_included_nodes()}, included edges: {g.count_included_edges()}")

if __name__== "__main__":
    filter_dataset()
    main()