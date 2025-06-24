import json
import numpy as np
from typing import List, Dict, Tuple
from src.utils import calculate_metrics
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
import re
from pathlib import Path


def add_element_order_to_json(json_path: str, backup: bool = True):
    """
    Adds 'element_order' key to each entry in a JSON file containing ABSA input-output pairs.
    The order is derived from the trailing sequence of [A], [O], [S] in the 'input' field.
    
    Args:
        json_path (str): Path to the JSON file to update.
        backup (bool): If True, create a backup file before overwriting.
    """
    json_path = Path(json_path)

    if backup:
        backup_path = json_path.with_suffix('.backup.json')
        shutil.copyfile(json_path, backup_path)
        print(f"Backup created at {backup_path}")

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Update element order
    for entry in data:
        match = re.search(r"(\[(A|O|S)\]\s*)+$", entry["input"])
        if match:
            elements = re.findall(r"\[([A-Z])\]", match.group(0))
            entry["element_order"] = "".join(e.lower() for e in elements)
        else:
            entry["element_order"] = ""

    # Overwrite the same file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated file saved to {json_path}")


def get_difficulty_assigner(f1_scores, method="quantile", bins=3, fixed_thresholds=None, min_easy_f1=0.8):
    # Define labels
    if bins == 3:
        labels = ["hard", "medium", "easy"]
    elif bins == 4:
        labels = ["very hard", "hard", "medium", "easy"]
    elif bins == 5:
        labels = ["very hard", "hard", "medium", "easy", "very easy"]
    else:
        labels = [f"bin_{i}" for i in range(bins)]

    if method == "fixed":
        assert fixed_thresholds and len(fixed_thresholds) == bins - 1, \
            "Provide fixed_thresholds = list of thresholds between bins"
        def assign(f1):
            if f1 == 1.0:
                return labels[-1]
            for i, t in enumerate(fixed_thresholds):
                if f1 <= t:
                    return labels[i]
            return labels[-1]
        return fixed_thresholds, assign

    elif method == "quantile":
        thresholds = np.quantile(f1_scores, np.linspace(0, 1, bins + 1)[1:-1])
        def assign(f1):
            if f1 == 1.0:
                return labels[-1]
            for i, t in enumerate(thresholds):
                if f1 <= t:
                    return labels[i]
            return labels[-1]
        return thresholds, assign

    elif method == "hybrid":
        assert bins == 3, "Hybrid only supports 3 bins (hard, medium, easy)"
        thresholds = np.quantile(f1_scores, [1/3, 2/3])
        def assign(f1):
            if f1 == 1.0:
                return "easy"
            if f1 <= thresholds[0]:
                return "hard"
            elif f1 <= thresholds[1]:
                return "medium"
            elif f1 >= min_easy_f1:
                return "easy"
            else:
                return "medium"
        return thresholds, assign

    else:
        raise ValueError(f"Unknown method: {method}")

def annotate_difficulty(
    input_path: str,
    output_path: str,
    difficulty_method: str = "quantile",
    bins: int = 3,
    fixed_thresholds: List[float] = None,
    min_easy_f1: float = 1
):
    with open(input_path) as f:
        dataset = json.load(f)

    #compute F1s
    f1_scores = []
    raw_metrics = []
    for item in dataset:
        target = [item["target_list"]]
        pred = [item["prediction_list"]]
        task = item.get("element_order", "")
        metrics = calculate_metrics(predictions=pred, targets=target, task=task)
        f1 = metrics[f"f1_{task}"]
        f1_scores.append(f1)
        raw_metrics.append((item, metrics, f1))

    # Get difficulty assigner
    thresholds, assign_difficulty = get_difficulty_assigner(
        f1_scores,
        method=difficulty_method,
        bins=bins,
        fixed_thresholds=fixed_thresholds,
        min_easy_f1=min_easy_f1
    )

    # Annotate dataset
    annotated = []
    for item, metrics, f1 in raw_metrics:
        task = item.get("element_order", "")
        item["precision"] = round(metrics[f"precision_{task}"], 4)
        item["recall"] = round(metrics[f"recall_{task}"], 4)
        item["f1"] = round(metrics[f"f1_{task}"], 4)
        item["difficulty"] = assign_difficulty(f1)
        annotated.append(item)

    with open(output_path, "w") as f:
        json.dump(annotated, f, indent=2)

    return annotated, thresholds


def get_triplet_count(item):
    return len(item["target_list"]) if "target_list" in item else 1

def get_input_length_bin(text, length_thresholds=(10, 25)):
    length = len(text.split())
    short_cutoff, medium_cutoff = length_thresholds
    if length <= short_cutoff:
        return "short"
    elif length <= medium_cutoff:
        return "medium"
    else:
        return "long"

def get_sentiment_bin(target):
    sentiments = [s for s in target.split() if s in {"positive", "negative"}]
    unique = set(sentiments)
    if len(unique) == 1:
        return list(unique)[0]
    else:
        return "mixed"

def get_sentiment_ratio_bin(item: Dict, binning_scheme: Tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)) -> str:
    sentiments = [s for s in item["target"].split() if s in {"positive", "negative"}]
    total = len(sentiments)
    pos = sentiments.count("positive")
    ratio = pos / total if total > 0 else 0.5  # neutral default if no sentiment

    low, mid_low, mid_high, high = binning_scheme
    if ratio <= low:
        return "mostly negative"
    elif ratio <= mid_low:
        return "somewhat negative"
    elif ratio <= mid_high:
        return "mixed"
    elif ratio <= high:
        return "somewhat positive"
    else:
        return "mostly positive"

def extract_stratification_key(item, factors: List[str], config: dict = {}):
    key_parts = []
    for factor in factors:
        if factor == "triplet_count":
            key_parts.append(get_triplet_count(item))
        elif factor == "input_length_bin":
            thresholds = config.get("input_length_thresholds", (10, 25))
            key_parts.append(get_input_length_bin(item["input"], thresholds))
        elif factor == "sentiment_bin":
            key_parts.append(get_sentiment_bin(item["target"]))
        elif factor == "sentiment_ratio_bin":
            thresholds = config.get("sentiment_ratio_thresholds", (0.2, 0.4, 0.6, 0.8))
            key_parts.append(get_sentiment_ratio_bin(item, thresholds))

        elif factor == "difficulty":
            key_parts.append(item.get("difficulty", "unknown"))
        else:
            raise ValueError(f"Unknown stratification factor: '{factor}'")
    return tuple(key_parts)


def visualize_strata_heatmap_from_meta_combo_xy(sentence_meta, sampled_ids, all_factors, combo_x_factors, combo_y_factors, permutations_per_sentence=5):
    def meta_to_df(meta, source):
        return pd.DataFrame([
            {
                "source": source,
                "combo_x": " | ".join(str(k) for k in key[:len(combo_x_factors)]),
                "combo_y": " | ".join(str(k) for k in key[len(combo_x_factors):])
            }
            for sid, val in meta.items()
            for key in [val["key"]]
            if source == "full" or sid in sampled_ids
        ])

    df_full = meta_to_df(sentence_meta, "full")
    df_sampled = meta_to_df(sentence_meta, "sampled")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    def plot_heatmap(data, ax, title):
        pivot_sentences = data.groupby(["combo_y", "combo_x"]).size().unstack(fill_value=0)
        pivot_actual = pivot_sentences * permutations_per_sentence
        pivot_total = pivot_actual.values.sum()
        pivot_percent = pivot_actual / pivot_total

        sns.heatmap(
            pivot_percent,
            annot=pivot_actual.astype(int),
            fmt="d",
            cmap="YlOrBr",
            ax=ax,
            cbar_kws={'label': 'Percentage of All Instances'}
        )
        ax.set_title(f"{title} (n={pivot_total})")

    plot_heatmap(df_full, axes[0], "Full Dataset")
    plot_heatmap(df_sampled, axes[1], "Sampled Dataset")
    plt.tight_layout()
    plt.show()

def visualize_strata_heatmap_from_meta(sentence_meta, sampled_ids, factors, permutations_per_sentence=5):
    if len(factors) > 3:
        combo_x_factors = factors[:2]
        combo_y_factors = factors[2:]
        return visualize_strata_heatmap_from_meta_combo_xy(
            sentence_meta, sampled_ids, factors,
            combo_x_factors, combo_y_factors,
            permutations_per_sentence
        )

    def meta_to_df(meta, source):
        return pd.DataFrame([
            {"source": source, **{f: k for f, k in zip(factors, key)}}
            for sid, val in meta.items()
            for key in [val["key"]]
            if source == "full" or sid in sampled_ids
        ])

    df_full = meta_to_df(sentence_meta, "full")
    df_sampled = meta_to_df(sentence_meta, "sampled")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    def plot_heatmap(data, ax, title):
        if len(factors) == 2:
            pivot_sentences = data.groupby([factors[1], factors[0]]).size().unstack(fill_value=0)
        else:
            data["combo"] = data[factors[1]] + " | " + data[factors[2]]
            pivot_sentences = data.groupby(["combo", factors[0]]).size().unstack(fill_value=0)

        pivot_actual = pivot_sentences * permutations_per_sentence
        pivot_total = pivot_actual.values.sum()
        pivot_percent = pivot_actual / pivot_total

        sns.heatmap(
            pivot_percent,
            annot=pivot_actual.astype(int),
            fmt="d",
            cmap="YlOrBr",
            ax=ax,
            cbar_kws={'label': 'Percentage of All Instances'}
        )
        ax.set_title(f"{title} (n={pivot_total})")

    plot_heatmap(df_full, axes[0], "Full Dataset")
    plot_heatmap(df_sampled, axes[1], "Sampled Dataset")

    plt.tight_layout()
    plt.show()

def visualize_strata_side_by_side(sentence_meta, sampled_ids, factors, permutations_per_sentence=5):
    def meta_to_df(meta, source):
        return pd.DataFrame([
            {
                "source": source,
                **{f: k for f, k in zip(factors, key)},
                "instances": permutations_per_sentence
            }
            for sid, val in meta.items()
            for key in [val["key"]]
            if source == "full" or sid in sampled_ids
        ])

    df_full = meta_to_df(sentence_meta, "full")
    df_sampled = meta_to_df(sentence_meta, "sampled")
    df_all = pd.concat([df_full, df_sampled], axis=0)

    df_all_expanded = df_all.copy()
    df_all_expanded = df_all_expanded.groupby(factors + ["source"])["instances"].sum().reset_index()
    custom_orders = {
    "difficulty": ["very easy", "easy", "medium", "hard", "very hard"],
    "sentiment_ratio_bin": [
        "mostly negative",
        "somewhat negative",
        "mixed",
        "somewhat positive",
        "mostly positive"
        ]
    }

    if len(factors) == 1:
        g = sns.barplot(data=df_all_expanded, x=factors[0], y="instances", hue="source", palette="Set2", order=custom_orders.get(factors[0]))
        g.set_title("Stratified Distribution Comparison")
        g.set_ylabel("Count (instances)")
        if "sentiment_ratio_bin" in factors:
            plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()
    else:
        g = sns.catplot(
            data=df_all_expanded,
            kind="bar",
            x=factors[0],
            y="instances",
            hue="source",
            col=factors[1],
            palette="Set2",
            col_wrap=3
        )
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels(factors[0], "Count (instances)")
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle("Stratified Distribution Comparison")
        plt.tight_layout()
        plt.show()

def stratified_sample_by_factors(
    full_data_path: str,
    output_path: str,
    factors: List[str],
    target_total_samples: int,
    permutations_per_sentence: int = 5,
    seed: int = 42,
    plot: bool = False,
    config: dict = {}
):
    rng = random.Random(seed)

    with open(full_data_path) as f:
        data = json.load(f)

    sentence_meta = {}
    for d in data:
        sid = d["sentence_id"]
        if sid not in sentence_meta:
            sentence_meta[sid] = {"key": extract_stratification_key(d, factors, config)}

    strata = defaultdict(list)
    for sid, meta in sentence_meta.items():
        strata[meta["key"]].append(sid)

    target_sentence_count = target_total_samples // permutations_per_sentence
    total_sentences = sum(len(sids) for sids in strata.values())

    sampled_ids = []
    for key, sids in strata.items():
        proportion = len(sids) / total_sentences
        n = round(proportion * target_sentence_count)
        sampled = rng.sample(sids, min(n, len(sids)))
        sampled_ids.extend(sampled)

    final_sample = [d for d in data if d["sentence_id"] in sampled_ids]

    with open(output_path, "w") as f:
        json.dump(final_sample, f, indent=2)

    if plot:
        if len(factors) == 1:
            visualize_strata_side_by_side(sentence_meta, sampled_ids, factors, permutations_per_sentence)
        else:
            visualize_strata_heatmap_from_meta(sentence_meta, sampled_ids, factors, permutations_per_sentence)

    return final_sample, strata
