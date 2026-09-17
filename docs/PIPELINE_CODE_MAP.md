# Pipeline Code Map & Class Diagram Notes

This document maps every part of the repository onto the five pipeline steps, and
describes the draft class diagrams in [`docs/diagrams/`](diagrams/).

## Contents

- [How to read the diagrams](#how-to-read-the-diagrams)
- [Execution order](#execution-order)
- [Step 1 — Dataset Creation](#step-1--dataset-creation)
- [Step 2 — Full SFT](#step-2--full-sft)
- [Step 3 — Circuit Discovery](#step-3--circuit-discovery)
- [Step 4 — Selective Circuit-Based SFT](#step-4--selective-circuit-based-sft)
- [Step 5 — Evaluation](#step-5--evaluation)
- [Code not owned by any of the five steps](#code-not-owned-by-any-of-the-five-steps)
- [Inconsistencies found while mapping](#inconsistencies-found-while-mapping)

---

## How to read the diagrams

Files in `docs/diagrams/`:

| File | Contents |
| --- | --- |
| `00_pipeline_overview.drawio` | All five steps, their entry points and the artifacts they hand to each other |
| `01_dataset_creation.drawio` | Step 1 |
| `02_full_sft.drawio` | Step 2 |
| `03_circuit_discovery.drawio` | Step 3 |
| `04_selective_circuit_sft.drawio` | Step 4 |
| `05_evaluation.drawio` | Step 5 |
| `absa_eap_class_diagrams.drawio` | All six of the above as pages of one file |
| `generate_drawio.py` | The script that produced them |

Open any of them with **File → Open** in [app.diagrams.net](https://app.diagrams.net)
or the draw.io desktop app. Re-running `python docs/diagrams/generate_drawio.py`
regenerates all seven files and **overwrites hand edits**, so once you start
editing in draw.io, treat the script as historical.

### Modelling convention

This codebase is mostly function-based rather than class-based: only a handful of
real classes exist (`Graph`, `Node` and its subclasses, `Edge`, `GraphConfig`,
`EAPDataset`, `ABSAAutoRegressiveDataset`, `HookedTransformerTrainConfig`). A
pure class diagram would therefore show almost nothing.

The diagrams use the standard UML *utility class* treatment: a box stereotyped
`«module»` is a Python module (or a coherent group of functions within one), and
its module-level functions are drawn as the box's operations. Real classes are
stereotyped `«class»` or `«dataclass»` and carry genuine attributes and methods.
Where one module serves several steps, it appears on several pages, each time
showing only the operations that step uses — e.g. `src.utils` is split into
"MVP / AOS pipeline", "triplet helpers", "model loading", "circuit masking" and
"scoring" boxes.

Colours: yellow = SLURM/bash job script · orange = Python entry point ·
blue = `src/` helper module · purple = `eap/` circuit package · green = on-disk
artifact · grey = third-party class.

Arrows: dashed open = `«use»` (calls/imports) · solid green = data flow
(reads/writes an artifact) · hollow triangle = generalization · filled diamond =
composition.

---

## Execution order

The README lists Full SFT first and the user-facing step list starts with Dataset
Creation. The real dependency order is:

```
Full SFT ──▶ Dataset Creation ──▶ Circuit Discovery ──▶ Selective Circuit SFT ──▶ Evaluation
   │               │                      │                      │                    │
   │               └── needs the fine-tuned model to filter on ───┘                    │
   └── also supplies the model that attribution runs through, and the full-SFT baseline┘
```

`scripts/create_dataset.sh` and `scripts/circuit_discovery.sh` both locate the
full-SFT run by globbing `outputs/models/eap/indo/seed_<SEED>/aos_sequence_variants`
for a directory whose name contains neither `topk` nor `_n<digits>` — that is how
they tell a full run from a circuit run or a sub-sampled run. The run-folder
naming in `run_sft.py` is therefore load-bearing, not cosmetic.

---

## Step 1 — Dataset Creation

**Diagram:** `01_dataset_creation.drawio` · **Driver:** `scripts/create_dataset.sh`
· **Entry point:** `run_create_dataset.py`

Produces the EAP-IG dataset: pairs of a clean prompt and a corrupted
(counterfactual) prompt that tokenize to the same length, plus the correct and
incorrect answer token ids.

### Entry point

| Symbol | File | Role |
| --- | --- | --- |
| `main(args)` | `run_create_dataset.py` | Orchestrates steps 0–4; branches on `--method` |
| CLI args | `run_create_dataset.py` | `--finetuned_model`, `--dataset_path`, `--filtered_data_path`, `--full_aos_path`, `--sequence_variants_path`, `--eap_output_path`, `--method`, `--filter_data` |

### MVP / AOS pipeline (`--method mvp` or `mvp_aos`)

| Function | File | Step |
| --- | --- | --- |
| `format_counterfactuals(input_path)` | `src/utils.py:1006` | 0 — splits legacy `original_pair` / `corrupted_pair` columns on `[A] [O] [S]` |
| `filter_correct_data(model, data, sentence_col, label_col, max_tokens, filter_only_correct, filter_mode, save_path)` | `src/utils.py:576` | 1 — generates with the fine-tuned model and keeps only rows it already gets right |
| `create_full_AOS_dataset(dataset_path)` | `src/utils.py:897` | 2 — splices the aspect of counterfactual 1 into counterfactual 3, giving `counterfact4_replaced` (a simultaneous A+O+S counterfactual) |
| `create_aos_sequence_variant(dataset_path)` | `src/utils.py:959` | 3 — emits one row per tag order |
| `create_sequences(a, o, s)` | `src/utils.py:939` | 3 — the five orders: `AOS`, `ASO`, `SAO`, `OAS`, `OSA` |
| `build_eap_dataset(model, df, …, suffix)` | `src/utils.py:701` | 4 — tokenizes, drops pairs whose clean/corrupted lengths differ, drops label pairs whose token counts differ |

`--method mvp_aos` runs the same code and then keeps only `order == 'AOS'`.

### GAS pipeline (`--method gas` or `gas_bar`)

Steps 2 and 3 are skipped; the whole triplet string is the label.

| Function | File |
| --- | --- |
| `format_counterfactuals_gas(input_path, col_original, col_counter, index_col_name, coerce_index_to_int)` | `src/utils.py:1045` |
| `filter_correct_data_gas(model, data, sentence_col, label_col, max_tokens, filter_only_correct, save_path)` | `src/utils.py:1150` |
| `build_eap_dataset_gas(model, df, …)` | `src/utils.py:1221` |
| `_extract_first_triplet(text)`, `_normalize_triplet_str(s)` | `src/utils.py:1130`, `:1141` / `:1208` |

### Shared helpers

`convert_triplet_string` (`:494`), `extract_triplet_fixed` (`:562`),
`get_tag_suffix` (`:689`), `format_by_mode` (`:509`), `build_suffix_from_mode`
(`:530`), `extract_by_mode` (`:545`), `load_finetuned_model_lens_from_dir` (`:147`).

`append_labels` (`:670`) is defined here for multi-token label prefixing but is
currently unused — the call site inside `build_eap_dataset` is commented out.

### Artifacts

```
hotel_dataset/filled_counterfacts/formatted_indo_counterfacts.csv
  └─ filter_correct_data     → hotel_dataset/<lang>/seed_<s>/…_counterfactual_filtered_AOS.csv
       └─ create_full_AOS_dataset       → …_counterfactual_full_AOS.csv
            └─ create_aos_sequence_variant → …_counterfactual_AOS_sequence_variants.csv
                 └─ build_eap_dataset        → eap_dataset/<lang>/seed_<s>/…_eap_dataset_AOS_sequence_variants.csv
```

The final CSV has columns `clean`, `corrupted`, `correct_label`,
`incorrect_label`, `correct_idx`, `incorrect_idx`.

---

## Step 2 — Full SFT

**Diagram:** `02_full_sft.drawio` · **Driver:** `scripts/sft.sh`
· **Entry point:** `run_sft.py --train_full_model`

### Entry point

`main(args)` in `run_sft.py`: optionally sub-samples the training set, loads the
model, builds the dataset, builds a `HookedTransformerTrainConfig`, calls `train`,
then reports throughput and saves.

### Model loading — `src/utils.py`

| Symbol | Line | Role |
| --- | --- | --- |
| `get_cfg_dict(base_model_name, hf_config)` | `:25` | HF config → TransformerLens config dict; supports Qwen2 and Bloom only |
| `load_model(base_model_name, fine_tuned_model_path, device)` | `:81` | Dispatcher: base model or fine-tuned |
| `load_finetuned_model(base_model_name, fine_tuned_model_path, device)` | `:109` | Loads an HF checkpoint and converts weights via `convert_qwen2_weights` / `convert_bloom_weights` |
| `load_finetuned_model_lens_from_dir(dir, device)` | `:147` | Reloads a run produced by this pipeline (`model.pt` + `model_config.pkl`) |

### Data and training

| Symbol | File | Role |
| --- | --- | --- |
| `ABSAAutoRegressiveDataset(Dataset)` | `src/utils.py:278` | Concatenates `input + " " + target + eos`, pads/truncates to `max_len=300`, forces a trailing EOS |
| `HookedTransformerTrainConfig` (dataclass) | `src/train.py:20` | All hyperparameters plus `top_k` / `sample_size` / `subtract_data_amount`, which exist purely for W&B logging |
| `train(model, config, dataset, val_dataset)` | `src/train.py:69` | Adam / AdamW / SGD, optional LambdaLR warm-up, `save_mode='best'` on epoch loss, per-epoch or per-N-step validation, W&B logging |

Saving: with `save_mode=None` (the default used by `scripts/sft.sh`) the entry
point writes `model.pt`, `model_config.pkl` and the tokenizer *after* training;
with `save_mode='best'`, `train` writes `model.pt` whenever epoch loss improves,
and the config/tokenizer up front.

### Optional stratified sub-sampling — `src/sampling.py`

Only runs when `--sample_size` is given. MVP and GAS have parallel implementations.

| MVP | GAS | Role |
| --- | --- | --- |
| `add_element_order_to_json` (`:16`) | `add_element_order_to_json_gas` (`:446`) | Derives `element_order` from the trailing `[A][O][S]` markers |
| `annotate_difficulty` (`:105`) | `annotate_difficulty_gas` (`:512`) | Per-item F1 against a previous inference file, then an easy/medium/hard bin |
| `get_difficulty_assigner` (`:52`) | `get_difficulty_assigner_gas` (`:468`) | Quantile / fixed / hybrid thresholds |
| `stratified_sample_by_factors` (`:358`) | `stratified_sample_by_factors_gas` (`:761`) | Proportional sampling over the strata |
| `extract_stratification_key` (`:192`) | `extract_stratification_key_gas` (`:599`) | Builds the stratum key from the chosen factors |
| `get_triplet_count`, `get_input_length_bin`, `get_sentiment_bin`, `get_sentiment_ratio_bin` | `*_gas` twins | The individual factors |
| `visualize_strata_heatmap_from_meta`, `visualize_strata_heatmap_from_meta_combo_xy`, `visualize_strata_side_by_side` | `*_gas` twins | Only when `plot=True` |

`annotate_difficulty` scores items with `src.utils.calculate_metrics`, the same
function Step 5 uses.

`--subtract_data_by <json>` removes every `sentence_id` present in that file from
the training set (used for the data-subtraction ablations).

---

## Step 3 — Circuit Discovery

**Diagram:** `03_circuit_discovery.drawio` · **Driver:** `scripts/circuit_discovery.sh`
· **Entry point:** `run_eap_multitokens.py`

EAP-IG: score every candidate edge of the model's computational graph by how much
it moves `logit_diff` between the clean and corrupted prompt, keep the top-k, then
measure the kept sub-graph's faithfulness.

### Entry point

| Symbol | File | Role |
| --- | --- | --- |
| `main(args)` | `run_eap_multitokens.py` | Sets the four required `model.cfg` flags, builds the graph, measures the baseline, attributes, then loops over `--topks` |
| `log_results_csv(...)` | `run_eap_multitokens.py:13` | Appends one row per top-k to the faithfulness log |

`main` sets `use_split_qkv_input`, `use_attn_result`, `use_hook_mlp_in` and
`ungroup_grouped_query_attention` to `True` before attribution — `attribute()`
asserts all four, because the per-head hooks have nothing to attach to otherwise.

### Graph model — `eap/graph.py`

| Class | Line | Role |
| --- | --- | --- |
| `Node` | `:13` | One graph node; `in_graph` / `score` / `neurons` are properties that read and write the parent `Graph`'s tensors rather than local state |
| `InputNode` | `:126` | `'input'`, `hook_embed` |
| `AttentionNode` | `:118` | `'a<L>.h<H>'`, with separate q/k/v input hooks |
| `MLPNode` | `:112` | `'m<L>'` |
| `LogitNode` | `:98` | `'logits'`; always in the graph |
| `Edge` | `:132` | `'<parent>-><child><qkv>'`; also proxies `score` / `in_graph` onto the `Graph` tensors |
| `GraphConfig(dict)` | `:188` | `n_layers`, `n_heads`, `d_model`, `parallel_attn_mlp` |
| `Graph` | `:193` | Owns `nodes`, `edges` and the score/mask tensors |

Key `Graph` operations: `from_model` / `from_pt` / `from_json` (factories),
`add_edge`, `forward_index` / `backward_index` / `prev_index` (the tensor indexing
scheme), `apply_topn` / `apply_threshold` / `apply_greedy` (circuit selection),
`prune`, `reset`, `count_included_edges` / `count_included_nodes` /
`count_included_neurons`, `to_pt` / `to_json` / `to_graphviz`.

Scores live in a single `[n_forward, n_backward]` tensor; `Node` and `Edge` are
views onto it. `real_edge_mask` marks which of those cells are actually reachable
edges — `graph.real_edge_mask.sum()` is the "total real edges" the entry point
prints (~171 K for Qwen2.5-0.5B).

### Attribution — `eap/attribute.py`

| Function | Line | Role |
| --- | --- | --- |
| `attribute(model, graph, dataloader, metric, method, …, is_absa, batch_size, device)` | `:598` | Dispatcher; writes the result into `graph.scores` |
| `modified_get_scores_eap_ig(model, graph, df, metric, batch_size, steps, quiet, device)` | `:362` | **The ABSA path.** Groups rows by label token length, then attributes one target token at a time, appending the already-correct prefix to the clean prompt and the incorrect prefix to the corrupted one |
| `get_scores_eap_ig` | `:279` | The stock non-ABSA EAP-IG |
| `get_scores_eap` | `:224` | Plain EAP (no integrated gradients) |
| `get_scores_ig_activations` | `:475` | `EAP-IG-activations` |
| `get_scores_clean_corrupted` | `:545` | `clean-corrupted` |
| `make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)` | `:55` | Builds the forward/backward hooks and the `[batch, pos, n_forward, d_model]` activation-difference buffer; contains `activation_hook` and `gradient_hook` |
| `tokenize_plus(model, inputs, max_length)` | `:29` | → `tokens, attention_mask, input_lengths, n_pos` |
| `compute_mean_activations(model, graph, dataloader, per_position)` | `:154` | For the `mean` / `mean-positional` interventions |

The pipeline calls `attribute(..., method='EAP-IG-inputs', is_absa=True,
ig_steps=5)`, which routes to `modified_get_scores_eap_ig`.

`eap/attribute_node.py` mirrors all of the above at node and neuron level
(`attribute_node`, `:350`). No pipeline script currently calls it.

### Circuit evaluation — `eap/evaluate.py`

| Function | Line | Role |
| --- | --- | --- |
| `evaluate_baseline_multitoken(model, df, metrics, run_corrupted, quiet, batch_size)` | `:445` | Unpatched score, token step by token step |
| `evaluate_graph_multitoken(model, graph, df, metrics, batch_size, intervention, …)` | `:218` | Rebuilds each node's input from in-graph edges only, patching everything else to its corrupted activation; contains `make_input_construction_hook(s)` |
| `evaluate_baseline`, `evaluate_graph` | `:406`, `:22` | Single-token originals, used by `run_eap.py` |

Faithfulness = `evaluate_graph_multitoken(...) / evaluate_baseline_multitoken(...)`.

### Metric — `src/metric.py`

`logit_diff(logits, clean_logits, input_length, labels, mean, loss)` returns
`logit(correct) − logit(incorrect)` at the answer position;
`get_logit_positions(logits, input_length)` selects that position.
The entry point passes `partial(logit_diff, loss=False, mean=True)`.

### Data plumbing — `src/utils.py`

`EAPDataset` (`:849`) wraps the EAP CSV, `collate_EAP` (`:832`) batches it,
`safe_parse` (`:808`) turns the stringified token-id lists back into lists, and
`edge_merging(graph_paths)` (`:874`) walks a saved `Graph` and emits the in-graph
edges as a DataFrame of `parent_node`, `child_node`, `child_type` — **this CSV is
the hand-off to Step 4.** It is only written when `--element aos`.

`eap/visualization.py` (`get_color`, `generate_random_color`, `cmap`, `color`)
serves `Graph.to_graphviz`.

### Artifacts

```
outputs/multitokens/seed_<s>/
  aos_circuit_topk-<k>.pt    serialized Graph (cfg, scores, in-graph masks)
  aos_circuit_topk-<k>.csv   parent_node, child_node, child_type
  <log_file>.csv             element, metric, total_edges, baseline_score,
                             top_k, edge_percentage, circuit_score, faithfulness
```

`run_eap.py` is the earlier single-element prototype — its own local `EAPDataset`,
`prob_diff_multitoken`, hard-coded paths and `mps` device. It is superseded by
`run_eap_multitokens.py` and is not called by any job script.

---

## Step 4 — Selective Circuit-Based SFT

**Diagram:** `04_selective_circuit_sft.drawio` · **Driver:** `scripts/sft_circuit.sh`
· **Entry point:** `run_sft.py --circuit_csv_path …` (i.e. without `--train_full_model`)

Same training loop as Step 2, but gradients outside the circuit are zeroed.

### Circuit masking — `src/utils.py`

| Symbol | Line | Role |
| --- | --- | --- |
| `apply_active_edge_unfreezing(model, csv_path, random_circuit, sample_n, random_state, strict, sample_like_topk)` | `:372` | Reads the circuit CSV (or samples a random circuit), parses `child_node` into `(layer, head, q\|k\|v)` and `m<L>` into MLP layers, freezes all buffers, then registers the gradient masks |
| `register_head_mask(weight_tensor, active_heads)` | `:449` (nested) | Builds a 0/1 mask and attaches `mask_hook` via `Tensor.register_hook` |
| `_derive_like_path(path, like_topk)` | `:381` (nested) | Rewrites `topk-<N>` in a path, for `--random_circuit_sample_like_topk` |
| `_count_unique_pairs_from_csv(path)` | `:384` (nested) | Counts unique `(child_node, child_type)` rows, so a random circuit can match a real circuit's size |
| `get_random_nodes(model, df, include_mlp, include_logits, sample_n, random_state, strict)` | `:313` | The random-circuit control: enumerates every `a<L>.h<H>` × `{q,k,v}`, anti-joins against the real circuit, then samples. `strict=True` also excludes whole heads that appear in the real circuit under any projection |

**What is actually frozen.** No parameter is detached. The whole model still
receives gradients; `register_head_mask` multiplies the incoming gradient of
`W_Q`/`W_K`/`W_V` (per layer, per projection) and `W_O` (per layer, for every head
that appears in any q/k/v role) by a per-head 0/1 mask, so inactive heads get a
zero update. Embeddings, layer norms, MLP weights and the unembedding are **not**
masked and keep training. `mlp_layers` is parsed and printed but never turned into
a mask — worth knowing before you interpret a "circuit-only" result.

### Reused from Step 2

`ABSAAutoRegressiveDataset`, `HookedTransformerTrainConfig`, `src.train.train`,
`load_model` / `load_finetuned_model_lens_from_dir`, and the whole of
`src.sampling` (when `--sample_size` is set) are unchanged.

`top_k` for the output folder name and the W&B run name is parsed out of the
circuit CSV filename (`…_topk-2000.csv` → `topk-2000`), so the filename convention
from Step 3 matters here.

### Artifacts

Input: `outputs/multitokens/seed_<s>/aos_circuit_topk-<k>.csv` and the training JSON.
Output: `outputs/models/eap/circuit-<lang>_finetune-<lang>/seed_<s>/aos_sequence_variants/<run>_topk-<k>/`
containing `model.pt`, `model_config.pkl` and tokenizer files, plus a W&B run in
project `absa-eap`.

---

## Step 5 — Evaluation

**Diagram:** `05_evaluation.drawio` · **Driver:** `scripts/eval.sh`
· **Entry point:** `run_eval.py`

### Entry point

`main(args)` in `run_eval.py`:

1. `load_finetuned_model_lens_from_dir(args.model_path)`, pick device, `model.eval()`
2. Batched greedy generation (`max_new_tokens=300`, `stop_at_eos=True`,
   `do_sample=False`, `padding_side='left'`)
3. Strip the prompt prefix from each generation
4. Split target and prediction into triplet lists — on `" [SSEP] "` for
   `--prompt_type mvp`, on `";"` for `--prompt_type gas`
5. Bucket by `element_order` and score each bucket

### Scoring — `src/utils.py`

| Function | Line | Role |
| --- | --- | --- |
| `calculate_metrics(predictions, targets, task)` | `:165` | Micro TP/FP/FN over triplet strings → `precision_<task>`, `recall_<task>`, `f1_<task>` |
| `parse_absa_string(text)` | `:196` | `"[A] a [O] o [S] s [SSEP] …"` → `[{A,O,S}, …]` |
| `postprocess_absa_outputs(preds, labels, sentence_id, task)` | `:229` | MvP-style majority vote across the permutations of one sentence — used by `run_eval_franken.py`, not by `run_eval.py` |

Results are scaled ×100 and written per model directory:

```
outputs/evals/…/<model_folder>/
  raw_inference_results.json   list of decoded generations (always written)
  evaluation_results.json      precision_<order>, recall_<order>, f1_<order>
  inference_results.json       full per-item records (only with --save_predictions)
```

Circuit **faithfulness** is measured in Step 3, not here; this step measures task
quality. The diagram cross-references `eap.evaluate` for that reason.

`run_eval_franken.py` is the variant for Franken-adapter checkpoints: it loads the
base Qwen model and a raw `state_dict` rather than a pipeline run directory.

---

## Code not owned by any of the five steps

These are side experiments, kept out of the five diagrams:

| Path | What it is |
| --- | --- |
| `franken_adapter/franken.py`, `franken_adapter/utils.py`, `run_franken_2a.py`, `run_franken_3b.py`, `run_eval_franken.py`, `scripts/franken/` | Franken-adapter experiments: `instruction_tuning`, `embedding_training`, `merge_instruction_with_embedding`, `finetune_franken_adapter`, plus `SundaEmbeddingDataset` and the Sundanese corpus filters |
| `lego_absa/data.py`, `lego_absa/train.py` | LEGO-ABSA multi-task baseline (HF `Seq2SeqTrainer`) |
| `zhang_gas/data.py`, `zhang_gas/train.py` | Zhang GAS baseline |
| `run_sft_t5.py` | T5 fine-tuning variant |
| `data.py` | Two-line HF `Dataset` loader used by the baselines |
| `scripts/job.sh` | Older per-element (`aspect` / `opinion` / `sentiment`) circuit discovery job |
| `scripts/bash/`, `scripts/bash/parallel_*` | Local/non-SLURM and parallel variants of the five job scripts |
| `analysis.ipynb`, `eap_demo.ipynb`, `baseline_demo.ipynb`, `check_counterfact.ipynb`, `create_dataset_debug.ipynb`, `franken_adapter_demo.ipynb`, `utils_notebooks/` | Exploration and result aggregation |

---

## Inconsistencies found while mapping

Three things that surfaced while tracing the call graphs. None are fixed here —
they are listed so the diagrams are not read as an endorsement of the current state.

1. **`run_create_dataset.py:155` — `NameError` in the GAS path without `--filter_data`.**
   The `else` branch at `:145` assigns `df_filtered`, but the `build_eap_dataset_gas`
   call at `:155` passes `filtered_df`, which is only bound inside the
   `if args.filter_data:` branch. GAS without `--filter_data` crashes.

2. **`scripts/eval.sh` never passes `--prompt_type`,** which `run_eval.py` declares
   `required=True`. As written, the job script fails at argument parsing for every
   model. `scripts/bash/eval.sh` has the same gap.

3. **`src/utils.py:1090` uses `np.inf` / `np.nan` but `numpy` is never imported**
   in that module. The statement sits inside a `try: … except Exception: pass`, so
   the `NameError` is swallowed and `format_counterfactuals_gas` silently skips the
   integer coercion of its index column rather than crashing.
