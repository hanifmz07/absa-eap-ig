# ABSA Circuit Fine-Tuning and Evaluation Pipeline

This repository contains a complete pipeline for Aspect-Based Sentiment Analysis (ABSA), including full model fine-tuning, circuit discovery, selective fine-tuning, and evaluation.

## Pipeline Overview

All scripts loop over a fixed list of random seeds to ensure robust and reproducible results. You can safely modify these scripts to run in parallel (e.g., background jobs with &, SLURM arrays, or GNU parallel), depending on your compute setup. Please make sure to keep all job logs, as we will use them to track training loss, total training time, and samples processed per second for performance analysis.

For a map of which modules, classes and functions implement each step — plus draft
class diagrams in draw.io format — see [`docs/PIPELINE_CODE_MAP.md`](docs/PIPELINE_CODE_MAP.md)
and [`docs/diagrams/`](docs/diagrams/).

## Step-by-Step Instructions

### 1. Full Fine-Tuning

Run full model fine-tuning across different seeds:

```bash
sbatch scripts/sft.sh
```

### 2. Dataset Creation

Filter predictions, generate simultaneous AOS counterfactuals, and create AOS sequence variants:

```bash
sbatch scripts/create_dataset.sh
```

### 3. Circuit Discovery

Run circuit discovery using EAP-IG for AOS elements:

```bash
sbatch scripts/circuit_discovery.sh
```

### 4. Selective Circuit-Based Fine-Tuning

Apply targeted fine-tuning using circuit masking (attention heads and MLP layers) across different seeds and training sizes:

```bash
sbatch scripts/sft_circuit.sh
```

### 5. Evaluation

Evaluate all models against the test set:

```bash
sbatch scripts/eval.sh
```

## Running the Scripts: SLURM vs Local

All job scripts (`*.sh`) in this project are written for SLURM using `sbatch`.

To run them on a SLURM-based cluster:
```bash
sbatch scripts/job.sh
```

If you want to run the script locally using bash (for debugging or testing small parts), you can do so, but SLURM-specific directives (e.g., `#SBATCH --gres=gpu:1`) will be ignored.

If you wish to run them locally (e.g., via `bash scripts/script_name.sh`), be aware:

- SLURM directives will be ignored
- You must manually ensure appropriate environment setup
- GPU/CPU/memory configuration will need to be managed manually
- Output will not go to logs/ unless you redirect it yourself

or you can use the scripts on the `scripts/bash` to manually configure the GPU and enable logs (`scripts/bash/*.sh`).

## Folder Structure Convention

All model and evaluation results follow this naming structure:

```
outputs/models/eap/circuit-{LANG}_finetune-{LANG}/seed_{SEED}/aos_sequence_variants/{MODEL_FOLDER}
outputs/evals/eap/circuit-{LANG}_finetune-{LANG}/seed_{SEED}/aos_sequence_variants/{MODEL_FOLDER}
```

- `{LANG}` is the language code (e.g., `indo`, `eng`, `sunda`)
- `{SEED}` is one of: `42`, `123`, `2024`, `31415`, `777`
- `{MODEL_FOLDER}` automatically includes sample size and/or circuit top-k

This structure ensures consistency across experiments, making it easier to evaluate and aggregate results later.

## Notes

- All training and evaluation scripts automatically skip completed outputs (based on folder presence).
- Evaluation is based on structured prompts using `[A] [O] [S]` markers and counterfactuals.
- Logging is handled per SLURM job via:
  - `logs/eval_%j.err`
  - etc.

