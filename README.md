# ARTS-Hybrid

Anonymous research release for the paper on mechanistic rubric analysis and **Adaptive Rubric Tree Search (ARTS-Hybrid)**.

## Contents

```text
data/
  gold_tree.jsonl                  # 1,500 manually constructed Gold Rubric Trees
  counterfactual_responses.jsonl   # canonical and controlled counterfactual responses
project/
  mechanism_analysis/              # structural, criterion, and aggregation analyses
  scripts/                         # paper-aligned experiment entry points
  rubrics/arts.py                  # ARTS-Hybrid implementation
scripts/
  run_arts_hybrid.py               # minimal ARTS-Hybrid runner
  validate_release_data.py         # released-data integrity check
configs/
  arts_hybrid.example.json         # anonymous endpoint/configuration template
```

The mechanism-analysis code covers tree versus flat rubric generation, controlled counterfactual violation detection, semantic and response-calibrated criterion selection, and aggregation/weighting comparisons. ARTS-Hybrid implements adaptive tree construction, semantic-first feasibility filtering, response-based calibration, global validation and repair, and globally normalized leaf weighting.

## Released data

`gold_tree.jsonl` contains `sample_id` and `gold_tree` for 750 HealthBench and 750 RubricBench instances. `counterfactual_responses.jsonl` contains `sample_id`, `canonical_correct_response`, and three counterfactual types: single-leaf, within-dimension multi-leaf, and cross-dimension multi-leaf. The files share the same 1,500 sample IDs; eligible counterfactual counts are 1,498, 1,482, and 1,409, respectively.

The release intentionally excludes raw benchmark records, private annotations, service endpoints, credentials, and machine-specific paths. Full experiment reruns require obtaining the corresponding HealthBench and RubricBench snapshots from their official sources and supplying the source queries/responses through the command-line arguments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_release_data.py
```

## ARTS-Hybrid

Copy the example configuration, replace endpoint placeholders, and store API keys only in environment variables.

```bash
cp configs/arts_hybrid.example.json configs/arts_hybrid.json
python scripts/run_arts_hybrid.py \
  --query "<user query>" \
  --config configs/arts_hybrid.json \
  --output outputs/rubric.json
```

The default paper configuration uses GLM-4.7-Flash as the rubric generator, two responses from each of Qwen3-30B-A3B-Instruct-2507, Llama-3.1-70B-Instruct, and Mistral-7B-Instruct-v0.2, and Qwen2.5-72B-Instruct plus Llama-3.3-70B-Instruct as calibration judges. 

## Mechanistic analysis

The paper-aligned entry points are:

```bash
python -m project.scripts.run_gold_rubric_mechanisms --help
python -m project.scripts.run_gold_rubric_experiment4 --help
python -m project.scripts.run_rq2_discriminability_stability --help
python -m project.scripts.run_rq2_semantic_first_calibration --help
```

