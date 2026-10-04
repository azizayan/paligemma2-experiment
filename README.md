# PaliGemma 2 Renaissance-painting experiment

This repository contains the controlled experiment and final figures used in a
blog post about [PaliGemma 2](https://arxiv.org/abs/2412.03555).

This is a public, read-only research archive. External issues, pull requests,
code contributions, and content submissions are not accepted. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

The experiment asks whether the 448-pixel PaliGemma 2 checkpoint answers
questions about small details in Renaissance paintings more accurately because
it receives more visual detail, or because the checkpoint and its training differ
from the 224-pixel model.

## Main result

We evaluated 60 questions about 12 paintings under three conditions (180 model
answers), then graded the randomized answers without condition labels.

| Condition | Mean correctness (0–2) | Hallucination rate | False premises rejected |
|---|---:|---:|---:|
| 224 native | 0.717 | 58.3% | 1/12 |
| 448 degraded | 0.933 | 46.7% | 5/12 |
| 448 native | 0.983 | 43.3% | 5/12 |

The native 448 input did not consistently beat the deliberately degraded 448
input: each won five questions and 50 of 60 tied. Most of the gap over the 224
condition therefore survived after fine visual detail was removed. In this small
sample, checkpoint/training differences appear more important than extra input
pixels alone. This is exploratory evidence, not a causal or benchmark-quality
result.

The complete design, breakdowns, interpretation, and limitations are in
[`reports/full_results.md`](reports/full_results.md).

## Conditions

| Condition | Checkpoint | Image supplied |
|---|---|---|
| `224_native` | `google/paligemma2-3b-mix-224` | Original image processed at 224×224 |
| `448_native` | `google/paligemma2-3b-mix-448` | Original image processed at 448×448 |
| `448_degraded` | `google/paligemma2-3b-mix-448` | Downsampled to 224×224, then enlarged to 448×448 |

`448_native` versus `448_degraded` holds the checkpoint fixed and is the cleanest
test of available visual detail. `448_native` versus `224_native` is a practical
checkpoint comparison, but it does not isolate resolution because the 448 model
also received additional high-resolution training.

## Repository contents

```text
data/
  paintings.csv               12-painting manifest and source links
  questions.jsonl             60 detail and false-premise questions
  model_costs.csv              compute figures transcribed from the paper
  outputs/                     raw answers, blind grades, and summaries
reports/
  full_results.md              complete experiment report
  figures/                     four figures used in the blog
scripts/
  run_zero_shot.py             deterministic three-condition inference
  anonymize_outputs.py         randomized blind-grading sheet generator
  summarize_scores.py         aggregate and paired results
  validate_dataset.py         experiment-design checks
src/paligemma2_experiment/     data and image utilities
tests/                         fast unit tests
```

The final raw outputs and grading files are included for auditability. Painting
files are not distributed: `data/paintings.csv` records the museum and image
source for every work, and the expected local filename under `data/images/`.
Review each source's terms before downloading or redistributing an image.

## Reproduce the analysis

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[test]"
python scripts/validate_dataset.py
python scripts/summarize_scores.py
python -m pytest
```

The summary command rebuilds the published CSV tables from the included blind
scores and hidden condition key. The grading shuffle seed was 50.

## Rerun model inference

PaliGemma weights are gated. Accept the model terms on Hugging Face and log in
without placing a token in this repository:

```bash
python -m pip install -e ".[ml,test]"
hf auth login
```

Download the 12 source images listed in `data/paintings.csv`, save them under the
specified `data/images/` paths, and run:

```bash
python scripts/run_zero_shot.py \
  --output data/outputs/reproduction.jsonl \
  --max-new-tokens 48
```

Results are appended after every question, so an interrupted run resumes without
repeating completed items. Decoding is greedy (`do_sample=False`). Hardware,
library versions, and model revisions can still affect exact reproducibility.

To generate a new randomized grading sheet without overwriting the published
one:

```bash
python scripts/anonymize_outputs.py \
  --input data/outputs/reproduction.jsonl \
  --scores data/outputs/reproduction_blind_scores.csv \
  --key data/outputs/reproduction_blind_key.csv \
  --seed 50
```

## Figures used in the blog

- `figure1_architecture.png`: image-to-LLM pipeline.
- `figure2_siglip.png`: SigLIP image–text matching.
- `bronzino_representative_comparison.png`: representative output from this
  experiment.
- `menu_3b_vs_10b_comparison.png`: a reader-facing annotation of outputs already
  published by Hugging Face. It is not an additional run from this repository.

Sources and reuse notes are collected in
[`THIRD_PARTY_SOURCES.md`](THIRD_PARTY_SOURCES.md).

## Scope

This public version intentionally excludes exploratory smoke tests, the abandoned
fine-tuning prototype, local environments, browser logs, notebooks, document
exports, and figures that were not used in the final blog.
