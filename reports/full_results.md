# Full experiment: PaliGemma 2 on Renaissance paintings

Run date: 2026-10-04

## Research question

Does the 448-pixel PaliGemma 2 checkpoint answer questions about small visual
details in Renaissance paintings more accurately than the 224-pixel checkpoint?
If so, is the difference caused by extra detail in the input, or by differences
between the two pretrained checkpoints?

This is an exploratory evaluation, not a benchmark-quality estimate of general
visual-question-answering performance.

## Design

- 12 public-domain paintings: four famous, four moderately familiar, and four
  relatively obscure works.
- Five questions per painting, for 60 unique image-question pairs.
- 24 distant-detail, 10 hand-held-attribute, two inscription, 12
  secondary-figure, and 12 false-premise questions.
- Three conditions and 180 answers in total.
- Greedy deterministic decoding with a 48-token response limit.
- Apple M4 Max with 36 GB unified memory, PyTorch MPS, and float16.

| Condition | Checkpoint | Input supplied |
|---|---|---|
| **224 native** | google/paligemma2-3b-mix-224 | Original image processed at 224×224 |
| **448 native** | google/paligemma2-3b-mix-448 | Original image processed at 448×448 |
| **448 degraded** | google/paligemma2-3b-mix-448 | Downsampled to 224×224, then enlarged to 448×448 |

The most useful comparison is 448 native versus 448 degraded: the checkpoint is
fixed, so the main manipulated variable is visual detail in the input. Comparing
448 native with 224 native is practically useful but confounded, because the
checkpoints also differ in their high-resolution training.

## Blind grading

All 180 outputs were shuffled with seed 50 and assigned anonymous IDs. The scoring
sheet contained the painting ID, question, reference answer, and response, but no
condition, checkpoint, or question ID. A grader who had not seen the raw
condition-labelled output assigned:

- correctness: 0 (incorrect), 1 (partly correct), or 2 (correct);
- hallucination: 1 when the answer asserted incorrect visual evidence, otherwise 0;
- false-premise rejection: 1 when the response rejected or declined the premise,
  otherwise 0.

The hidden key was joined only after all 180 rows had been graded and validated.

## Results

### Overall

| Condition | Mean correctness (0–2) | Hallucination rate | False premises rejected |
|---|---:|---:|---:|
| 224 native | 0.717 | 58.3% | 1/12 (8.3%) |
| 448 degraded | 0.933 | 46.7% | 5/12 (41.7%) |
| 448 native | 0.983 | 43.3% | 5/12 (41.7%) |

### Paired comparison on the same questions

| Comparison | Mean-score difference | A wins | Ties | B wins |
|---|---:|---:|---:|---:|
| 448 native (A) vs 448 degraded (B) | +0.050 | 5 | 50 | 5 |
| 448 native (A) vs 224 native (B) | +0.267 | 13 | 43 | 4 |
| 448 degraded (A) vs 224 native (B) | +0.217 | 11 | 48 | 1 |

The native 448 input had a slightly higher average score than its degraded
counterpart, but the paired outcomes do not show a consistent win: each condition
won five questions, and 50 of 60 tied. False-premise rejection was identical at
5/12. This run therefore does **not** provide convincing evidence that additional
input pixels alone improved performance.

Both 448-checkpoint conditions performed better than 224 native, even after the
448 input was stripped of high-resolution detail. The degraded condition retained
most of the correctness advantage (+0.217) and exactly the same false-premise
advantage as native 448 (5/12 versus 1/12). The most plausible reading is that the
checkpoint and its additional training matter more here than raw input resolution
alone. This is an inference from the controlled degradation comparison, not proof
of a specific causal mechanism.

### By question type

| Type | n per condition | 224 native | 448 degraded | 448 native |
|---|---:|---:|---:|---:|
| Distant detail | 24 | 1.292 | 1.375 | **1.583** |
| Hand-held attribute | 10 | **0.700** | 0.600 | 0.500 |
| Inscription | 2 | 0.000 | 0.000 | 0.000 |
| Secondary figure | 12 | 0.333 | **0.750** | 0.583 |
| False premise | 12 | 0.083 | 0.667 | **0.750** |

There is a local signal for native 448 on distant details, but it does not
generalize across all small-detail categories. Higher resolution did not help the
hand-held attributes in this sample, and none of the checkpoints read either
inscription correctly. The inscription result is too small to generalize, but it
is a useful failure case.

### Familiarity groups

| Condition | Famous | Moderate | Obscure |
|---|---:|---:|---:|
| 224 native | 0.75 | 0.80 | 0.60 |
| 448 degraded | 0.90 | 1.05 | 0.85 |
| 448 native | 0.75 | 1.05 | **1.15** |

The 448-native checkpoint scored highest on the obscure group rather than the
famous group. That is inconsistent with a simple story in which correct answers
mainly come from memorizing famous paintings. With only four works per group,
however, this is a diagnostic observation rather than strong evidence about
memorization.

## What persuaded me

The degradation control is the most persuasive part of the experiment. A plain
224-versus-448 comparison could tempt us to credit extra pixels for every
improvement. The degraded 448 condition shows why that conclusion would be too
quick: much of the advantage survives when the additional visual detail is
removed. The model checkpoint and training recipe are part of the result.

The false-premise questions were also unusually informative. They exposed a
behavior that ordinary accuracy questions hide: the model often gives a fluent,
specific answer to an object that is not present. Even the better conditions
rejected only five of twelve false premises.

## What did not persuade me

This run does not establish a broad “higher resolution sees finer details” claim.
The native-versus-degraded comparison was nearly tied overall, and effects changed
direction by question type. The two inscription probes were failed by every
condition. More pixels cannot be assumed to fix grounding automatically.

## Limitations and next experiment

- Twelve paintings and one blind grader support an exploratory blog experiment,
  not a definitive benchmark.
- Paintings and questions were curated rather than randomly sampled.
- Fame categories are subjective and contain only four works each.
- Only two inscription questions were included.
- The 224 and 448 checkpoints cannot isolate resolution because their training
  histories differ.
- A single deterministic generation was evaluated for each pair.
- Downsampling and re-enlarging approximates lost detail but is not identical to
  the 224 checkpoint's learned representation.

A stronger follow-up would add more inscriptions and hand-held attributes, use
two independent blind graders, report agreement, and repeat the native/degraded
comparison over a larger random sample.

## Painting sources

The image manifest stores both museum and image-source URLs. Official records
used for verification include the [Uffizi](https://www.uffizi.it/en/artworks/birth-of-venus),
[Cenacolo Vinciano](https://cenacolovinciano.org/en/museum/the-works/the-last-supper-leonardo-da-vinci-1452-1519/),
[Louvre](https://collections.louvre.fr/en/ark:/53355/cl010064382),
[National Gallery](https://www.nationalgallery.org.uk/paintings/bronzino-an-allegory-with-venus-and-cupid),
[Art Institute of Chicago](https://www.artic.edu/artworks/80530/virgin-and-child-with-an-angel),
and [National Gallery of Art](https://www.nga.gov/artworks/1138-feast-gods).
