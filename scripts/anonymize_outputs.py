#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from paligemma2_experiment.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/outputs/zero_shot.jsonl")
    parser.add_argument("--scores", default="data/outputs/blind_scores.csv")
    parser.add_argument("--key", default="data/outputs/blind_key.csv")
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    random.Random(args.seed).shuffle(rows)

    score_rows = []
    key_rows = []
    for index, row in enumerate(rows, start=1):
        blind_id = f"b{index:04d}"
        score_rows.append(
            {
                "blind_id": blind_id,
                "painting_id": row["painting_id"],
                "question_type": row["question_type"],
                "prompt": row["prompt"],
                "reference_answer": row["reference_answer"],
                "response": row["response"],
                "score_0_to_2": "",
                "hallucination_0_or_1": "",
                "rejects_false_premise_0_or_1": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "condition": row["condition"],
                "model_id": row["model_id"],
                "question_id": row["question_id"],
            }
        )

    Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_csv(args.scores, index=False)
    pd.DataFrame(key_rows).to_csv(args.key, index=False)
    print(f"Wrote {len(rows)} blinded rows to {args.scores}")
    print(f"Keep {args.key} hidden until scoring is finished.")


if __name__ == "__main__":
    main()
