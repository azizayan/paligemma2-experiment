#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paligemma2_experiment.io import (
    read_csv,
    read_jsonl,
    validate_full_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paintings", default="data/paintings.csv")
    parser.add_argument("--questions", default="data/questions.jsonl")
    args = parser.parse_args()

    paintings = read_csv(args.paintings)
    questions = read_jsonl(args.questions)
    validate_full_experiment(paintings, questions)
    print(
        f"Valid full experiment: {len(paintings)} paintings, "
        f"{len(questions)} questions"
    )


if __name__ == "__main__":
    main()
