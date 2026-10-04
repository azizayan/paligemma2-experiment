from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


PAINTING_FIELDS = {
    "painting_id",
    "title",
    "artist",
    "year",
    "fame_group",
    "image_path",
    "source_url",
}
QUESTION_FIELDS = {
    "painting_id",
    "question_id",
    "question_type",
    "prompt",
    "reference_answer",
    "source_url",
}
FULL_EXPERIMENT_QUESTION_TYPES = {
    "distant_detail",
    "handheld_attribute",
    "inscription",
    "secondary_figure",
    "false_premise",
}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_fields(
    rows: Iterable[dict[str, Any]], required: set[str], dataset_name: str
) -> None:
    for index, row in enumerate(rows, start=1):
        missing = required - row.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{dataset_name} row {index} is missing: {names}")


def validate_full_experiment(
    paintings: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> None:
    validate_fields(paintings, PAINTING_FIELDS, "paintings")
    validate_fields(questions, QUESTION_FIELDS, "questions")
    if len(paintings) != 12:
        raise ValueError(f"Full experiment requires 12 paintings; found {len(paintings)}")

    painting_ids = [row["painting_id"] for row in paintings]
    if len(set(painting_ids)) != len(painting_ids):
        raise ValueError("Painting IDs must be unique")

    question_ids = [row["question_id"] for row in questions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Question IDs must be unique")

    counts = {painting_id: 0 for painting_id in painting_ids}
    false_premise_counts = {painting_id: 0 for painting_id in painting_ids}
    for row in questions:
        painting_id = row["painting_id"]
        if painting_id not in counts:
            raise ValueError(f"Question references unknown painting: {painting_id}")
        counts[painting_id] += 1
        if row["question_type"] == "false_premise":
            false_premise_counts[painting_id] += 1

    wrong_counts = {key: value for key, value in counts.items() if value != 5}
    if wrong_counts:
        raise ValueError(f"Each painting requires five questions: {wrong_counts}")

    missing_false = [
        painting_id
        for painting_id, count in false_premise_counts.items()
        if count != 1
    ]
    if missing_false:
        raise ValueError(
            "Each painting requires exactly one false-premise question: "
            f"{missing_false}"
        )

    observed_types = {row["question_type"] for row in questions}
    missing_types = FULL_EXPERIMENT_QUESTION_TYPES - observed_types
    if missing_types:
        raise ValueError(
            f"Full experiment is missing question types: {sorted(missing_types)}"
        )
