#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/outputs/blind_scores.csv")
    parser.add_argument("--key", default="data/outputs/blind_key.csv")
    parser.add_argument("--output", default="data/outputs/summary.csv")
    parser.add_argument("--paintings", default="data/paintings.csv")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    key = pd.read_csv(args.key)
    paintings = pd.read_csv(args.paintings)[["painting_id", "fame_group"]]
    merged = scores.merge(key, on="blind_id", validate="one_to_one").merge(
        paintings, on="painting_id", validate="many_to_one"
    )

    metric_columns = [
        "score_0_to_2",
        "hallucination_0_or_1",
        "rejects_false_premise_0_or_1",
    ]
    for column in metric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    required = ["score_0_to_2", "hallucination_0_or_1"]
    if merged[required].isna().any().any():
        raise ValueError("Blind grading is incomplete")
    false_rows = merged["question_type"].eq("false_premise")
    if merged.loc[false_rows, "rejects_false_premise_0_or_1"].isna().any():
        raise ValueError("False-premise rejection grading is incomplete")

    summary = (
        merged.groupby(["condition", "question_type"], dropna=False)[metric_columns]
        .agg(["mean", "count"])
        .round(3)
    )
    summary.to_csv(args.output)

    overall = (
        merged.groupby("condition", dropna=False)[metric_columns]
        .agg(["mean", "count"])
        .round(3)
    )
    by_fame = (
        merged.groupby(["condition", "fame_group"], dropna=False)[metric_columns]
        .agg(["mean", "count"])
        .round(3)
    )
    output_path = Path(args.output)
    overall_path = output_path.with_name("summary_overall.csv")
    fame_path = output_path.with_name("summary_by_fame_group.csv")
    pairwise_path = output_path.with_name("summary_pairwise.csv")
    overall.to_csv(overall_path)
    by_fame.to_csv(fame_path)

    pairwise_rows = []
    comparisons = [
        ("448_native", "448_degraded"),
        ("448_native", "224_native"),
        ("448_degraded", "224_native"),
    ]
    for metric in metric_columns:
        metric_rows = merged.dropna(subset=[metric])
        pivot = metric_rows.pivot(
            index="question_id", columns="condition", values=metric
        )
        for condition_a, condition_b in comparisons:
            paired = pivot[[condition_a, condition_b]].dropna()
            delta = paired[condition_a] - paired[condition_b]
            higher_is_better = metric != "hallucination_0_or_1"
            a_better = delta.gt(0) if higher_is_better else delta.lt(0)
            b_better = delta.lt(0) if higher_is_better else delta.gt(0)
            pairwise_rows.append(
                {
                    "metric": metric,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "n": len(paired),
                    "condition_a_mean": paired[condition_a].mean(),
                    "condition_b_mean": paired[condition_b].mean(),
                    "delta_a_minus_b": delta.mean(),
                    "a_better": int(a_better.sum()),
                    "tie": int(delta.eq(0).sum()),
                    "b_better": int(b_better.sum()),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows).round(3)
    pairwise.to_csv(pairwise_path, index=False)

    print("Overall\n", overall.to_string())
    print("\nBy question type\n", summary.to_string())
    print("\nBy fame group\n", by_fame.to_string())
    print("\nPaired comparisons\n", pairwise.to_string(index=False))
    print(
        f"\nSaved {args.output}, {overall_path}, {fame_path}, and {pairwise_path}"
    )


if __name__ == "__main__":
    main()
