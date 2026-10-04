#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from paligemma2_experiment.images import degrade_then_restore
from paligemma2_experiment.io import (
    PAINTING_FIELDS,
    QUESTION_FIELDS,
    append_jsonl,
    read_csv,
    read_jsonl,
    validate_fields,
    validate_full_experiment,
)


CONDITIONS = {
    "448_native": ("google/paligemma2-3b-mix-448", False),
    "448_degraded": ("google/paligemma2-3b-mix-448", True),
    "224_native": ("google/paligemma2-3b-mix-224", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paintings", default="data/paintings.csv")
    parser.add_argument("--questions", default="data/questions.jsonl")
    parser.add_argument("--output", default="data/outputs/zero_shot.jsonl")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser.parse_args()


def completed_keys(output_path: Path) -> set[tuple[str, str, str]]:
    if not output_path.exists():
        return set()
    return {
        (row["condition"], row["painting_id"], row["question_id"])
        for row in read_jsonl(output_path)
    }


def inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = {"": 0}
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, **load_kwargs
    ).eval()
    if device.type != "cuda":
        model = model.to(device)
    return processor, model


def generate(
    processor,
    model,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    model_prompt = prompt if prompt.startswith("<image>") else f"<image>{prompt}"
    inputs = processor(images=image, text=model_prompt, return_tensors="pt").to(
        device, dtype=dtype
    )
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return processor.decode(
        generated[0][input_length:], skip_special_tokens=True
    ).strip()


def main() -> None:
    args = parse_args()
    paintings_path = Path(args.paintings)
    questions_path = Path(args.questions)
    output_path = Path(args.output)

    paintings = read_csv(paintings_path)
    questions = read_jsonl(questions_path)
    validate_fields(paintings, PAINTING_FIELDS, "paintings")
    validate_fields(questions, QUESTION_FIELDS, "questions")
    validate_full_experiment(paintings, questions)

    painting_by_id = {row["painting_id"]: row for row in paintings}
    missing = sorted({q["painting_id"] for q in questions} - painting_by_id.keys())
    if missing:
        raise ValueError(f"Questions reference unknown paintings: {missing}")

    done = completed_keys(output_path)
    project_root = paintings_path.resolve().parent.parent
    device = inference_device()
    print(f"Using device: {device}")

    for condition in args.conditions:
        model_id, degraded = CONDITIONS[condition]
        processor, model = load_model(model_id, device)

        for question in questions:
            key = (condition, question["painting_id"], question["question_id"])
            if key in done:
                continue

            painting = painting_by_id[question["painting_id"]]
            image_path = project_root / painting["image_path"]
            image = Image.open(image_path).convert("RGB")
            if degraded:
                image = degrade_then_restore(image)

            started = time.perf_counter()
            response = generate(
                processor,
                model,
                device,
                image,
                question["prompt"],
                args.max_new_tokens,
            )
            append_jsonl(
                output_path,
                {
                    "condition": condition,
                    "model_id": model_id,
                    "painting_id": question["painting_id"],
                    "question_id": question["question_id"],
                    "question_type": question["question_type"],
                    "prompt": question["prompt"],
                    "reference_answer": question["reference_answer"],
                    "false_premise": bool(question.get("false_premise", False)),
                    "response": response,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
            )
            print(f"{condition} {question['question_id']}: {response}")

        del model, processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
