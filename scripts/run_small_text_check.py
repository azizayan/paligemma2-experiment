#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from paligemma2_experiment.images import degrade_then_restore
from paligemma2_experiment.io import append_jsonl, read_jsonl


CONDITIONS = {
    "224_native": ("google/paligemma2-3b-mix-224", False),
    "448_degraded": ("google/paligemma2-3b-mix-448", True),
    "448_native": ("google/paligemma2-3b-mix-448", False),
}

CASES = [
    ("label_01", 10, ["NORTH HALL", "LUMINOUS HARBOR", "ELENA MARIN", "ROOM 27B", "CATALOG QF-318"]),
    ("label_02", 11, ["ARCHIVE DESK", "SILVER ORCHARD", "TOMAS VELIN", "SHELF 14C", "RECORD MK-572"]),
    ("label_03", 12, ["EAST GALLERY", "WINTER SIGNAL", "MARA SOLEN", "ROOM 08A", "CATALOG RD-406"]),
    ("label_04", 13, ["STUDY CENTER", "QUIET ESTUARY", "NIKOS DAREN", "CASE 31F", "RECORD BT-925"]),
    ("label_05", 14, ["UPPER COURT", "AMBER CURRENT", "LINA VOREL", "ROOM 19D", "CATALOG HX-743"]),
    ("label_06", 15, ["WEST ANNEX", "GREEN MERIDIAN", "OMAR SELIN", "CASE 42B", "RECORD JP-861"]),
    ("label_07", 16, ["PRINT ROOM", "DISTANT LANTERN", "IRIS NAVEN", "SHELF 06E", "CATALOG CW-294"]),
    ("label_08", 17, ["SOUTH WING", "PALE HORIZON", "DENIZ ARVEN", "ROOM 23C", "RECORD LS-680"]),
    ("label_09", 18, ["RIVER GALLERY", "CRIMSON TIDE", "SERA MOLIN", "CASE 17A", "CATALOG VK-539"]),
    ("label_10", 20, ["CITY ARCHIVE", "BLUE PASSAGE", "ARIN DEMER", "SHELF 35D", "RECORD NZ-812"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/images/small_text_check")
    parser.add_argument(
        "--output", default="data/outputs/small_text_check_read_text.jsonl"
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--prompt", default="Read text.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def font_path() -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Arial or DejaVu Sans is required to generate labels")


def generate_images(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, font_size, lines in CASES:
        path = output_dir / f"{case_id}.png"
        image = Image.new("RGB", (448, 448), "#f5f1e8")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path(), font_size)
        line_height = font_size + 8
        block_height = len(lines) * line_height - 8
        y = (448 - block_height) // 2
        draw.rectangle((36, y - 28, 412, y + block_height + 28), outline="#28251f", width=2)
        for line in lines:
            width = draw.textbbox((0, 0), line, font=font)[2]
            draw.text(((448 - width) / 2, y), line, font=font, fill="#151412")
            y += line_height
        image.save(path)
        cases.append(
            {
                "case_id": case_id,
                "font_size_px": font_size,
                "image_path": str(path),
                "reference": "\n".join(lines),
            }
        )
    return cases


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return " ".join(value.split())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, response: str) -> float:
    normalized_reference = normalize(reference)
    normalized_response = normalize(response)
    return edit_distance(normalized_reference, normalized_response) / max(
        len(normalized_reference), 1
    )


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str, target: torch.device):
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if target.type != "cpu" else torch.float32
    kwargs = {"torch_dtype": dtype, "attn_implementation": "sdpa"}
    if target.type == "cuda":
        kwargs["device_map"] = {"": 0}
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
    if target.type != "cuda":
        model = model.to(target)
    return processor, model


def generate(
    processor,
    model,
    target,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
):
    prompt = f"<image>{prompt}"
    dtype = torch.float16 if target.type != "cpu" else torch.float32
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        target, dtype=dtype
    )
    input_length = inputs["input_ids"].shape[-1]
    if target.type == "cuda":
        torch.cuda.synchronize()
    elif target.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if target.type == "cuda":
        torch.cuda.synchronize()
    elif target.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    response = processor.decode(
        generated[0][input_length:], skip_special_tokens=True
    ).strip()
    return response, elapsed, generated.shape[-1] - input_length


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    cases = generate_images(Path(args.images))
    completed = set()
    if output.exists():
        completed = {
            (row["condition"], row["case_id"]) for row in read_jsonl(output)
        }

    target = device()
    print(f"Using device: {target}")
    for condition in args.conditions:
        model_id, degraded = CONDITIONS[condition]
        processor, model = load_model(model_id, target)
        for case in cases:
            key = (condition, case["case_id"])
            if key in completed:
                continue
            image = Image.open(case["image_path"]).convert("RGB")
            if degraded:
                image = degrade_then_restore(image)
            response, elapsed, generated_tokens = generate(
                processor,
                model,
                target,
                image,
                args.prompt,
                args.max_new_tokens,
            )
            cer = character_error_rate(case["reference"], response)
            append_jsonl(
                output,
                {
                    **case,
                    "condition": condition,
                    "model_id": model_id,
                    "prompt": args.prompt,
                    "response": response,
                    "normalized_cer": round(cer, 4),
                    "exact_match": normalize(case["reference"])
                    == normalize(response),
                    "generated_tokens": generated_tokens,
                    "elapsed_seconds": round(elapsed, 4),
                },
            )
            print(
                f"{condition} {case['case_id']} font={case['font_size_px']} "
                f"CER={cer:.3f}: {response!r}"
            )
        del model, processor
        if target.type == "cuda":
            torch.cuda.empty_cache()
        elif target.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
