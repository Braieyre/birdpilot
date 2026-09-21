#!/usr/bin/env python3
"""Generate BirdPilot's deterministic degradation pilot or validation benchmark.

The script writes selected source images and five degradation families at
three strengths. The test partition requires an explicit final-evaluation
flag so it cannot accidentally influence tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageEnhance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "birds"
DEFAULT_CSV = DEFAULT_DATA_ROOT / "birds.csv"

LEVELS = {
    "light": {"low_light": 0.75, "motion_blur": 3, "occlusion": 0.08, "downscale": 0.75, "jpeg": 70},
    "medium": {"low_light": 0.55, "motion_blur": 7, "occlusion": 0.16, "downscale": 0.50, "jpeg": 45},
    "heavy": {"low_light": 0.38, "motion_blur": 13, "occlusion": 0.28, "downscale": 0.35, "jpeg": 25},
}
FAMILIES = ("low_light", "motion_blur", "occlusion", "downscale", "jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--source-count", type=int, default=20)
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_sources(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["labels"], item["filepaths"])):
        by_label[str(row["labels"])].append(str(row["filepaths"]))
    labels = sorted(by_label)
    if count > len(labels):
        raise ValueError(f"source-count {count} exceeds available classes {len(labels)}")
    rng = random.Random(seed)
    rng.shuffle(labels)
    selected: list[dict[str, str]] = []
    for label in labels[:count]:
        candidates = by_label[label]
        selected.append({"label": label, "filepaths": candidates[rng.randrange(len(candidates))]})
    return selected


def degrade(image: Image.Image, family: str, value: float | int, seed: int) -> Image.Image:
    image = image.convert("RGB")
    if family == "low_light":
        return ImageEnhance.Brightness(image).enhance(float(value))
    if family == "motion_blur":
        length = int(value)
        radius = length // 2
        pixels = np.asarray(image, dtype=np.float32)
        padded = np.pad(pixels, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        blurred = np.zeros_like(pixels)
        for offset in range(-radius, radius + 1):
            blurred += padded[
                radius + offset : radius + offset + pixels.shape[0],
                radius + offset : radius + offset + pixels.shape[1],
                :,
            ]
        return Image.fromarray(np.clip(blurred / length, 0, 255).astype(np.uint8))
    if family == "occlusion":
        rng = np.random.default_rng(seed)
        width, height = image.size
        area = width * height * float(value)
        aspect = 1.5
        ellipse_height = max(1, int(round(math.sqrt(4.0 * area / (math.pi * aspect)))))
        ellipse_width = max(1, int(round(aspect * ellipse_height)))
        center_x = float(rng.uniform(0.4, 0.6)) * width
        center_y = float(rng.uniform(0.4, 0.6)) * height
        x = int(round(center_x - ellipse_width / 2))
        y = int(round(center_y - ellipse_height / 2))
        x = min(max(0, x), max(0, width - ellipse_width))
        y = min(max(0, y), max(0, height - ellipse_height))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        from PIL import ImageDraw

        ImageDraw.Draw(overlay).ellipse(
            (x, y, x + ellipse_width, y + ellipse_height), fill=(68, 77, 42, 255)
        )
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    if family == "downscale":
        width, height = image.size
        reduced = image.resize((max(1, int(width * float(value))), max(1, int(height * float(value)))), Image.Resampling.BILINEAR)
        return reduced.resize((width, height), Image.Resampling.BILINEAR)
    if family == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(value), optimize=False)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").copy()
    raise ValueError(f"Unsupported family: {family}")


def main() -> None:
    args = parse_args()
    if args.source_split == "test" and not args.allow_test:
        raise ValueError("--source-split test requires --allow-test for final evaluation")
    if args.source_count <= 0:
        raise ValueError("--source-count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)
    with args.csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    source_rows = [row for row in rows if row.get("data set") == args.source_split]
    if not source_rows:
        raise ValueError(f"No {args.source_split} rows found")
    available_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    for row in source_rows:
        if (args.data_root / row["filepaths"]).is_file():
            available_rows.append(row)
        else:
            excluded_rows.append({"label": row["labels"], "relative_path": row["filepaths"], "reason": "missing_file"})
    if args.all_sources:
        sources = [
            {"label": row["labels"], "filepaths": row["filepaths"]}
            for row in sorted(available_rows, key=lambda item: (item["labels"], item["filepaths"]))
        ]
    else:
        sources = select_sources(available_rows, args.source_count, args.seed)
    args.output_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "generator_version": "degradation_v2",
        "dependencies": {"numpy": np.__version__, "pillow": PIL.__version__},
        "seed": args.seed,
        "source_partition": args.source_split,
        "source_count": len(sources),
        "excluded_source_rows": excluded_rows,
        "families": list(FAMILIES),
        "levels": LEVELS,
        "records": [],
    }
    records: list[dict[str, object]] = []
    for source_index, source in enumerate(sources):
        source_path = args.data_root / source["filepaths"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        image = Image.open(source_path).convert("RGB")
        source_hash = sha256(source_path)
        for family_index, family in enumerate(FAMILIES):
            for level_index, (level, values) in enumerate(LEVELS.items()):
                output_name = f"s{source_index:02d}_{family}_{level}.jpg"
                output_path = args.output_dir / output_name
                variant_seed = args.seed + source_index * 100 + family_index * 10
                if family != "occlusion":
                    variant_seed += level_index
                degrade(image, family, values[family], variant_seed).save(output_path, quality=95, optimize=False)
                records.append(
                    {
                        "source_index": source_index,
                        "source_partition": args.source_split,
                        "source_relative_path": source["filepaths"],
                        "source_label": source["label"],
                        "source_sha256": source_hash,
                        "family": family,
                        "level": level,
                        "parameter": values[family],
                        "seed": variant_seed,
                        "output_path": output_name,
                        "output_sha256": sha256(output_path),
                    }
                )
    manifest["records"] = records
    manifest["generated_count"] = len(records)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_count": len(sources), "generated_count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
