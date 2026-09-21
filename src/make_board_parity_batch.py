#!/usr/bin/env python3
"""Package deterministic validation images with expected ONNX predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--degradation-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    expected: dict[str, dict[str, object]] = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["family"] == "clean":
            expected[str(row["source_relative_path"])] = row
    degraded = json.loads(args.degradation_manifest.read_text(encoding="utf-8"))
    sources: dict[int, dict[str, object]] = {}
    for row in degraded["records"]:
        sources.setdefault(int(row["source_index"]), row)
    first_source_by_class: dict[int, dict[str, object]] = {}
    for source in (sources[index] for index in sorted(sources)):
        relative = str(source["source_relative_path"])
        prediction = expected.get(relative)
        if prediction is None:
            continue
        first_source_by_class.setdefault(int(prediction["true_index"]), source)
    candidates = [first_source_by_class[index] for index in sorted(first_source_by_class)]
    if args.count > len(candidates):
        raise ValueError(
            f"Requested {args.count} class-diverse samples, but only {len(candidates)} classes are available"
        )
    if args.count == 1:
        selected = [candidates[0]]
    else:
        # Spread fixed indices over the complete ordered class list. This keeps the
        # package reproducible while preventing adjacent-directory sampling.
        positions = [round(index * (len(candidates) - 1) / (args.count - 1)) for index in range(args.count)]
        selected = [candidates[position] for position in positions]
    args.output_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for index, source in enumerate(selected):
        relative = str(source["source_relative_path"])
        if relative not in expected:
            raise KeyError(f"No clean ONNX prediction for {relative}")
        source_path = args.data_root / relative
        suffix = source_path.suffix.lower() or ".jpg"
        name = f"image_{index:03d}{suffix}"
        destination = args.output_dir / name
        shutil.copy2(source_path, destination)
        prediction = expected[relative]
        records.append(
            {
                "image": name,
                "source_relative_path": relative,
                "sha256": sha256(destination),
                "true_index": prediction["true_index"],
                "true_label": prediction["true_label"],
                "expected_onnx_index": prediction["predicted_index"],
                "expected_onnx_label": prediction["predicted_label"],
            }
        )
    manifest = {
        "schema_version": 1,
        "source_partition": "valid",
        "expected_backend": "ONNXRuntime",
        "source_onnx": str(args.onnx.resolve()),
        "source_onnx_sha256": sha256(args.onnx),
        "predictions_sha256": sha256(args.predictions),
        "sampling": "first_readable_source_from_evenly_spaced_class_indices",
        "available_class_count": len(candidates),
        "record_count": len(records),
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "record_count": len(records)}))


if __name__ == "__main__":
    main()
