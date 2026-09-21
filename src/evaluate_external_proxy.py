#!/usr/bin/env python3
"""Run the frozen ONNX classifier on an external full-scene proxy manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return ((array - MEAN) / STD).transpose(2, 0, 1)


def softmax_top(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    indices = probabilities.argmax(axis=1)
    return indices, probabilities[np.arange(len(indices)), indices]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for path in (args.onnx, args.labels, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    labels = {int(index): str(label) for index, label in json.loads(args.labels.read_text()).items()}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    for record in records:
        path = args.image_root / str(record["candidate_file"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing image or hash mismatch: {path}")

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    predictions: list[dict[str, object]] = []
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        batch = np.stack(
            [preprocess(args.image_root / str(record["candidate_file"])) for record in batch_records]
        )
        logits = session.run(None, {input_name: batch})[0]
        indices, confidences = softmax_top(logits)
        for record, index, confidence in zip(batch_records, indices, confidences):
            expected_index = record.get("class_index")
            prediction = {
                    "candidate_file": record["candidate_file"],
                    "source_category": record.get("source_category", "licensed_species_candidate"),
                    "expected_index": expected_index,
                    "expected_label": record.get("model_label"),
                    "predicted_index": int(index),
                    "predicted_label": labels[int(index)],
                    "softmax_top1": float(confidence),
                    "matches_expected": int(index) == expected_index if expected_index is not None else None,
                }
            for key in (
                "source_candidate_file",
                "detector_confidence",
                "detector_box_xyxy",
                "crop_box_xyxy",
                "detector_area_fraction",
            ):
                if key in record:
                    prediction[key] = record[key]
            predictions.append(prediction)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["source_category"])].append(row)
    groups: dict[str, dict[str, object]] = {}
    for name, rows in sorted(grouped.items()):
        confidences = [float(row["softmax_top1"]) for row in rows]
        labels_seen = Counter(str(row["predicted_label"]) for row in rows)
        eligible = [row for row in rows if row["matches_expected"] is not None]
        groups[name] = {
            "count": len(rows),
            "softmax_top1_mean": float(np.mean(confidences)),
            "softmax_top1_median": float(np.median(confidences)),
            "softmax_top1_max": max(confidences),
            "top_predicted_labels": labels_seen.most_common(10),
            "species_accuracy_count": len(eligible),
            "species_top1_accuracy": (
                sum(bool(row["matches_expected"]) for row in eligible) / len(eligible)
                if eligible
                else None
            ),
        }

    detected_scene_summary = None
    crop_rows = [row for row in predictions if row.get("source_candidate_file")]
    if crop_rows:
        best_by_scene: dict[str, dict[str, object]] = {}
        for row in crop_rows:
            scene = str(row["source_candidate_file"])
            if scene not in best_by_scene or float(row["detector_confidence"]) > float(
                best_by_scene[scene]["detector_confidence"]
            ):
                best_by_scene[scene] = row
        scene_rows = list(best_by_scene.values())
        eligible_scenes = [row for row in scene_rows if row["matches_expected"] is not None]
        detected_scene_summary = {
            "selection": "highest_detector_confidence_crop_per_source_scene",
            "detected_scene_count": len(scene_rows),
            "species_accuracy_count": len(eligible_scenes),
            "species_top1_accuracy": (
                sum(bool(row["matches_expected"]) for row in eligible_scenes) / len(eligible_scenes)
                if eligible_scenes
                else None
            ),
        }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for row in predictions:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": 1,
        "purpose": "external_proxy_classifier_probe",
        "input_manifest_purpose": manifest.get("purpose"),
        "onnx_sha256": sha256(args.onnx),
        "label_map_sha256": sha256(args.labels),
        "source_manifest_sha256": sha256(args.manifest),
        "record_count": len(predictions),
        "groups": groups,
        "detected_scene_summary": detected_scene_summary,
        "interpretation_boundary": (
            "Classifier outputs on external scenes or detector crops; no bird-presence claim from "
            "classifier confidence and no camera or outdoor claim."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
