#!/usr/bin/env python3
"""Evaluate the frozen BirdPilot ONNX model on clean and degraded validation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--degradation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parity-count", type=int, default=16)
    parser.add_argument("--expected-partition", choices=("valid", "test"), default="valid")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    index_to_label = {int(index): str(label) for index, label in raw.items()}
    expected = set(range(len(index_to_label)))
    if set(index_to_label) != expected:
        raise ValueError("Label-map indices must be contiguous from zero")
    label_to_index = {label: index for index, label in index_to_label.items()}
    if len(label_to_index) != len(index_to_label):
        raise ValueError("Label map contains duplicate names")
    return index_to_label, label_to_index


def preprocess(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return array.transpose(2, 0, 1)


def clean_records(
    csv_path: Path,
    data_root: Path,
    labels: dict[str, int],
    partition: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    records: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("data set") != partition:
                continue
            label = row["labels"]
            if label not in labels:
                exclusions.append({"relative_path": row["filepaths"], "label": label, "reason": "label_not_in_model"})
                continue
            path = data_root / row["filepaths"]
            if not path.is_file():
                exclusions.append({"relative_path": row["filepaths"], "label": label, "reason": "missing_file"})
                continue
            records.append(
                {
                    "image_path": path,
                    "source_relative_path": row["filepaths"],
                    "label": label,
                    "true_index": labels[label],
                    "family": "clean",
                    "level": "clean",
                }
            )
    return records, exclusions


def degraded_records(
    manifest_path: Path,
    labels: dict[str, int],
    partition: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_partition") != partition:
        raise ValueError(f"Degradation manifest must contain {partition} sources only")
    records: list[dict[str, object]] = []
    for row in manifest["records"]:
        label = str(row["source_label"])
        if label not in labels:
            raise ValueError(f"Manifest label is absent from model mapping: {label}")
        path = manifest_path.parent / str(row["output_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "image_path": path,
                "source_relative_path": row["source_relative_path"],
                "label": label,
                "true_index": labels[label],
                "family": row["family"],
                "level": row["level"],
            }
        )
    return records, manifest


def verify_checkpoint_parity(
    checkpoint_path: Path,
    session: ort.InferenceSession,
    sample_records: list[dict[str, object]],
    label_to_index: dict[str, int],
    count: int,
) -> dict[str, object]:
    import torch
    import torch.nn as nn
    from torchvision import models

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("num_classes") != len(label_to_index):
        raise ValueError("Checkpoint class count and label map differ")
    if checkpoint.get("class_to_idx") != label_to_index:
        raise ValueError("Checkpoint class mapping and label map differ")
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(label_to_index))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batch = np.stack([preprocess(Path(record["image_path"])) for record in sample_records[:count]])
    with torch.no_grad():
        torch_output = model(torch.from_numpy(batch)).numpy()
    onnx_output = session.run(None, {session.get_inputs()[0].name: batch})[0]
    torch_pred = torch_output.argmax(axis=1)
    onnx_pred = onnx_output.argmax(axis=1)
    return {
        "sample_count": int(len(batch)),
        "prediction_mismatches": int(np.count_nonzero(torch_pred != onnx_pred)),
        "max_abs_logit_delta": float(np.max(np.abs(torch_output - onnx_output))),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_best_valid_acc": float(checkpoint["best_valid_acc"]),
        "checkpoint_config": checkpoint["config"],
    }


def metrics(predictions: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["family"]), str(row["level"]))].append(row)
    conditions: dict[str, dict[str, object]] = {}
    degraded_accuracies: list[float] = []
    for (family, level), rows in sorted(grouped.items()):
        correct = sum(bool(row["correct"]) for row in rows)
        by_class: dict[int, list[bool]] = defaultdict(list)
        for row in rows:
            by_class[int(row["true_index"])].append(bool(row["correct"]))
        macro = float(np.mean([np.mean(values) for values in by_class.values()]))
        accuracy = correct / len(rows)
        key = "clean" if family == "clean" else f"{family}/{level}"
        conditions[key] = {
            "count": len(rows),
            "class_count": len(by_class),
            "correct": correct,
            "top1_accuracy": accuracy,
            "macro_top1_accuracy": macro,
        }
        if family != "clean":
            degraded_accuracies.append(accuracy)
    return {
        "conditions": conditions,
        "degraded_condition_mean_top1_accuracy": float(np.mean(degraded_accuracies)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.parity_count <= 0:
        raise ValueError("Batch and parity counts must be positive")
    for path in (args.onnx, args.labels, args.csv, args.degradation_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.checkpoint and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.onnx_data and not args.onnx_data.is_file():
        raise FileNotFoundError(args.onnx_data)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    index_to_label, label_to_index = load_labels(args.labels)
    clean, clean_exclusions = clean_records(
        args.csv, args.data_root, label_to_index, args.expected_partition
    )
    degraded, manifest = degraded_records(
        args.degradation_manifest, label_to_index, args.expected_partition
    )
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    parity = None
    if args.checkpoint:
        parity = verify_checkpoint_parity(
            args.checkpoint, session, clean, label_to_index, args.parity_count
        )
        if parity["prediction_mismatches"] != 0:
            raise RuntimeError(f"Checkpoint/ONNX prediction mismatch: {parity}")

    all_records = clean + degraded
    predictions: list[dict[str, object]] = []
    input_name = session.get_inputs()[0].name
    for start in range(0, len(all_records), args.batch_size):
        rows = all_records[start : start + args.batch_size]
        batch = np.stack([preprocess(Path(row["image_path"])) for row in rows])
        output = session.run(None, {input_name: batch})[0]
        predicted = output.argmax(axis=1)
        scores = output[np.arange(len(rows)), predicted]
        for row, pred, score in zip(rows, predicted, scores):
            pred_index = int(pred)
            predictions.append(
                {
                    "source_relative_path": row["source_relative_path"],
                    "family": row["family"],
                    "level": row["level"],
                    "true_index": row["true_index"],
                    "true_label": row["label"],
                    "predicted_index": pred_index,
                    "predicted_label": index_to_label[pred_index],
                    "raw_top1_score": float(score),
                    "correct": pred_index == row["true_index"],
                }
            )

    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as file:
        for row in predictions:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": 1,
        "source_partition": args.expected_partition,
        "model": {
            "onnx_path": str(args.onnx.resolve()),
            "onnx_sha256": sha256(args.onnx),
            "onnx_data_path": str(args.onnx_data.resolve()) if args.onnx_data else None,
            "onnx_data_sha256": sha256(args.onnx_data) if args.onnx_data else None,
            "checkpoint_path": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
            "label_map_sha256": sha256(args.labels),
        },
        "degradation_manifest_sha256": sha256(args.degradation_manifest),
        "clean_count": len(clean),
        "degraded_count": len(degraded),
        "clean_exclusions": clean_exclusions,
        "manifest_exclusions": manifest.get("excluded_source_rows", []),
        "checkpoint_onnx_parity": parity,
        **metrics(predictions),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), **summary["conditions"]["clean"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
