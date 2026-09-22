#!/usr/bin/env python3
"""Compare YOLOX and fixed-ROI routes using one ONNX result contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from close_view_contract import change_gate, change_metrics, roi_pixels
from yolox_contract import decode_birds, padded_box, preprocess_onnx


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classifier_input(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray(((array - MEAN) / STD).transpose(2, 0, 1)[None])


def classify(
    session: ort.InferenceSession, labels: dict[int, str], image: Image.Image
) -> dict[str, object]:
    input_name = session.get_inputs()[0].name
    start = time.perf_counter_ns()
    logits = np.asarray(session.run(None, {input_name: classifier_input(image)})[0]).reshape(-1)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    shifted = logits - logits.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return {
        "index": index,
        "label": labels[index],
        "softmax": float(probabilities[index]),
        "latency_ms": elapsed_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("yolox", "fixed-roi"), required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--roi", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--pixel-delta", type=float, default=0.12)
    parser.add_argument("--mean-change-threshold", type=float, default=0.04)
    parser.add_argument("--changed-pixel-fraction", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "yolox" and not args.detector:
        raise ValueError("--detector is required in yolox mode")
    if args.mode == "fixed-roi" and not args.background:
        raise ValueError("--background is required in fixed-roi mode")
    images = sorted(
        path for path in args.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in {args.image_dir}")
    labels = {int(index): str(label) for index, label in json.loads(args.labels.read_text()).items()}
    classifier = ort.InferenceSession(str(args.classifier), providers=["CPUExecutionProvider"])
    detector = (
        ort.InferenceSession(str(args.detector), providers=["CPUExecutionProvider"])
        if args.mode == "yolox"
        else None
    )
    background = Image.open(args.background).convert("RGB") if args.background else None
    records = []
    for path in images:
        image = Image.open(path).convert("RGB")
        route_start = time.perf_counter_ns()
        detection = None
        metrics = None
        if detector:
            input_meta = detector.get_inputs()[0]
            input_size = int(input_meta.shape[2])
            tensor, scale = preprocess_onnx(np.asarray(image), input_size)
            detector_start = time.perf_counter_ns()
            output = detector.run(None, {input_meta.name: tensor})[0]
            detector_ms = (time.perf_counter_ns() - detector_start) / 1_000_000
            detections = decode_birds(
                output, args.confidence, args.iou, input_size, scale, image.width, image.height
            )
            detection = detections[0] if detections else None
            gate_open = detection is not None
            crop_box = (
                padded_box(detection["box_xyxy"], image.width, image.height, args.padding)
                if detection
                else None
            )
        else:
            detector_ms = None
            metrics = change_metrics(image, background, args.roi, pixel_delta=args.pixel_delta)
            gate_open = change_gate(
                metrics,
                mean_threshold=args.mean_change_threshold,
                fraction_threshold=args.changed_pixel_fraction,
            )
            crop_box = roi_pixels(args.roi, image.width, image.height)
        classification = classify(classifier, labels, image.crop(crop_box)) if gate_open else None
        records.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "route": args.mode,
                "gate_open": gate_open,
                "detector_latency_ms": detector_ms,
                "best_detection": detection,
                "change_metrics": metrics,
                "crop_box_xyxy": list(crop_box) if crop_box else None,
                "classification": classification,
                "total_latency_ms": (time.perf_counter_ns() - route_start) / 1_000_000,
            }
        )
    result = {
        "schema_version": 1,
        "purpose": "close_feeder_dual_route_onnx_acceptance",
        "route": args.mode,
        "classifier_sha256": sha256(args.classifier),
        "detector_sha256": sha256(args.detector) if args.detector else None,
        "background_sha256": sha256(args.background) if args.background else None,
        "contract": {
            "roi_normalized_xyxy": list(args.roi),
            "detector_confidence": args.confidence if detector else None,
            "detector_nms_iou": args.iou if detector else None,
            "crop_padding": args.padding if detector else None,
            "pixel_delta": args.pixel_delta if background is not None else None,
            "mean_change_threshold": args.mean_change_threshold if background is not None else None,
            "changed_pixel_fraction": args.changed_pixel_fraction if background is not None else None,
        },
        "record_count": len(records),
        "gate_open_count": sum(bool(row["gate_open"]) for row in records),
        "records": records,
        "interpretation_boundary": (
            "Thresholds are provisional until frozen on intended-camera frames; this output is not "
            "camera, board, species-accuracy, or outdoor evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"route": args.mode, "images": len(records), "gate_open": result["gate_open_count"]}))


if __name__ == "__main__":
    main()
