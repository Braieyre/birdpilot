#!/usr/bin/env python3
"""Run YOLOX or fixed-ROI close-view inference with RKNNLite on RK3588."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from yolox_contract import decode_birds, padded_box, preprocess_rknn  # noqa: E402
from close_view_contract import change_gate, change_metrics, roi_pixels  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("yolox", "fixed-roi"), default="yolox")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--classifier-model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--roi", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--pixel-delta", type=float, default=0.12)
    parser.add_argument("--mean-change-threshold", type=float, default=0.04)
    parser.add_argument("--changed-pixel-fraction", type=float, default=0.08)
    args = parser.parse_args()

    if args.mode == "yolox" and not args.model:
        raise ValueError("--model is required in yolox mode")
    if args.mode == "fixed-roi" and not args.background:
        raise ValueError("--background is required in fixed-roi mode")

    images = sorted(
        path for path in args.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No images found in {args.image_dir}")
    labels = json.loads(args.labels.read_text())
    detector = RKNNLite(verbose=False) if args.mode == "yolox" else None
    classifier = RKNNLite(verbose=False)
    background = Image.open(args.background).convert("RGB") if args.background else None
    records = []
    try:
        if detector and detector.load_rknn(str(args.model)) != 0:
            raise RuntimeError("Detector RKNN model load failed")
        if detector and detector.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError("Detector RKNN runtime initialization failed")
        if classifier.load_rknn(str(args.classifier_model)) != 0:
            raise RuntimeError("Classifier RKNN model load failed")
        if classifier.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError("Classifier RKNN runtime initialization failed")
        for path in images:
            image = Image.open(path).convert("RGB")
            array = np.asarray(image)
            start = time.perf_counter_ns()
            metrics = None
            if detector:
                tensor, scale = preprocess_rknn(array)
                detector_start = time.perf_counter_ns()
                output = detector.inference(inputs=[tensor])[0]
                detector_latency_ms = (time.perf_counter_ns() - detector_start) / 1_000_000
                detections = decode_birds(
                    output, args.confidence, args.iou, 416, scale, image.width, image.height
                )
                best = detections[0] if detections else None
                gate_open = best is not None
                crop_box = (
                    padded_box(best["box_xyxy"], image.width, image.height, args.padding)
                    if best
                    else None
                )
            else:
                detector_latency_ms = None
                best = None
                metrics = change_metrics(image, background, args.roi, pixel_delta=args.pixel_delta)
                gate_open = change_gate(
                    metrics,
                    mean_threshold=args.mean_change_threshold,
                    fraction_threshold=args.changed_pixel_fraction,
                )
                crop_box = roi_pixels(args.roi, image.width, image.height)
            classification = None
            if gate_open:
                crop = image.crop(crop_box).resize((224, 224), Image.Resampling.BILINEAR)
                classifier_input = np.asarray(crop, dtype=np.uint8)[None]
                scores = np.asarray(
                    classifier.inference(inputs=[classifier_input])[0]
                ).reshape(-1)
                index = int(scores.argmax())
                classification = {
                    "index": index,
                    "label": labels[str(index)],
                    "raw_score": float(scores[index]),
                }
            records.append(
                {
                    "file": path.name,
                    "sha256": sha256(path),
                    "route": args.mode,
                    "gate_open": gate_open,
                    "detector_latency_ms": detector_latency_ms,
                    "best_detection": best,
                    "change_metrics": metrics,
                    "crop_box_xyxy": list(crop_box) if crop_box else None,
                    "classification": classification,
                    "total_latency_ms": (time.perf_counter_ns() - start) / 1_000_000,
                }
            )
    finally:
        if detector:
            detector.release()
        classifier.release()
    result = {
        "schema_version": 1,
        "purpose": "close_feeder_dual_route_rknn_acceptance",
        "route": args.mode,
        "model_sha256": sha256(args.model) if args.model else None,
        "classifier_model_sha256": sha256(args.classifier_model),
        "background_sha256": sha256(args.background) if args.background else None,
        "contract": {
            "roi_normalized_xyxy": list(args.roi),
            "detector_input": "BGR uint8 NHWC 1x416x416" if detector else None,
            "confidence": args.confidence if detector else None,
            "nms_iou": args.iou if detector else None,
            "crop_padding": args.padding if detector else None,
            "pixel_delta": args.pixel_delta if background is not None else None,
            "mean_change_threshold": args.mean_change_threshold if background is not None else None,
            "changed_pixel_fraction": args.changed_pixel_fraction if background is not None else None,
        },
        "record_count": len(records),
        "gate_open_count": sum(bool(record["gate_open"]) for record in records),
        "records": records,
        "interpretation_boundary": (
            "Thresholds are provisional until frozen on intended-camera frames; this output is not "
            "camera, species-accuracy, or outdoor evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"route": args.mode, "images": len(records),
                      "gate_open": sum(r["gate_open"] for r in records)}))


if __name__ == "__main__":
    main()
