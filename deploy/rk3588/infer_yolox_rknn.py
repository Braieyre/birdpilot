#!/usr/bin/env python3
"""Run the frozen YOLOX bird detector contract with RKNNLite on RK3588."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier-model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--padding", type=float, default=0.10)
    args = parser.parse_args()

    images = sorted(
        path for path in args.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No images found in {args.image_dir}")
    labels = json.loads(args.labels.read_text())
    rknn = RKNNLite(verbose=False)
    classifier = RKNNLite(verbose=False)
    records = []
    try:
        if rknn.load_rknn(str(args.model)) != 0:
            raise RuntimeError("RKNN model load failed")
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError("RKNN runtime initialization failed")
        if classifier.load_rknn(str(args.classifier_model)) != 0:
            raise RuntimeError("Classifier RKNN model load failed")
        if classifier.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError("Classifier RKNN runtime initialization failed")
        for path in images:
            image = Image.open(path).convert("RGB")
            array = np.asarray(image)
            tensor, scale = preprocess_rknn(array)
            start = time.perf_counter_ns()
            output = rknn.inference(inputs=[tensor])[0]
            latency_ms = (time.perf_counter_ns() - start) / 1_000_000
            detections = decode_birds(
                output, args.confidence, args.iou, 416, scale, image.width, image.height
            )
            best = detections[0] if detections else None
            crop_box = (
                padded_box(best["box_xyxy"], image.width, image.height, args.padding)
                if best
                else None
            )
            classification = None
            if crop_box:
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
                    "latency_ms": latency_ms,
                    "bird_detected": best is not None,
                    "best_detection": best,
                    "crop_box_xyxy": list(crop_box) if crop_box else None,
                    "classification": classification,
                }
            )
    finally:
        rknn.release()
        classifier.release()
    result = {
        "schema_version": 1,
        "model_sha256": sha256(args.model),
        "classifier_model_sha256": sha256(args.classifier_model),
        "contract": {"input": "BGR uint8 NHWC 1x416x416", "confidence": args.confidence,
                     "nms_iou": args.iou, "crop_padding": args.padding},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"images": len(records), "detections": sum(r["bird_detected"] for r in records)}))


if __name__ == "__main__":
    main()
