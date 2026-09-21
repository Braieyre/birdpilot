#!/usr/bin/env python3
"""Run one RGB image through a BirdPilot RKNN model and persist the evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--model-version", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.warmup_runs < 0:
        raise ValueError("--runs must be positive and --warmup-runs cannot be negative")
    for path, name in ((args.model, "Model"), (args.labels, "Labels"), (args.image, "Image")):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    image = Image.open(args.image).convert("RGB").resize((224, 224), Image.BILINEAR)
    # The converter's RKNN config performs ImageNet normalization.  RKNNLite
    # accepts the raw RGB image through its NHWC runtime interface even though
    # the source ONNX graph itself is NCHW.
    input_tensor = np.asarray(image, dtype=np.uint8)[None, ...]

    rknn = RKNNLite(verbose=False)
    try:
        result = rknn.load_rknn(str(args.model))
        if result != 0:
            raise RuntimeError(f"RKNN model load failed: {result}")
        result = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if result != 0:
            raise RuntimeError(f"RKNN runtime initialization failed: {result}")

        for _ in range(args.warmup_runs):
            rknn.inference(inputs=[input_tensor])

        latencies_ms: list[float] = []
        output = None
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            output = rknn.inference(inputs=[input_tensor])
            latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        assert output is not None
        scores = np.asarray(output[0]).reshape(-1)
        prediction_index = int(np.argmax(scores))
        prediction_label = labels[str(prediction_index)]

        record = {
            "schema_version": 1,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "local_image_rknn_baseline",
            "model": {
                "path": str(args.model.resolve()),
                "sha256": sha256(args.model),
                "version": args.model_version,
                "backend": "RKNNLite",
                "npu_core_mask": "0_1_2",
            },
            "input": {
                "path": str(args.image.resolve()),
                "sha256": sha256(args.image),
                "color_space": "RGB",
                "layout": "NHWC",
                "shape": list(input_tensor.shape),
            },
            "prediction": {
                "index": prediction_index,
                "label": prediction_label,
                "raw_score": float(scores[prediction_index]),
            },
            "benchmark": {
                "warmup_runs": args.warmup_runs,
                "measured_runs": args.runs,
                "latency_ms": {
                    "min": min(latencies_ms),
                    "median": percentile(latencies_ms, 50),
                    "p95": percentile(latencies_ms, 95),
                    "max": max(latencies_ms),
                },
            },
        }
        args.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        record_path = args.results_dir / f"{stamp}_{args.image.stem}_rknn.json"
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"Evidence record written to: {record_path}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
