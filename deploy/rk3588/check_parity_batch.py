#!/usr/bin/env python3
"""Compare a batch of validation images between expected ONNX and RKNN FP16."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("source_partition") != "valid":
        raise ValueError("Parity manifest must contain validation images")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    rknn = RKNNLite(verbose=False)
    try:
        if rknn.load_rknn(str(args.model)) != 0:
            raise RuntimeError("RKNN load failed")
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError("RKNN init failed")
        first_path = args.manifest.parent / manifest["records"][0]["image"]
        first = np.asarray(Image.open(first_path).convert("RGB").resize((224, 224)), dtype=np.uint8)[None]
        for _ in range(5):
            rknn.inference(inputs=[first])
        results: list[dict[str, object]] = []
        for record in manifest["records"]:
            path = args.manifest.parent / record["image"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Image hash mismatch: {path}")
            tensor = np.asarray(Image.open(path).convert("RGB").resize((224, 224)), dtype=np.uint8)[None]
            latencies: list[float] = []
            output = None
            for _ in range(args.runs):
                start = time.perf_counter_ns()
                output = rknn.inference(inputs=[tensor])
                latencies.append((time.perf_counter_ns() - start) / 1_000_000)
            scores = np.asarray(output[0]).reshape(-1)
            index = int(scores.argmax())
            results.append(
                {
                    **record,
                    "rknn_index": index,
                    "rknn_label": labels[str(index)],
                    "rknn_raw_score": float(scores[index]),
                    "matches_onnx": index == record["expected_onnx_index"],
                    "latency_ms_median": float(np.median(latencies)),
                }
            )
    finally:
        rknn.release()
    report = {
        "schema_version": 1,
        "model_sha256": sha256(args.model),
        "manifest_sha256": sha256(args.manifest),
        "record_count": len(results),
        "match_count": sum(bool(row["matches_onnx"]) for row in results),
        "mismatch_count": sum(not bool(row["matches_onnx"]) for row in results),
        "median_of_image_medians_ms": float(np.median([row["latency_ms_median"] for row in results])),
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("record_count", "match_count", "mismatch_count", "median_of_image_medians_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
