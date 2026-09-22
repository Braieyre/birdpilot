#!/usr/bin/env python3
"""Convert a frozen official YOLOX COCO ONNX model to FP16 RKNN for RK3588."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from rknn.api import RKNN


MODEL_SOURCE = "https://github.com/Megvii-BaseDetection/YOLOX"
MODEL_LICENSE = "Apache-2.0"
MEAN_VALUES = [[0.0, 0.0, 0.0]]
STD_VALUES = [[1.0, 1.0, 1.0]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="yolox-nano")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    try:
        result = rknn.config(
            target_platform="rk3588",
            mean_values=MEAN_VALUES,
            std_values=STD_VALUES,
            optimization_level=3,
        )
        if result != 0:
            raise RuntimeError(f"RKNN config failed: {result}")
        result = rknn.load_onnx(
            model=str(args.onnx),
            inputs=["images"],
            input_size_list=[[1, 3, 416, 416]],
        )
        if result != 0:
            raise RuntimeError(f"ONNX import failed: {result}")
        result = rknn.build(do_quantization=False)
        if result != 0:
            raise RuntimeError(f"RKNN FP16 build failed: {result}")
        result = rknn.export_rknn(str(args.output))
        if result != 0:
            raise RuntimeError(f"RKNN export failed: {result}")

        manifest = {
            "schema_version": 1,
            "purpose": "birdpilot_yolox_fp16_detector_for_rk3588",
            "variant": args.variant,
            "model_source": MODEL_SOURCE,
            "model_license": MODEL_LICENSE,
            "rknn_toolkit2_version": importlib.metadata.version("rknn-toolkit2"),
            "source_onnx": str(args.onnx.resolve()),
            "source_onnx_sha256": sha256(args.onnx),
            "output_rknn": str(args.output.resolve()),
            "output_rknn_sha256": sha256(args.output),
            "target": "rk3588",
            "input_name": "images",
            "input_shape": [1, 3, 416, 416],
            "source_input_layout": "NCHW",
            "runtime_input_layout": "NHWC",
            "input_color_space": "BGR",
            "input_value_range": [0, 255],
            "letterbox_alignment": "top_left",
            "letterbox_value": 114,
            "normalization": {"mean": MEAN_VALUES[0], "std": STD_VALUES[0]},
            "quantized_int8": False,
            "output_contract": "YOLOX raw [1,3549,85], decoded outside the graph",
        }
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
