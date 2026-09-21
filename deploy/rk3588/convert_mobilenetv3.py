#!/usr/bin/env python3
"""Convert BirdPilot's frozen MobileNetV3 ONNX model for RK3588.

Run this only in an RKNN-Toolkit2 environment.  The source model expects an
NCHW RGB tensor normalised with ImageNet mean/std.  RKNN receives raw RGB
pixels and performs that normalisation from the fixed conversion settings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from rknn.api import RKNN


MEAN_VALUES = [[123.675, 116.28, 103.53]]
STD_VALUES = [[58.395, 57.12, 57.375]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-list",
        type=Path,
        required=True,
        help="One absolute RGB image path per line, sampled only from training data.",
    )
    parser.add_argument("--no-quantization", action="store_true")
    return parser.parse_args()


def require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    require_file(args.onnx, "ONNX model")
    require_file(args.calibration_list, "Calibration list")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.calibration_list.read_text(encoding="utf-8").strip():
        raise ValueError("Calibration list is empty")
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
            inputs=["input"],
            input_size_list=[[1, 3, 224, 224]],
        )
        if result != 0:
            raise RuntimeError(f"ONNX import failed: {result}")

        quantized = not args.no_quantization
        result = rknn.build(
            do_quantization=quantized,
            dataset=str(args.calibration_list) if quantized else None,
        )
        if result != 0:
            raise RuntimeError(f"RKNN build failed: {result}")

        result = rknn.export_rknn(str(args.output))
        if result != 0:
            raise RuntimeError(f"RKNN export failed: {result}")

        manifest = {
            "rknn_toolkit2_version": importlib.metadata.version("rknn-toolkit2"),
            "source_onnx": str(args.onnx.resolve()),
            "source_onnx_sha256": sha256(args.onnx),
            "output_rknn": str(args.output.resolve()),
            "output_rknn_sha256": sha256(args.output),
            "target": "rk3588",
            "input_name": "input",
            "input_shape": [1, 3, 224, 224],
            "input_color_space": "RGB",
            "source_input_layout": "NCHW",
            "runtime_input_layout": "NHWC",
            "normalization": {"mean": MEAN_VALUES[0], "std": STD_VALUES[0]},
            "quantized_int8": quantized,
            "calibration_list": str(args.calibration_list.resolve()) if quantized else None,
        }
        args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"RKNN model written to: {args.output}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
