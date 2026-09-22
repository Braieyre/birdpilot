#!/usr/bin/env python3
"""Compare YOLOX ONNX and RKNN simulator outputs on a fixed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from rknn.api import RKNN

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from yolox_contract import box_iou, decode_birds, padded_box, preprocess_onnx, preprocess_rknn  # noqa: E402

CLASSIFIER_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
CLASSIFIER_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def classify_crop(session: ort.InferenceSession, image: Image.Image, box: tuple[int, int, int, int]) -> int:
    crop = image.crop(box).resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(crop, dtype=np.float32) / 255.0
    tensor = ((array - CLASSIFIER_MEAN) / CLASSIFIER_STD).transpose(2, 0, 1)[None]
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return int(np.asarray(logits).reshape(-1).argmax())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--classifier-onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text())["records"]
    onnx = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    classifier = ort.InferenceSession(
        str(args.classifier_onnx), providers=["CPUExecutionProvider"]
    )
    input_name = onnx.get_inputs()[0].name
    rknn = RKNN(verbose=False)
    # Toolkit2 2.0.0 cannot initialize its simulator from an exported RKNN file.
    # Rebuild the exact FP16 graph in memory; the separately exported file hash is
    # still recorded so the board can verify the artifact produced by the same contract.
    if rknn.config(
        target_platform="rk3588",
        mean_values=[[0.0, 0.0, 0.0]],
        std_values=[[1.0, 1.0, 1.0]],
        optimization_level=3,
    ) != 0:
        raise RuntimeError("RKNN simulator config failed")
    if rknn.load_onnx(
        model=str(args.onnx), inputs=["images"], input_size_list=[[1, 3, 416, 416]]
    ) != 0:
        raise RuntimeError("RKNN simulator ONNX import failed")
    if rknn.build(do_quantization=False) != 0 or rknn.init_runtime() != 0:
        raise RuntimeError("RKNN simulator initialization failed")
    comparisons = []
    try:
        for source in records:
            path = args.image_root / source["candidate_file"]
            if sha256(path) != source["sha256"]:
                raise ValueError(f"Hash mismatch: {path}")
            pil = Image.open(path).convert("RGB")
            image = np.asarray(pil)
            onnx_input, scale = preprocess_onnx(image)
            rknn_input, rknn_scale = preprocess_rknn(image)
            onnx_output = onnx.run(None, {input_name: onnx_input})[0]
            rknn_output = rknn.inference(inputs=[rknn_input], data_format="nhwc")[0]
            onnx_det = decode_birds(onnx_output, args.confidence, args.iou, 416, scale, pil.width, pil.height)
            rknn_det = decode_birds(rknn_output, args.confidence, args.iou, 416, rknn_scale, pil.width, pil.height)
            onnx_best = onnx_det[0] if onnx_det else None
            rknn_best = rknn_det[0] if rknn_det else None
            matched_iou = None
            crop_equal = onnx_best is None and rknn_best is None
            onnx_top1 = None
            rknn_top1 = None
            if onnx_best and rknn_best:
                matched_iou = float(box_iou(
                    np.asarray(onnx_best["box_xyxy"]), np.asarray([rknn_best["box_xyxy"]])
                )[0])
                onnx_crop = padded_box(onnx_best["box_xyxy"], pil.width, pil.height)
                rknn_crop = padded_box(rknn_best["box_xyxy"], pil.width, pil.height)
                crop_equal = onnx_crop == rknn_crop
                onnx_top1 = classify_crop(classifier, pil, onnx_crop)
                rknn_top1 = classify_crop(classifier, pil, rknn_crop)
            comparisons.append({
                "file": path.name,
                "onnx_detected": onnx_best is not None,
                "rknn_detected": rknn_best is not None,
                "gate_consistent": (onnx_best is None) == (rknn_best is None),
                "best_box_iou": matched_iou,
                "crop_box_exact": crop_equal,
                "onnx_crop_classifier_top1": onnx_top1,
                "rknn_crop_classifier_top1": rknn_top1,
                "crop_classifier_top1_consistent": onnx_top1 == rknn_top1,
                "onnx_best": onnx_best,
                "rknn_best": rknn_best,
                "threshold_near": bool(
                    (onnx_best and abs(float(onnx_best["confidence"]) - args.confidence) <= 0.05)
                    or (rknn_best and abs(float(rknn_best["confidence"]) - args.confidence) <= 0.05)
                ),
            })
    finally:
        rknn.release()
    paired = [row for row in comparisons if row["best_box_iou"] is not None]
    summary = {
        "schema_version": 1,
        "onnx_sha256": sha256(args.onnx),
        "rknn_sha256": sha256(args.rknn),
        "record_count": len(comparisons),
        "gate_consistent_count": sum(row["gate_consistent"] for row in comparisons),
        "minimum_best_box_iou": min((row["best_box_iou"] for row in paired), default=None),
        "all_matched_boxes_iou_at_least_0_95": all(row["best_box_iou"] >= 0.95 for row in paired),
        "highest_score_selection_consistent": all(row["gate_consistent"] and (row["best_box_iou"] is None or row["best_box_iou"] >= 0.95) for row in comparisons),
        "crop_classifier_top1_consistent": all(
            row["crop_classifier_top1_consistent"] for row in comparisons
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "record_count", "gate_consistent_count", "minimum_best_box_iou",
        "all_matched_boxes_iou_at_least_0_95", "highest_score_selection_consistent",
        "crop_classifier_top1_consistent")}, indent=2))
    if (summary["gate_consistent_count"] != len(comparisons)
            or not summary["all_matched_boxes_iou_at_least_0_95"]
            or not summary["crop_classifier_top1_consistent"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
