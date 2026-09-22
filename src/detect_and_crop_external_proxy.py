#!/usr/bin/env python3
"""Detect COCO birds in external scenes and write classifier-ready crops.

This is an exp016 proxy probe.  It uses a frozen generic COCO detector and does
not claim species, camera, board, or outdoor performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from yolox_contract import decode_birds as decode_yolox_contract
from yolox_contract import padded_box as padded_yolox_box
from yolox_contract import preprocess_onnx as preprocess_yolox_onnx


COCO_BIRD_INDEX = 14
DETECTOR_METADATA = {
    "yolov8-coco": {
        "name": "YOLOv8n pretrained on COCO",
        "license": "AGPL-3.0",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "yolox-coco": {
        "name": "YOLOX pretrained on COCO",
        "license": "Apache-2.0",
        "source": "https://github.com/Megvii-BaseDetection/YOLOX",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument(
        "--detector-format",
        choices=sorted(DETECTOR_METADATA),
        default="yolov8-coco",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--padding", type=float, default=0.10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def letterbox(
    image: np.ndarray, size: int, *, centered: bool
) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = np.asarray(
        Image.fromarray(image).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    )
    left = (size - resized_width) // 2 if centered else 0
    top = (size - resized_height) // 2 if centered else 0
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, scale, left, top


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    upper_left = np.maximum(one[:2], many[:, :2])
    lower_right = np.minimum(one[2:], many[:, 2:])
    overlap = np.maximum(lower_right - upper_left, 0)
    intersection = overlap[:, 0] * overlap[:, 1]
    one_area = max((one[2] - one[0]) * (one[3] - one[1]), 0)
    many_area = np.maximum(many[:, 2] - many[:, 0], 0) * np.maximum(
        many[:, 3] - many[:, 1], 0
    )
    return intersection / np.maximum(one_area + many_area - intersection, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        index = int(order[0])
        kept.append(index)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[index], boxes[remaining]) <= threshold]
    return kept


def decode_yolov8_birds(
    output: np.ndarray,
    confidence: float,
    iou: float,
    scale: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    predictions = np.asarray(output)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    if predictions.ndim != 2 or predictions.shape[1] < 5 + COCO_BIRD_INDEX:
        raise ValueError(f"Unexpected detector output shape: {output.shape}")

    scores = predictions[:, 4 + COCO_BIRD_INDEX]
    selected = scores >= confidence
    predictions, scores = predictions[selected], scores[selected]
    if not len(predictions):
        return []

    centers = predictions[:, :2]
    sizes = predictions[:, 2:4]
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

    detections = []
    for index in nms(boxes, scores, iou):
        box = boxes[index]
        detections.append(
            {
                "confidence": float(scores[index]),
                "box_xyxy": [float(value) for value in box],
                "area_fraction": float(
                    max(box[2] - box[0], 0) * max(box[3] - box[1], 0) / (width * height)
                ),
            }
        )
    return detections


def decode_yolox_birds(
    output: np.ndarray,
    confidence: float,
    iou: float,
    input_size: int,
    scale: float,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    return decode_yolox_contract(
        output, confidence, iou, input_size, scale, width, height
    )


def preprocess_detector(
    image: np.ndarray, size: int, detector_format: str
) -> tuple[np.ndarray, float, int, int]:
    yolox = detector_format == "yolox-coco"
    canvas, scale, left, top = letterbox(image, size, centered=not yolox)
    if yolox:
        batch, scale = preprocess_yolox_onnx(image, size)
        left = top = 0
    else:
        batch = (
            np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        )
    return batch, scale, left, top


def padded_box(box: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    """Legacy pixel-exact crop for YOLOv8 proxy results."""
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, int(np.floor(x1 - pad_x))),
        max(0, int(np.floor(y1 - pad_y))),
        min(width, int(np.ceil(x2 + pad_x))),
        min(height, int(np.ceil(y2 + pad_y))),
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1 or args.padding < 0:
        raise ValueError("Invalid confidence, IoU, or padding")
    for path in (args.detector, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(args.detector), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_size = int(input_meta.shape[2])
    if input_meta.shape[3] != input_size:
        raise ValueError(f"Detector input must be square: {input_meta.shape}")
    detector_metadata = DETECTOR_METADATA[args.detector_format]
    output_crops = args.output_dir / "crops"
    output_annotated = args.output_dir / "annotated"
    output_crops.mkdir(parents=True)
    output_annotated.mkdir()

    crop_records: list[dict[str, object]] = []
    scene_records: list[dict[str, object]] = []
    for record in source["records"]:
        file_name = str(record["candidate_file"])
        image_path = args.image_root / file_name
        if not image_path.is_file() or sha256(image_path) != record["sha256"]:
            raise ValueError(f"Missing image or hash mismatch: {image_path}")
        pil_image = Image.open(image_path).convert("RGB")
        image = np.asarray(pil_image)
        height, width = image.shape[:2]
        batch, scale, left, top = preprocess_detector(
            image, input_size, args.detector_format
        )
        output = session.run(None, {input_meta.name: batch})[0]
        if args.detector_format == "yolox-coco":
            detections = decode_yolox_birds(
                output,
                args.confidence,
                args.iou,
                input_size,
                scale,
                width,
                height,
            )
        else:
            detections = decode_yolov8_birds(
                output, args.confidence, args.iou, scale, left, top, width, height
            )

        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        for detection_index, detection in enumerate(detections):
            crop_box = (
                padded_yolox_box(detection["box_xyxy"], width, height, args.padding)
                if args.detector_format == "yolox-coco"
                else padded_box(detection["box_xyxy"], width, height, args.padding)
            )
            crop_name = f"{Path(file_name).stem}_bird_{detection_index:02d}.jpg"
            crop_path = output_crops / crop_name
            pil_image.crop(crop_box).save(crop_path, quality=95)
            draw.rectangle(detection["box_xyxy"], outline=(255, 0, 0), width=max(2, width // 500))
            draw.text(
                (detection["box_xyxy"][0], detection["box_xyxy"][1]),
                f"bird {detection['confidence']:.2f}",
                fill=(255, 0, 0),
            )
            crop_records.append(
                {
                    "candidate_file": f"crops/{crop_name}",
                    "sha256": sha256(crop_path),
                    "source_candidate_file": file_name,
                    "source_category": record.get("source_category", "licensed_species_candidate"),
                    "class_index": record.get("class_index"),
                    "model_label": record.get("model_label"),
                    "detector_format": args.detector_format,
                    "detector_name": detector_metadata["name"],
                    "detector_confidence": detection["confidence"],
                    "detector_box_xyxy": detection["box_xyxy"],
                    "crop_box_xyxy": list(crop_box),
                    "detector_area_fraction": detection["area_fraction"],
                }
            )
        annotated_name = f"{Path(file_name).stem}_annotated.jpg"
        annotated.save(output_annotated / annotated_name, quality=90)
        scene_records.append(
            {
                "candidate_file": file_name,
                "source_sha256": record["sha256"],
                "source_category": record.get("source_category", "licensed_species_candidate"),
                "class_index": record.get("class_index"),
                "model_label": record.get("model_label"),
                "bird_detection_count": len(detections),
                "scene_status": "detected" if detections else "no_detection",
                "label_conflict_review_status": (
                    "pending" if detections and record.get("source_category") == "empty" else None
                ),
                "detections": detections,
                "annotated_file": f"annotated/{annotated_name}",
            }
        )

    scene_detection_count = sum(bool(row["bird_detection_count"]) for row in scene_records)
    known_positive = [row for row in scene_records if row["class_index"] is not None]
    empty = [row for row in scene_records if row["source_category"] == "empty"]
    summary = {
        "schema_version": 1,
        "purpose": "exp016_generic_coco_bird_detection_and_crop_probe",
        "detector_sha256": sha256(args.detector),
        "source_manifest_sha256": sha256(args.manifest),
        "detector_format": args.detector_format,
        "detector": detector_metadata["name"],
        "detector_license": detector_metadata["license"],
        "detector_source": detector_metadata["source"],
        "detector_input_size": input_size,
        "bird_class_index": COCO_BIRD_INDEX,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.iou,
        "crop_padding_fraction": args.padding,
        "scene_count": len(scene_records),
        "scene_with_detection_count": scene_detection_count,
        "scene_with_detection_rate": scene_detection_count / len(scene_records),
        "known_species_scene_count": len(known_positive),
        "known_species_scene_detection_rate": (
            sum(bool(row["bird_detection_count"]) for row in known_positive) / len(known_positive)
            if known_positive
            else None
        ),
        "empty_scene_count": len(empty),
        "empty_scene_false_detection_rate": (
            sum(bool(row["bird_detection_count"]) for row in empty) / len(empty) if empty else None
        ),
        "crop_count": len(crop_records),
        "interpretation_boundary": (
            "Generic COCO detector proxy; not trained for BirdPilot and not camera or board evidence. "
            "Wellington bird labels may describe the sequence rather than the selected frame."
        ),
    }
    (args.output_dir / "scene_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scene_records),
        encoding="utf-8",
    )
    crop_manifest = {
        "schema_version": 1,
        "purpose": "exp016_detector_crop_classifier_inputs",
        "record_count": len(crop_records),
        "records": crop_records,
    }
    (args.output_dir / "crop_manifest.json").write_text(
        json.dumps(crop_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
