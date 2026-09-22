"""Frozen YOLOX preprocessing and bird-only postprocessing contract."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image


COCO_BIRD_INDEX = 14


def letterbox_top_left(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    height, width = image_rgb.shape[:2]
    scale = min(size / height, size / width)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = np.asarray(
        Image.fromarray(image_rgb).resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    return canvas, scale


def preprocess_onnx(image_rgb: np.ndarray, size: int = 416) -> tuple[np.ndarray, float]:
    canvas, scale = letterbox_top_left(image_rgb, size)
    bgr = canvas[:, :, ::-1]
    return np.ascontiguousarray(bgr.transpose(2, 0, 1)[None], dtype=np.float32), scale


def preprocess_rknn(image_rgb: np.ndarray, size: int = 416) -> tuple[np.ndarray, float]:
    canvas, scale = letterbox_top_left(image_rgb, size)
    return np.ascontiguousarray(canvas[:, :, ::-1][None], dtype=np.uint8), scale


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


def normalize_output(output: np.ndarray) -> np.ndarray:
    predictions = np.asarray(output).copy()
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected YOLOX output shape: {np.asarray(output).shape}")
    if predictions.shape == (85, 3549):
        predictions = predictions.T
    if predictions.shape != (3549, 85):
        raise ValueError(f"Unexpected YOLOX output shape: {np.asarray(output).shape}")
    return predictions.astype(np.float32, copy=False)


def decode_birds(
    output: np.ndarray,
    confidence: float,
    iou: float,
    input_size: int,
    scale: float,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    predictions = normalize_output(output)
    grids = []
    expanded_strides = []
    for stride in (8, 16, 32):
        size = input_size // stride
        grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
        grid = np.stack((grid_x, grid_y), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((len(grid), 1), stride))
    grid = np.concatenate(grids, axis=0)
    strides = np.concatenate(expanded_strides, axis=0)
    if len(predictions) != len(grid):
        raise ValueError("YOLOX output location count does not match the input size")
    predictions[:, :2] = (predictions[:, :2] + grid) * strides
    predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * strides
    scores = predictions[:, 4] * predictions[:, 5 + COCO_BIRD_INDEX]
    selected = scores >= confidence
    predictions, scores = predictions[selected], scores[selected]
    if not len(predictions):
        return []
    centers, sizes = predictions[:, :2], predictions[:, 2:4]
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / scale
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
                    max(box[2] - box[0], 0)
                    * max(box[3] - box[1], 0)
                    / (width * height)
                ),
            }
        )
    return detections


def padded_box(
    box: list[float], width: int, height: int, padding: float = 0.10, grid: int = 16
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    return (
        max(0, math.floor((x1 - pad_x) / grid) * grid),
        max(0, math.floor((y1 - pad_y) / grid) * grid),
        min(width, math.ceil((x2 + pad_x) / grid) * grid),
        min(height, math.ceil((y2 + pad_y) / grid) * grid),
    )
