"""Shared geometry and change-gate contract for close feeder views."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def validate_normalized_roi(values: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("ROI must contain x1 y1 x2 y2")
    x1, y1, x2, y2 = (float(value) for value in values)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("ROI must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return x1, y1, x2, y2


def roi_pixels(
    normalized_roi: list[float] | tuple[float, ...], width: int, height: int
) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x1, y1, x2, y2 = validate_normalized_roi(normalized_roi)
    return (
        int(np.floor(x1 * width)),
        int(np.floor(y1 * height)),
        int(np.ceil(x2 * width)),
        int(np.ceil(y2 * height)),
    )


def change_metrics(
    current: Image.Image,
    reference: Image.Image,
    normalized_roi: list[float] | tuple[float, ...],
    *,
    pixel_delta: float = 0.12,
    blur_radius: float = 2.0,
) -> dict[str, float]:
    """Measure exposure-tolerant image change inside a fixed normalized ROI.

    The reference is resized to the current frame before both crops are blurred.
    Results are descriptive signals; product thresholds must be frozen on frames
    from the intended camera.
    """

    if not 0 <= pixel_delta <= 1:
        raise ValueError("pixel_delta must be between 0 and 1")
    if blur_radius < 0:
        raise ValueError("blur_radius must be nonnegative")
    frame = current.convert("RGB")
    background = reference.convert("RGB").resize(frame.size, Image.Resampling.BILINEAR)
    box = roi_pixels(normalized_roi, frame.width, frame.height)
    frame_array = np.asarray(
        frame.crop(box).filter(ImageFilter.GaussianBlur(blur_radius)), dtype=np.float32
    )
    background_array = np.asarray(
        background.crop(box).filter(ImageFilter.GaussianBlur(blur_radius)), dtype=np.float32
    )
    difference = np.abs(frame_array - background_array) / 255.0
    per_pixel = difference.mean(axis=2)
    return {
        "mean_absolute_change": float(per_pixel.mean()),
        "changed_pixel_fraction": float((per_pixel >= pixel_delta).mean()),
        "pixel_delta": float(pixel_delta),
    }


def change_gate(
    metrics: dict[str, float], *, mean_threshold: float, fraction_threshold: float
) -> bool:
    if not 0 <= mean_threshold <= 1 or not 0 <= fraction_threshold <= 1:
        raise ValueError("Change thresholds must be between 0 and 1")
    return bool(
        metrics["mean_absolute_change"] >= mean_threshold
        or metrics["changed_pixel_fraction"] >= fraction_threshold
    )
