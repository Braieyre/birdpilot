#!/usr/bin/env python3
"""Probe whether the configured BirdPilot batch fits on the selected CUDA GPU."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.optim import AdamW

try:
    from train_robustness import build_model, configure_determinism, load_config, load_label_map, resolve
except ModuleNotFoundError:
    from src.train_robustness import build_model, configure_determinism, load_config, load_label_map, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = load_config(args.config)
    configure_determinism(config.seed)
    _, label_to_index = load_label_map(resolve(config.label_map_path))
    checkpoint = torch.load(resolve(config.init_checkpoint), map_location="cpu", weights_only=False)
    if checkpoint.get("class_to_idx") != label_to_index:
        raise ValueError("Initial checkpoint and configured label map differ")
    device = torch.device("cuda:0")
    model = build_model(len(label_to_index)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    elapsed: list[float] = []
    for step in range(args.steps):
        images = torch.randn(config.batch_size, 3, 224, 224, device=device)
        labels = torch.arange(config.batch_size, device=device) % len(label_to_index)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=config.use_amp):
            loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite probe loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - start)
    free_after, _ = torch.cuda.mem_get_info(device)
    report = {
        "schema_version": 1,
        "purpose": "disposable_random_tensor_memory_probe",
        "config": str(args.config),
        "batch_size": config.batch_size,
        "steps": args.steps,
        "device_visible_index": 0,
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "total_memory_bytes": total_memory,
        "free_memory_before_bytes": free_before,
        "free_memory_after_bytes": free_after,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "step_seconds": elapsed,
        "final_loss": float(loss.detach()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
