#!/usr/bin/env python3
"""Paired, resumable robustness fine-tuning for BirdPilot MobileNetV3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from generate_degradation_pilot import FAMILIES, LEVELS, degrade
except ModuleNotFoundError:
    from src.generate_degradation_pilot import FAMILIES, LEVELS, degrade


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class RobustnessConfig:
    run_name: str
    data_root: str
    csv_path: str
    label_map_path: str
    init_checkpoint: str
    robust_validation_manifest: str
    output_root: str
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    num_workers: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    degradation_probability: float = 0.0
    min_clean_valid_accuracy: float = 0.9808778625954199
    use_amp: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> RobustnessConfig:
    config = RobustnessConfig(**json.loads(path.read_text(encoding="utf-8")))
    if not 0 <= config.degradation_probability <= 1:
        raise ValueError("degradation_probability must be in [0, 1]")
    if config.epochs <= 0 or config.batch_size <= 0 or config.num_workers < 0:
        raise ValueError("Invalid epoch, batch, or worker count")
    return config


def load_label_map(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    index_to_label = {int(index): str(label) for index, label in raw.items()}
    if set(index_to_label) != set(range(len(index_to_label))):
        raise ValueError("Label indices are not contiguous")
    label_to_index = {label: index for index, label in index_to_label.items()}
    if len(label_to_index) != len(index_to_label):
        raise ValueError("Duplicate label name")
    return index_to_label, label_to_index


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class PairedTrainTransform:
    def __init__(self, seed: int, degradation_probability: float) -> None:
        self.seed = seed
        self.epoch = 0
        self.degradation_probability = degradation_probability
        self.ordinary = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            ]
        )
        self.finalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(self, image: Image.Image, index: int) -> torch.Tensor:
        ordinary_seed = self.seed * 1_000_003 + self.epoch * 100_003 + index
        python_state = random.getstate()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(ordinary_seed)
            random.seed(ordinary_seed)
            image = self.ordinary(image)
        random.setstate(python_state)
        degradation_rng = random.Random(ordinary_seed + 9_000_001)
        if degradation_rng.random() < self.degradation_probability:
            family = degradation_rng.choice(FAMILIES)
            level = degradation_rng.choice(tuple(LEVELS))
            image = degrade(
                image,
                family,
                LEVELS[level][family],
                degradation_rng.randrange(0, 2**32),
            )
        return self.finalize(image)


class TrainDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], root: Path, labels: dict[str, int], transform: PairedTrainTransform) -> None:
        self.rows = rows
        self.root = root
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = Image.open(self.root / row["filepaths"]).convert("RGB")
        return self.transform(image, index), self.labels[row["labels"]]


class EvalDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], labels: dict[str, int]) -> None:
        self.rows = rows
        self.labels = labels
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        image = Image.open(Path(str(row["image_path"]))).convert("RGB")
        condition = str(row["condition"])
        return self.transform(image), self.labels[str(row["label"])], condition


def read_data(config: RobustnessConfig, labels: dict[str, int]) -> tuple[list[dict[str, str]], list[dict[str, object]], list[dict[str, object]]]:
    root = resolve(config.data_root)
    with resolve(config.csv_path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    train_rows: list[dict[str, str]] = []
    valid_rows: list[dict[str, object]] = []
    for row in rows:
        split = row.get("data set")
        if split not in {"train", "valid"}:
            continue
        label = row["labels"]
        path = root / row["filepaths"]
        if label not in labels or not path.is_file():
            continue
        if split == "train":
            train_rows.append(row)
        else:
            valid_rows.append({"image_path": str(path), "label": label, "condition": "clean"})
    manifest_path = resolve(config.robust_validation_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_partition") != "valid":
        raise ValueError("Robust validation manifest is not validation-only")
    robust_rows: list[dict[str, object]] = []
    for row in manifest["records"]:
        label = str(row["source_label"])
        if label not in labels:
            raise ValueError(f"Robust manifest label not in model mapping: {label}")
        path = manifest_path.parent / str(row["output_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        robust_rows.append(
            {
                "image_path": str(path),
                "label": label,
                "condition": f"{row['family']}/{row['level']}",
            }
        )
    return train_rows, valid_rows, robust_rows


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for images, labels, conditions in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        predictions = model(images).argmax(dim=1)
        matches = (predictions == labels).cpu().tolist()
        for condition, match in zip(conditions, matches):
            totals[condition] += 1
            correct[condition] += int(match)
    condition_metrics = {
        condition: {"count": totals[condition], "correct": correct[condition], "accuracy": correct[condition] / totals[condition]}
        for condition in sorted(totals)
    }
    return {
        "conditions": condition_metrics,
        "condition_mean_accuracy": float(np.mean([value["accuracy"] for value in condition_metrics.values()])),
    }


def rng_state(sampler_generator: torch.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler": sampler_generator.get_state(),
    }


def restore_rng(state: dict[str, object], sampler_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    sampler_generator.set_state(state["sampler"])


def save_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    config: RobustnessConfig,
    labels: dict[str, int],
    sampler_generator: torch.Generator,
    best_robust: float,
    best_clean: float,
    best_epoch: int,
    clean_accuracy: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "config": asdict(config),
            "class_to_idx": labels,
            "num_classes": len(labels),
            "best_valid_acc": clean_accuracy,
            "rng_state": rng_state(sampler_generator),
            "best_robust_accuracy": best_robust,
            "best_clean_accuracy": best_clean,
            "best_epoch": best_epoch,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_determinism(config.seed)
    index_to_label, label_to_index = load_label_map(resolve(config.label_map_path))
    train_rows, valid_rows, robust_rows = read_data(config, label_to_index)
    if not train_rows or not valid_rows or not robust_rows:
        raise ValueError("A required train or validation dataset is empty")
    run_dir = resolve(config.output_root) / config.run_name
    metrics_path = run_dir / "metrics.jsonl"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    if args.resume is None:
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
    else:
        if resolve(str(args.resume)) != last_path.resolve():
            raise ValueError("Resume must use this run's last.pt")
        if not run_dir.is_dir() or not metrics_path.is_file():
            raise FileNotFoundError("Resume run directory or metrics is missing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = PairedTrainTransform(config.seed, config.degradation_probability)
    train_dataset = TrainDataset(train_rows, resolve(config.data_root), label_to_index, train_transform)
    clean_dataset = EvalDataset(valid_rows, label_to_index)
    robust_dataset = EvalDataset(robust_rows, label_to_index)
    sampler_generator = torch.Generator().manual_seed(config.seed + 7001)
    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "worker_init_fn": worker_seed,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=sampler_generator, **loader_args)
    clean_loader = DataLoader(clean_dataset, shuffle=False, **loader_args)
    robust_loader = DataLoader(robust_dataset, shuffle=False, **loader_args)

    model = build_model(len(index_to_label))
    init_checkpoint = torch.load(resolve(config.init_checkpoint), map_location="cpu", weights_only=False)
    if init_checkpoint.get("class_to_idx") != label_to_index:
        raise ValueError("Initial checkpoint and configured label map differ")
    model.load_state_dict(init_checkpoint["model_state_dict"])
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    start_epoch = 1
    best_robust = -1.0
    best_clean = -1.0
    best_epoch = 0
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config") != asdict(config) or checkpoint.get("class_to_idx") != label_to_index:
            raise ValueError("Resume checkpoint does not match immutable config or labels")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng(checkpoint["rng_state"], sampler_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_robust = float(checkpoint["best_robust_accuracy"])
        best_clean = float(checkpoint.get("best_clean_accuracy", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))

    if args.stop_after_epoch is not None and not start_epoch <= args.stop_after_epoch <= config.epochs:
        raise ValueError("stop-after-epoch is outside the remaining planned epochs")
    stop_epoch = args.stop_after_epoch or config.epochs
    if args.resume is None:
        manifest = {
            "schema_version": 1,
            "config": asdict(config),
            "config_sha256": sha256(args.config),
            "init_checkpoint_sha256": sha256(resolve(config.init_checkpoint)),
            "csv_sha256": sha256(resolve(config.csv_path)),
            "label_map_sha256": sha256(resolve(config.label_map_path)),
            "robust_validation_manifest_sha256": sha256(resolve(config.robust_validation_manifest)),
            "train_count": len(train_rows),
            "clean_valid_count": len(valid_rows),
            "robust_valid_count": len(robust_rows),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (run_dir / "labels.json").write_text(
            json.dumps(index_to_label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for epoch in range(start_epoch, stop_epoch + 1):
        start = time.time()
        train_transform.set_epoch(epoch)
        model.train()
        total = 0
        correct_sum = torch.zeros((), dtype=torch.long, device=device)
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        for images, labels in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            context = torch.amp.autocast("cuda", enabled=True) if amp_enabled else nullcontext()
            with context:
                output = model(images)
                loss = criterion(output, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total += labels.numel()
            correct_sum += (output.argmax(dim=1) == labels).sum()
            loss_sum += loss.detach().to(torch.float64) * labels.numel()
        clean = evaluate(model, clean_loader, device)
        robust = evaluate(model, robust_loader, device)
        clean_accuracy = clean["conditions"]["clean"]["accuracy"]
        robust_accuracy = robust["condition_mean_accuracy"]
        eligible = clean_accuracy >= config.min_clean_valid_accuracy
        record = {
            "epoch": epoch,
            "train_loss": float((loss_sum / total).cpu()),
            "train_accuracy": float((correct_sum / total).cpu()),
            "clean_validation_accuracy": clean_accuracy,
            "robust_condition_mean_accuracy": robust_accuracy,
            "eligible": eligible,
            "robust_conditions": robust["conditions"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - start,
        }
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        scheduler.step()
        candidate_key = (robust_accuracy, clean_accuracy, -epoch)
        best_key = (best_robust, best_clean, -best_epoch)
        if eligible and candidate_key > best_key:
            best_robust = robust_accuracy
            best_clean = clean_accuracy
            best_epoch = epoch
            save_state(
                best_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                config,
                label_to_index,
                sampler_generator,
                best_robust,
                best_clean,
                best_epoch,
                clean_accuracy,
            )
        save_state(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            config,
            label_to_index,
            sampler_generator,
            best_robust,
            best_clean,
            best_epoch,
            clean_accuracy,
        )
        print(json.dumps({key: record[key] for key in ("epoch", "train_loss", "clean_validation_accuracy", "robust_condition_mean_accuracy", "eligible", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
