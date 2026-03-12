from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# 允许读取部分轻微损坏的图片，减少训练中断概率
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# 0. 路径与常量
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "birds"
DEFAULT_CSV_PATH = DEFAULT_DATA_ROOT / "birds.csv"

LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "labels"
FILEPATH_COL = "filepaths"
SPLIT_COL = "data set"
VALID_SPLITS = {"train", "valid", "test"}


# =========================================================
# 1. 训练配置
# =========================================================
@dataclass
class TrainConfig:
    experiment_name: str = "exp004_resnet18_full_5090"

    data_root: str = str(DEFAULT_DATA_ROOT)
    csv_path: str = str(DEFAULT_CSV_PATH)

    model_name: str = "resnet18"
    pretrained: bool = True
    img_size: int = 224

    # 对标 5090 的默认参数
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1

    num_workers: int = 8
    seed: int = 42

    # AMP 混合精度
    use_amp: bool = True

    # 是否保存最后一轮 checkpoint
    save_last_checkpoint: bool = True


# =========================================================
# 2. 工具函数
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        # 固定输入尺寸的卷积任务，开启 benchmark 通常更快
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_config_json(config: TrainConfig, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)


def append_metrics_to_csv(csv_path: Path, row: dict) -> None:
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =========================================================
# 3. 数据集
# =========================================================
class BirdDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        label_to_idx: dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = self.data_root / row[FILEPATH_COL]
        label_name = row[LABEL_COL]
        label_idx = self.label_to_idx[label_name]

        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx


# =========================================================
# 4. 数据增强 / 预处理
# =========================================================
def build_transforms(cfg: TrainConfig):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.img_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, eval_tf


# =========================================================
# 5. 构建 DataLoader
# =========================================================
def build_dataloaders(cfg: TrainConfig, device: torch.device):
    data_root = Path(cfg.data_root)
    csv_path = Path(cfg.csv_path)

    if not data_root.exists():
        raise FileNotFoundError(f"找不到数据目录: {data_root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {LABEL_COL, FILEPATH_COL, SPLIT_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}")

    # 只保留 train / valid / test
    df = df[df[SPLIT_COL].isin(VALID_SPLITS)].copy()

    # 过滤缺失文件
    full_paths = df[FILEPATH_COL].apply(lambda x: data_root / x)
    exists_mask = full_paths.apply(lambda p: p.exists())

    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"警告：发现 {missing_count} 条记录对应的图片文件不存在，已自动过滤。")
        examples = df.loc[~exists_mask, FILEPATH_COL].head(10).tolist()
        for fp in examples:
            print(f"  - {fp}")

    df = df.loc[exists_mask].copy()
    df.reset_index(drop=True, inplace=True)

    # 固定排序，保证类别映射可复现
    classes = sorted(df[LABEL_COL].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_df = df[df[SPLIT_COL] == "train"].copy()
    valid_df = df[df[SPLIT_COL] == "valid"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()

    train_tf, eval_tf = build_transforms(cfg)

    train_ds = BirdDataset(train_df, data_root, label_to_idx, transform=train_tf)
    valid_ds = BirdDataset(valid_df, data_root, label_to_idx, transform=eval_tf)
    test_ds = BirdDataset(test_df, data_root, label_to_idx, transform=eval_tf)

    use_pin_memory = device.type == "cuda"
    use_persistent_workers = cfg.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    print("=" * 72)
    print("数据集加载完成")
    print(f"CSV_PATH       : {csv_path}")
    print(f"DATA_ROOT      : {data_root}")
    print(f"num_classes    : {len(classes)}")
    print(f"train size     : {len(train_ds)}")
    print(f"valid size     : {len(valid_ds)}")
    print(f"test size      : {len(test_ds)}")
    print(f"batch_size     : {cfg.batch_size}")
    print(f"num_workers    : {cfg.num_workers}")
    print("=" * 72)

    return train_loader, valid_loader, test_loader, label_to_idx, idx_to_label, len(classes)


# =========================================================
# 6. 构建模型
# =========================================================
def build_model(cfg: TrainConfig, num_classes: int) -> nn.Module:
    if cfg.model_name.lower() != "resnet18":
        raise ValueError(f"当前脚本仅实现 resnet18，收到: {cfg.model_name}")

    if cfg.pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# =========================================================
# 7. 单轮训练 / 验证
# =========================================================
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    autocast_enabled = use_amp and device.type == "cuda"

    for images, labels in loader:
        if device.type == "cuda":
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = images.contiguous(memory_format=torch.channels_last)
        else:
            images = images.to(device)
            labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            amp_context = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if autocast_enabled
                else nullcontext()
            )

            with amp_context:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        if device.type == "cuda":
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = images.contiguous(memory_format=torch.channels_last)
        else:
            images = images.to(device)
            labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples

    return avg_loss, avg_acc


# =========================================================
# 8. checkpoint
# =========================================================
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_valid_acc: float,
    cfg: TrainConfig,
    num_classes: int,
    label_to_idx: dict[str, int],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_valid_acc": best_valid_acc,
        "num_classes": num_classes,
        "config": asdict(cfg),
        "class_to_idx": label_to_idx,
    }
    torch.save(checkpoint, path)


# =========================================================
# 9. 主流程
# =========================================================
def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    device = get_device()
    print(f"当前设备: {device}")
    if device.type == "cuda":
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print("当前为 Apple MPS；此脚本主要面向云端 CUDA 训练。")
    else:
        print("当前使用 CPU，速度会比较慢。")

    train_loader, valid_loader, test_loader, label_to_idx, idx_to_label, num_classes = build_dataloaders(cfg, device)

    model = build_model(cfg, num_classes=num_classes)

    if device.type == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-6,
    )

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    metrics_csv_path = LOGS_DIR / f"{cfg.experiment_name}_metrics.csv"
    best_ckpt_path = OUTPUTS_DIR / f"{cfg.experiment_name}_best.pt"
    last_ckpt_path = OUTPUTS_DIR / f"{cfg.experiment_name}_last.pt"
    label_map_path = OUTPUTS_DIR / f"{cfg.experiment_name}_idx_to_label.json"
    config_json_path = CONFIGS_DIR / f"{cfg.experiment_name}.json"
    summary_path = OUTPUTS_DIR / f"{cfg.experiment_name}_summary.json"

    if metrics_csv_path.exists():
        metrics_csv_path.unlink()

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)

    save_config_json(cfg, config_json_path)

    print(f"类别映射已保存到: {label_map_path}")
    print(f"实验配置已保存到: {config_json_path}")
    print(f"CSV日志将保存到: {metrics_csv_path}")
    print("=" * 72)

    best_valid_acc = -1.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )

        valid_loss, valid_acc = run_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        metrics_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc * 100, 4),
            "valid_loss": round(valid_loss, 6),
            "valid_acc": round(valid_acc * 100, 4),
            "lr": current_lr,
            "time_sec": round(epoch_time, 2),
        }
        append_metrics_to_csv(metrics_csv_path, metrics_row)

        print(
            f"[Epoch {epoch:02d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4%} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_acc={valid_acc:.4%} "
            f"lr={current_lr:.6f} "
            f"time={epoch_time:.1f}s"
        )

        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_valid_acc=best_valid_acc,
                cfg=cfg,
                num_classes=num_classes,
                label_to_idx=label_to_idx,
            )
            print(f"✅ 已保存最佳模型到: {best_ckpt_path}")

        if cfg.save_last_checkpoint:
            save_checkpoint(
                path=last_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_valid_acc=best_valid_acc,
                cfg=cfg,
                num_classes=num_classes,
                label_to_idx=label_to_idx,
            )

        scheduler.step()

    print("=" * 72)
    print(f"训练完成，最佳 valid_acc = {best_valid_acc:.4%}（epoch {best_epoch}）")

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate_test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4%}")
    print("=" * 72)

    summary = {
        "experiment_name": cfg.experiment_name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "best_epoch": best_epoch,
        "best_valid_acc": round(best_valid_acc * 100, 4),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc * 100, 4),
        "num_classes": num_classes,
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path) if cfg.save_last_checkpoint else None,
        "metrics_csv": str(metrics_csv_path),
        "label_map": str(label_map_path),
        "config_json": str(config_json_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"实验总结已保存到: {summary_path}")
    print("全部完成。")


# =========================================================
# 10. 命令行参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train BirdPilot on full birds dataset.")

    parser.add_argument("--experiment-name", type=str, default="exp004_resnet18_full_5090")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))

    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-save-last", action="store_true")

    return parser.parse_args()


def build_config_from_args(args) -> TrainConfig:
    return TrainConfig(
        experiment_name=args.experiment_name,
        data_root=args.data_root,
        csv_path=args.csv_path,
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=not args.no_amp,
        save_last_checkpoint=not args.no_save_last,
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config_from_args(args)
    train(cfg)