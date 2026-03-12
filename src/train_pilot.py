from pathlib import Path
import json
import random
import time

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


# =========================
# 0. 基础配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 小样本数据目录
DATA_ROOT = PROJECT_ROOT / "data" / "birds_1pct"
CSV_PATH = DATA_ROOT / "birds_small.csv"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 训练参数
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 0
SEED = 42

# 是否使用预训练权重
USE_PRETRAINED = True

# 类别列 / 路径列 / 数据集划分列
LABEL_COL = "labels"
FILEPATH_COL = "filepaths"
SPLIT_COL = "data set"


# =========================
# 1. 固定随机种子
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 2. 设备选择
# =========================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# =========================
# 3. 自定义数据集
# =========================
class BirdDataset(Dataset):
    """
    基于 CSV 读取图像与标签。
    CSV 中的 filepaths 形如：
        train/ABBOTTS BABBLER/001.jpg
    对于小样本目录 data/birds_1pct 来说，
    直接拼成：
        data/birds_1pct/train/ABBOTTS BABBLER/001.jpg
    """

    def __init__(self, df: pd.DataFrame, data_root: Path, label_to_idx: dict, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.data_root / row[FILEPATH_COL]
        label_name = row[LABEL_COL]
        label_idx = self.label_to_idx[label_name]

        # RGB 统一三通道
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx


# =========================
# 4. 数据增强 / 预处理
# =========================
def build_transforms(img_size: int = 160):
    """
    训练集做轻量增强；
    验证集只做 resize + normalize。
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    valid_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tf, valid_tf


# =========================
# 5. 构建 DataLoader
# =========================
def build_dataloaders():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_cols = {LABEL_COL, FILEPATH_COL, SPLIT_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}")

    # 只保留 train / valid / test
    df = df[df[SPLIT_COL].isin(["train", "valid", "test"])].copy()

    # -------------------------------------------------
    # 关键补丁：过滤掉 CSV 中存在，但磁盘上不存在的图片
    # -------------------------------------------------
    full_paths = df[FILEPATH_COL].apply(lambda x: DATA_ROOT / x)
    exists_mask = full_paths.apply(lambda p: p.exists())

    missing_count = (~exists_mask).sum()
    if missing_count > 0:
        print(f"警告：发现 {missing_count} 条记录对应的图片文件不存在，已自动过滤。")

        missing_examples = df.loc[~exists_mask, FILEPATH_COL].head(10).tolist()
        print("缺失文件示例：")
        for fp in missing_examples:
            print(f"  - {fp}")

    df = df.loc[exists_mask].copy()
    df.reset_index(drop=True, inplace=True)

    # 类别映射：固定排序，保证可复现
    classes = sorted(df[LABEL_COL].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_df = df[df[SPLIT_COL] == "train"].copy()
    valid_df = df[df[SPLIT_COL] == "valid"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()

    train_tf, valid_tf = build_transforms(IMG_SIZE)

    train_ds = BirdDataset(train_df, DATA_ROOT, label_to_idx, transform=train_tf)
    valid_ds = BirdDataset(valid_df, DATA_ROOT, label_to_idx, transform=valid_tf)
    test_ds = BirdDataset(test_df, DATA_ROOT, label_to_idx, transform=valid_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print("=" * 60)
    print("数据集加载完成")
    print(f"CSV_PATH     : {CSV_PATH}")
    print(f"DATA_ROOT    : {DATA_ROOT}")
    print(f"num_classes  : {len(classes)}")
    print(f"train size   : {len(train_ds)}")
    print(f"valid size   : {len(valid_ds)}")
    print(f"test size    : {len(test_ds)}")
    print("=" * 60)

    return train_loader, valid_loader, test_loader, label_to_idx, idx_to_label

# =========================
# 6. 构建模型
# =========================
def build_model(num_classes: int):
    """
    用 ResNet18 做试点训练。
    最后一层替换成 num_classes 输出。
    """
    if USE_PRETRAINED:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =========================
# 7. 训练 / 验证 一个 epoch
# =========================
def run_one_epoch(model, loader, criterion, device, optimizer=None):
    """
    optimizer 为 None 时，表示验证模式。
    返回：
        avg_loss, avg_acc
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
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


# =========================
# 8. 测试集评估
# =========================
@torch.no_grad()
def evaluate_test(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
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


# =========================
# 9. 主训练流程
# =========================
def main():
    set_seed(SEED)
    device = get_device()

    print(f"当前设备: {device}")
    if device.type == "mps":
        print("使用 Apple Silicon MPS 加速")
    elif device.type == "cuda":
        print("使用 CUDA 加速")
    else:
        print("使用 CPU 训练")

    train_loader, valid_loader, test_loader, label_to_idx, idx_to_label = build_dataloaders()

    num_classes = len(label_to_idx)
    model = build_model(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_valid_acc = 0.0
    best_model_path = OUTPUT_DIR / "resnet18_pilot_best.pt"
    label_map_path = OUTPUT_DIR / "idx_to_label.json"

    # 保存类别映射，后续推理时很有用
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)

    print(f"类别映射已保存到: {label_map_path}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        valid_loss, valid_acc = run_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        epoch_time = time.time() - start_time

        print(
            f"[Epoch {epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4%} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_acc={valid_acc:.4%} "
            f"time={epoch_time:.1f}s"
        )

        # 保存最佳模型（按 valid_acc）
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_acc": best_valid_acc,
                    "num_classes": num_classes,
                    "img_size": IMG_SIZE,
                    "class_to_idx": label_to_idx,
                },
                best_model_path,
            )
            print(f"✅ 已保存最佳模型到: {best_model_path}")

    print("=" * 60)
    print(f"训练完成，最佳 valid_acc = {best_valid_acc:.4%}")

    # 重新加载最佳模型，再做一次 test 评估
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate_test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4%}")
    print("=" * 60)
    print("全部完成。")


if __name__ == "__main__":
    main()