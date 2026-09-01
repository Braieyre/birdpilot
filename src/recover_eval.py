"""
恢复实验数据：从 checkpoint 读取训练信息 + 重新评估 test set。

用法：
    python src/recover_eval.py
    python src/recover_eval.py --checkpoint models/exp005_mobilenetv3_full_best.pt
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "exp005_mobilenetv3_full_best.pt"
DATA_ROOT = PROJECT_ROOT / "data" / "birds"
CSV_PATH = DATA_ROOT / "birds.csv"

LABEL_COL = "labels"
FILEPATH_COL = "filepaths"
SPLIT_COL = "data set"


# =========================================================
# 1. 从 checkpoint 恢复训练信息
# =========================================================
def recover_from_checkpoint(ckpt_path: Path):
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 只输出标量字段
    SKIP_TYPES = (dict, list, type(None))
    for k, v in ckpt.items():
        if isinstance(v, SKIP_TYPES):
            continue
        print(f"  {k}: {v}")

    # 读取 config 中的关键字段
    cfg = ckpt.get("config")
    if cfg:
        print(f"\n[训练配置]")
        KEYS = ["model_name", "epochs", "batch_size", "lr", "pretrained",
                "label_smoothing", "use_amp", "img_size", "seed"]
        for k in KEYS:
            if k in cfg:
                print(f"  {k}: {cfg[k]}")

    return ckpt


# =========================================================
# 2. 构建模型 (匹配 train_full.py 的逻辑)
# =========================================================
def build_model(model_name: str, num_classes: int):
    name = model_name.lower()
    if name == "resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenetv3":
        model = models.mobilenet_v3_large()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


# =========================================================
# 3. 数据集 (与 train_full.py 一致)
# =========================================================
class BirdDataset(Dataset):
    def __init__(self, df, data_root, label_to_idx, transform):
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_root / row[FILEPATH_COL]
        label_idx = self.label_to_idx[row[LABEL_COL]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx


# =========================================================
# 4. 评估 test set
# =========================================================
@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    running_correct = 0
    total_samples = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    acc = running_correct / total_samples
    avg_loss = running_loss / total_samples
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--skip-test", action="store_true",
                        help="只读取 checkpoint 信息，不跑 test 评估")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint 不存在: {ckpt_path}")
        return

    # --- 1) 从 checkpoint 恢复信息 ---
    ckpt = recover_from_checkpoint(ckpt_path)

    if args.skip_test:
        print("\n(--skip-test: 跳过 test 评估)")
        return

    # --- 2) 准备数据 ---
    num_classes = ckpt.get("num_classes")
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", "mobilenetv3")

    if num_classes is None:
        print("ERROR: checkpoint 中没有 num_classes，无法重建模型")
        return

    print(f"\n[模型] {model_name}, num_classes={num_classes}")

    if not CSV_PATH.exists():
        print(f"ERROR: CSV 不存在: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df[SPLIT_COL].isin(["train", "valid", "test"])]

    # 过滤缺失文件
    data_root = DATA_ROOT
    full_paths = df[FILEPATH_COL].apply(lambda x: data_root / x)
    exists_mask = full_paths.apply(lambda p: p.exists())
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"过滤 {missing} 条缺失文件记录")
    df = df.loc[exists_mask].copy()

    classes = sorted(df[LABEL_COL].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    actual_num_classes = len(classes)
    if actual_num_classes != num_classes:
        print(f"⚠️  checkpoint 记录 {num_classes} 类，当前数据有 {actual_num_classes} 类")

    test_df = df[df[SPLIT_COL] == "test"].copy()

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_ds = BirdDataset(test_df, data_root, label_to_idx, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print(f"Test set: {len(test_ds)} images, {actual_num_classes} classes")

    # --- 3) 加载模型 ---
    model = build_model(model_name, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    # --- 4) 评估 ---
    print("\n评估中...")
    test_loss, test_acc = evaluate_test(model, test_loader, device)

    print("=" * 60)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
