import argparse
import json
import hashlib

import torch
from pathlib import Path
from torchvision import models
import torch.nn as nn


IMG_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/exp005_mobilenetv3_full_best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mobilenetv3_birds.onnx"))
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(num_classes):
    model = models.mobilenet_v3_large(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    num_classes = checkpoint["num_classes"]

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    external_data_path = Path(str(args.output) + ".data")
    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx": str(args.output.resolve()),
        "onnx_sha256": sha256(args.output),
        "onnx_data": str(external_data_path.resolve()) if external_data_path.is_file() else None,
        "onnx_data_sha256": sha256(external_data_path) if external_data_path.is_file() else None,
        "num_classes": int(num_classes),
        "epoch": checkpoint.get("epoch"),
        "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
        "opset_version": 12,
        "dynamic_batch": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
