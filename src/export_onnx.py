import torch
from pathlib import Path
from torchvision import models
import torch.nn as nn


CHECKPOINT_PATH = "models/exp005_mobilenetv3_full_best.pt"
ONNX_PATH = "outputs/mobilenetv3_birds.onnx"
IMG_SIZE = 224


def build_model(num_classes):
    model = models.mobilenet_v3_large(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def main():

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    num_classes = checkpoint["num_classes"]

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    print("ONNX model exported to:", ONNX_PATH)


if __name__ == "__main__":
    main()