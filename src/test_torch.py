import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CHECKPOINT_PATH = "models/exp005_mobilenetv3_full_best.pt"
IMG_PATH = "data/birds/test/ABYSSINIAN GROUND HORNBILL/1.jpg"

def build_model(num_classes: int):
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    image = Image.open(IMG_PATH).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        pred = outputs.argmax(dim=1).item()

    print("predicted class index:", pred)

if __name__ == "__main__":
    main()