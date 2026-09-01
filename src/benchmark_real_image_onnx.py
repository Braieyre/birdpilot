import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms


# =========================
# 1. 路径配置
# =========================
IMG_PATH = "data/birds/valid/ABBOTTS BOOBY/1.jpg"   # 改成你实际存在的一张图
ONNX_PATH = "outputs/mobilenetv3_birds.onnx"
LABEL_MAP_PATH = "outputs/exp005_mobilenetv3_full_idx_to_label.json"

NUM_RUNS = 100
WARMUP_RUNS = 10


# =========================
# 2. 预处理
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


def load_label_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(img_path: str):
    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).numpy()
    return x


def main():
    img_path = Path(IMG_PATH)
    onnx_path = Path(ONNX_PATH)
    label_map_path = Path(LABEL_MAP_PATH)

    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在: {img_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"label map 不存在: {label_map_path}")

    idx_to_label = load_label_map(str(label_map_path))
    session = ort.InferenceSession(str(onnx_path))

    # -------------------------
    # warmup
    # -------------------------
    for _ in range(WARMUP_RUNS):
        x = preprocess_image(str(img_path))
        _ = session.run(None, {"input": x})

    # -------------------------
    # benchmark
    # -------------------------
    load_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    final_pred_idx = None
    final_pred_label = None

    for _ in range(NUM_RUNS):
        total_start = time.perf_counter()

        # 1) load
        t0 = time.perf_counter()
        image = Image.open(img_path).convert("RGB")
        t1 = time.perf_counter()

        # 2) preprocess
        x = transform(image).unsqueeze(0).numpy()
        t2 = time.perf_counter()

        # 3) inference
        outputs = session.run(None, {"input": x})
        t3 = time.perf_counter()

        # 4) postprocess
        pred_idx = int(np.argmax(outputs[0]))
        pred_label = idx_to_label[str(pred_idx)]
        t4 = time.perf_counter()

        total_end = time.perf_counter()

        load_times.append((t1 - t0) * 1000)
        preprocess_times.append((t2 - t1) * 1000)
        inference_times.append((t3 - t2) * 1000)
        postprocess_times.append((t4 - t3) * 1000)
        total_times.append((total_end - total_start) * 1000)

        final_pred_idx = pred_idx
        final_pred_label = pred_label

    avg_load = sum(load_times) / len(load_times)
    avg_preprocess = sum(preprocess_times) / len(preprocess_times)
    avg_inference = sum(inference_times) / len(inference_times)
    avg_postprocess = sum(postprocess_times) / len(postprocess_times)
    avg_total = sum(total_times) / len(total_times)
    fps = 1000 / avg_total

    print("=" * 60)
    print(f"Image path          : {img_path}")
    print(f"Predicted index     : {final_pred_idx}")
    print(f"Predicted label     : {final_pred_label}")
    print("-" * 60)
    print(f"Average load time       : {avg_load:.3f} ms")
    print(f"Average preprocess time : {avg_preprocess:.3f} ms")
    print(f"Average inference time  : {avg_inference:.3f} ms")
    print(f"Average postprocess time: {avg_postprocess:.3f} ms")
    print(f"Average total time      : {avg_total:.3f} ms")
    print(f"Approx FPS              : {fps:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()