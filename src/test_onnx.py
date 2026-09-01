import json
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

IMG_PATH = "data/birds/test/ABYSSINIAN GROUND HORNBILL/1.jpg"   # 改成你的实际图片
ONNX_PATH = "outputs/mobilenetv3_birds.onnx"
LABEL_MAP_PATH = "outputs/exp005_mobilenetv3_full_idx_to_label.json"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    idx_to_label = json.load(f)

image = Image.open(IMG_PATH).convert("RGB")
x = transform(image).unsqueeze(0).numpy()

session = ort.InferenceSession(ONNX_PATH)
outputs = session.run(None, {"input": x})

pred_idx = int(np.argmax(outputs[0]))
pred_label = idx_to_label[str(pred_idx)]

print("predicted class index:", pred_idx)
print("predicted class label:", pred_label)