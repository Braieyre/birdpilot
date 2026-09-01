# exp007_real_image_onnx_benchmark

## Goal

Measure end-to-end single-image ONNX inference latency on a real bird image.

Pipeline:

image load → preprocess → ONNX inference → postprocess

---

## Model

- Model: MobileNetV3
- Source checkpoint: `outputs/exp005_mobilenetv3_full_best.pt`
- Exported ONNX: `outputs/mobilenetv3_birds.onnx`

---

## Test Image

- Image path: `data/birds/test/ABYSSINIAN GROUND HORNBILL/1.jpg`

Prediction result:

- Predicted index: `2`
- Predicted label: `ABYSSINIAN GROUND HORNBILL`

---

## Benchmark Result

- Average load time: **0.500 ms**
- Average preprocess time: **0.495 ms**
- Average inference time: **5.680 ms**
- Average postprocess time: **0.015 ms**
- Average total time: **6.690 ms**
- Approx FPS: **149.47**

---

## Observation

The ONNX model runs correctly on a real image and produces the correct bird species label.

Most of the latency comes from model inference itself, while image loading, preprocessing, and postprocessing contribute very little overhead.

This indicates that the current MobileNetV3 + ONNX pipeline is already efficient enough for edge-side single-image inference.

---

## Conclusion

The BirdPilot project has completed:

- training validation
- lightweight model selection
- ONNX export
- PyTorch / ONNX consistency verification
- real-image end-to-end local benchmark

The next step is to move toward RK3588 deployment and board-side inference validation.