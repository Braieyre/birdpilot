## exp006_onnx_export_validation

Model: MobileNetV3
Checkpoint: exp005_mobilenetv3_full_best.pt

Steps:
- Exported PyTorch checkpoint to ONNX
- Verified ONNX prediction matches PyTorch prediction on the same image
- Benchmarked ONNX inference on local Mac

Results:
- PyTorch pred == ONNX pred
- Average ONNX latency: 5.24 ms
- Approx FPS: 191.00

Conclusion:
The model is successfully portable and suitable for further edge deployment tests.