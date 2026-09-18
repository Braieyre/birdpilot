# BirdPilot RK3588 Deployment

## Goal
Convert ONNX classification model to RKNN and run on RK3588 NPU.

## Current status
- [x] PyTorch model trained
- [x] ONNX exported
- [x] ONNX benchmark verified on Mac
- [x] QuarkPi-CA2 remote development path verified (USB OTG/ADB rescue + cross-network Tailscale SSH)
- [ ] RKNN Toolkit2 environment ready
- [ ] ONNX -> RKNN conversion
- [ ] RK3588 single-image inference
- [ ] RK3588 benchmark
- [ ] INT8 quantization

The remote-access milestone is development infrastructure, not RK3588 inference evidence. See [REMOTE_ACCESS_CN.md](REMOTE_ACCESS_CN.md) for the verified environment, the CA2 kernel/TUN limitation, the userspace-networking workaround, security gates, and the path from one prototype to fleet provisioning.
