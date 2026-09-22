# BirdPilot RK3588 Deployment

## Goal
Convert ONNX classification model to RKNN and run on RK3588 NPU.

## Current status
- [x] PyTorch model trained
- [x] ONNX exported
- [x] ONNX benchmark verified on Mac
- [x] QuarkPi-CA2 remote development path verified (USB OTG/ADB rescue + cross-network Tailscale SSH)
- [x] Version-matched RKNN Toolkit2 environment ready
- [x] Accepted ONNX -> FP16 RKNN conversion
- [x] RK3588 single-image inference
- [x] RK3588 20-image ONNX parity and latency benchmark
- [x] External-scene proxy confirms detection/cropping is required
- [x] Detector/crop ONNX-to-RKNN local simulator parity
- [ ] Detector/crop real-board NPU parity and latency
- [ ] Physical camera enumeration and one-frame capture
- [ ] Bird/no-bird gating and observation loop
- [ ] INT8 quantization

The accepted exp014 model is installed under a versioned path on the real
board, while the previous FP16 model remains available for rollback. It
matches ONNX top-1 on 20/20 fixed validation images. The board currently has
RKISP device nodes but no sensor attached to the media pipeline, so camera
evidence begins only after a real frame can be captured. Exp016 also shows
that a complete scene must first pass through bird detection and cropping;
the closed-set species classifier cannot act as a bird/no-bird gate.

## First board-baseline workflow

The first device milestone is deliberately narrow: use the accepted local
MobileNetV3 model to recognise a local image on the CA2, then save the raw
prediction and timing evidence. It does not include camera, field, or outdoor
claims.

1. On a conversion machine with RKNN-Toolkit2, generate a deterministic list
   of training-only calibration images with `make_calibration_list.py`. Pass a
   new `--staging-dir` when the dataset has spaces in bird names: it creates
   no-space symlinks plus a source mapping manifest for RKNN's list parser.
2. Convert the frozen ONNX with `convert_mobilenetv3.py`. The converter fixes
   RGB, NCHW, ImageNet normalisation, RK3588 target, and the conversion
   manifest. Do not change these settings without re-running parity checks.
3. Copy the `.rknn`, its manifest, the matching label map, and one known image
   to the CA2. Never copy full training data or model checkpoints to a public
   repository.
4. Run `bootstrap_board.sh` once on the CA2. It installs Python packages under
   `/opt/birdpilot/python-packages` and leaves the system Python untouched.
   If package wheels have been transferred to the board, run it with
   `BIRDPILOT_WHEEL_DIR=/opt/birdpilot/wheels` to avoid a network download.
5. Start board scripts with `run_board_python.sh`, then use `infer_rknn.py` for warm-up, repeated NPU inference, and a JSON evidence
   record containing hashes, prediction, and median/P95 latency.

The CA2 has a Python 3.9/aarch64 environment. `bootstrap_board.sh` therefore
uses the official RKNN-Toolkit-Lite2 Python 3.9 wheel. Its package directory
exists because the current Debian image does not expose a matching
`python3-venv` package; it must be rechecked if the board image, Python version,
RKNN runtime, or NPU driver changes.

The remote-access milestone is development infrastructure, not RK3588 inference evidence. See [REMOTE_ACCESS_CN.md](REMOTE_ACCESS_CN.md) for the verified environment, the CA2 kernel/TUN limitation, the userspace-networking workaround, security gates, and the path from one prototype to fleet provisioning.

## YOLOX-Nano detector

`convert_yolox.py` freezes the RK3588 FP16 conversion contract for the official
416-pixel YOLOX-Nano ONNX model. `check_yolox_parity.py` rebuilds the same FP16
graph in Toolkit2 2.0.0's simulator and compares the gate, highest-score box,
IoU, stabilized crop, and final classifier top-1. Toolkit2 2.0.0 cannot start
its simulator from an already exported RKNN, so the exported artifact hash is
recorded while the identical ONNX/config graph is rebuilt in memory.

The ignored `outputs/birdpilot_exp016_board_bundle.tar.gz` contains both FP16
models, 24 licensed acceptance images, reference outputs, scripts, manifests,
and `SHA256SUMS`. After extraction on the board, run the command in its README.
