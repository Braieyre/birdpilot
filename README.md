# BirdPilot

[English](README.md) | [中文](README_CN.md)

BirdPilot is an experimental **Edge AI system for automatic bird species recognition**. The project explores the full engineering pipeline from dataset training to deployable edge inference.

The goal is not only to train a classifier, but to build a **complete autonomous bird observation system** capable of running on embedded hardware.

Planned system pipeline:

```
Camera
  ↓
Bird Detection
  ↓
Bird Crop
  ↓
Species Classification
  ↓
Local Logging
```

Target deployment platform: **RK3588 edge device**.

---

## Project Objectives

BirdPilot explores three technical layers:

1. Bird species classification model training
2. Model portability and inference validation
3. Edge-device deployment and automated observation

The project focuses on **engineering reproducibility and deployment readiness**, rather than proposing new neural network architectures.

---

# Dataset

Dataset used:

**Birds-525 Species Dataset**

Statistics:

| Split | Images |
|------|------|
| Train | ~84k |
| Validation | ~2.6k |
| Test | ~2.6k |

Number of classes:

```
524 bird species
```

Image resolution:

```
224 × 224 RGB
```

Dataset structure:

```
data/birds
├── train
├── valid
├── test
└── birds.csv
```

Images typically contain a single bird occupying most of the frame.

---

# Model Architectures

Two CNN architectures were evaluated.

## ResNet18

Standard convolutional neural network pretrained on ImageNet.

Advantages:

- stable training
- strong baseline performance


## MobileNetV3

Lightweight architecture designed for mobile and embedded inference.

Advantages:

- significantly smaller parameter count
- faster inference
- better suited for edge devices

---

# Experiment Timeline

## exp001 — Pilot Subset (Mac)

Initial training pipeline validation on a very small dataset subset.

Purpose:

- verify dataset loading
- validate training scripts

---

## exp002 — Subset Training

Training on a larger subset to observe learning behavior.

Observation:

- training loss decreases steadily
- validation accuracy improves

---

## exp003 — Cloud Smoke Test

Short training run on cloud GPU.

Purpose:

- verify CUDA environment
- confirm dataset loading on remote machine
- validate training script compatibility

---

## exp004 — Full Dataset Training (ResNet18)

First full-scale training experiment.

Hardware:

```
NVIDIA RTX 5090
```

Results:

| Metric | Value |
|------|------|
| Best validation accuracy | 97.94% |
| Test accuracy | 99.20% |

Observation:

The dataset is extremely clean and classification accuracy becomes very high.

---

## exp005 — Lightweight Model Evaluation (MobileNetV3)

Goal:

Evaluate a model architecture more suitable for edge deployment.

Hardware:

```
NVIDIA RTX 5090
```

Results:

| Metric | Value |
|------|------|
| Best validation accuracy | 98.63% |
| Test accuracy | 99.24% |

Conclusion:

MobileNetV3 achieves slightly higher validation accuracy while being significantly lighter than ResNet18.

Therefore MobileNetV3 becomes the **primary deployment candidate**.

---

## exp006 — ONNX Export & Runtime Validation

Goal:

Validate model portability outside the PyTorch training environment.

Steps:

- Export PyTorch checkpoint → ONNX
- Run inference with ONNX Runtime
- Compare predictions between PyTorch and ONNX

Result:

```
PyTorch prediction == ONNX prediction
```

This confirms the correctness of the ONNX export pipeline.

---

## exp007 — Real Image ONNX Benchmark

Goal:

Measure **end-to-end inference latency** using a real image.

Test image:

```
data/birds/test/ABYSSINIAN GROUND HORNBILL/1.jpg
```

Prediction:

```
Predicted index : 2
Predicted label : ABYSSINIAN GROUND HORNBILL
```

Measured pipeline:

```
image load → preprocess → ONNX inference → postprocess
```

Benchmark result:

| Stage | Average Latency |
|------|------|
| Image load | 0.500 ms |
| Preprocess | 0.495 ms |
| Model inference | 5.680 ms |
| Postprocess | 0.015 ms |
| **Total latency** | **6.690 ms** |

Approx throughput:

```
~149 FPS (single-image inference)
```

Observation:

- Most latency comes from model inference (~5.7 ms).
- Data loading and preprocessing overhead are minimal.

Conclusion:

The ONNX model is efficient and suitable for edge-side inference.

---

# Model Comparison

| Model | Params | Valid Acc | Test Acc | Deployment Priority |
|------|------|------|------|------|
| ResNet18 | ~11M | 97.94% | 99.20% | Secondary |
| MobileNetV3 | ~5.4M | 98.63% | 99.24% | Primary |

MobileNetV3 provides the best balance between **accuracy and computational cost**.

---

# Project Structure

```
birdpilot
├── configs
├── data
├── experiments
├── logs
├── outputs
├── src
│   ├── train_full.py
│   ├── export_onnx.py
│   ├── test_onnx.py
│   ├── test_torch.py
│   └── benchmark_onnx.py
├── deploy
│   ├── onnx
│   └── rk3588
└── README.md
```

---

# Current Project Status

Training pipeline        ✔

Model selection          ✔

ONNX export              ✔

Inference validation     ✔

Real-image benchmark     ✔

QuarkPi-CA2 development access (USB OTG/ADB rescue and cross-network Tailscale SSH)     ✔

Three-seed paired robustness experiment and locked reused-test evaluation              ✔

Accepted FP16 RKNN on the real RK3588S, 20/20 ONNX top-1 parity                       ✔

The accepted augmented model reaches 99.20% clean and 98.50% mean synthetic-degradation top-1 on the reused held-out test partition. Its installed FP16 RKNN matches ONNX on 20/20 validation images. A 30-run board record has 22.69 ms median and 29.98 ms P95 NPU latency. See the [exp014 results](experiments/exp014_robustness_finetune_results.md) for hashes, uncertainty, and evidence limits.

Next stage:

```
Real-board detector parity, then physical camera integration
```

The detector is now frozen as Apache-2.0 YOLOX-Nano. It detects 44/54 external bird scenes; the highest-score crop classifies 33/44 correctly and reaches 33/54 end to end, with zero triggers on 20 empty-labelled frames. The Toolkit2 2.0.0 FP16 RKNN passes local simulator parity on all 94 proxy images for gate, best-box selection, and crop classification; the minimum matched-box IoU is 0.9668. Real-board NPU output and latency remain the next gate. This is external proxy evidence, not camera or outdoor evidence. See [exp016](experiments/exp016_fixed_view_proxy_plan.md).

A separate close-feeder check found that YOLOX-Nano detects only 6/24 pre-cropped near-view images, so the distant/full-bird result is not transferred to the intended sub-20 cm composition. [Exp018](experiments/exp018_close_view_dual_route.md) provides one result schema for YOLOX and fixed-ROI/background-change routes on ONNX and RKNN. The route and thresholds remain pending intended-camera frames; the final enclosure is not required for that comparison.

---

# Deployment Roadmap

Planned pipeline:

```
PyTorch (.pt)
   ↓
ONNX
   ↓
RKNN conversion
   ↓
RK3588 inference
   ↓
Camera integration
   ↓
Bird detection and empty-frame gating
   ↓
Bird crop and species classification
   ↓
Automatic bird observation system
```

Upcoming tasks:

1. Upload the fixed 24-image detector bundle and verify real RK3588 NPU parity and latency
2. Connect and enumerate the intended camera sensor
3. Capture and process one timestamped real frame end to end
4. Add a short observation loop
5. Run bounded outdoor validation

---

# Long-Term Vision

BirdPilot aims to build a lightweight autonomous bird observation system:

```
camera monitoring
   ↓
bird detection
   ↓
species classification
   ↓
automatic logging
```

The project explores the intersection of:

- Computer Vision
- Edge AI
- Embedded Systems

---

# Author

BirdPilot

Edge AI / Computer Vision Engineering Exploration
