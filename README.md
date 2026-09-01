# BirdPilot

> 团队新成员请先阅读 [TEAM_ONBOARDING.md](TEAM_ONBOARDING.md)，使用小型入门数据包跑通流程后，再申请完整数据和模型文件。

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

# Project Objectives

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

Next stage:

```
Edge-device deployment (RK3588)
```

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
Automatic bird observation system
```

Upcoming tasks:

1. Install RKNN Toolkit
2. Convert ONNX → RKNN
3. Measure RK3588 inference latency
4. Integrate camera input
5. Build autonomous bird monitoring device

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
