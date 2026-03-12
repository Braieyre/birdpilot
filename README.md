# BirdPilot

BirdPilot is a prototype project for an automatic bird recognition system running on edge devices.

The goal of this project is to build a complete pipeline:

camera → bird detection → bird crop → species classification → edge deployment

---

# Project Goal

The project aims to develop a lightweight bird recognition system that can run on embedded hardware such as RK3588.

The system is designed to:

• detect bird visits automatically  
• classify bird species  
• store images and labels locally  
• support lightweight edge deployment  

---

# Dataset

Dataset used in this project:

Birds 525 Species Dataset

Dataset statistics:

Train images: ~84,635  
Validation images: ~2,625  
Test images: ~2,625  
Number of classes: 525  

Images are RGB JPG with resolution:

224 × 224

Each image typically contains a single bird occupying most of the frame.

---

# Model

Baseline model used in the pilot experiments:

ResNet18 (pretrained)

Final fully connected layer replaced to match the number of bird species.

Loss function:

CrossEntropyLoss

Optimizer:

Adam

---

# Training Pipeline

The training pipeline includes:

1. CSV dataset parsing  
2. missing image filtering  
3. PyTorch Dataset construction  
4. DataLoader batching  
5. ResNet18 training  
6. validation evaluation  
7. model checkpoint saving  

Training is first tested on a small subset to validate the pipeline.

---

# Pilot Experiments

## Experiment 001

Small pilot training.

Dataset:

Train: 2245  
Valid: 1048  
Test: 1048  
Classes: 524  

Training:

Epochs: 3  
Batch size: 16  
Device: Apple MPS  

Results:

Best validation accuracy:

0.4771%

Test accuracy:

0.0954%

Observation:

The pipeline runs successfully but dataset size is too small for meaningful learning.

---

## Experiment 002

Subset10 training.

Dataset:

Train: 8238  
Valid: 1048  
Test: 1048  
Classes: 524  

Training:

Epochs: 5  
Batch size: 16  
Device: Apple MPS  

Results:

Best validation accuracy:

10.21%

Test accuracy:

11.45%

Observation:

Model begins to learn meaningful bird visual features.

Loss decreases consistently and validation accuracy improves across epochs.

---

# Training Curves

Example training curves generated from experiment logs.

Loss curve:

figures/exp002_loss_curve.png

Accuracy curve:

figures/exp002_accuracy_curve.png

---

# Project Structure
```text
birdpilot
├── configs
├── data
├── experiments
├── figures
├── logs
├── notebooks
├── outputs
├── src
└── README.md
```
---

# Next Steps

Next development stages:

• train on full dataset using cloud GPU  
• test different backbone models  
• export model to ONNX  
• deploy inference on RK3588  
• integrate camera pipeline  

---

# Long-term Vision

The final system will be able to run autonomously:

camera monitoring → bird detection → species recognition → local logging

forming a lightweight intelligent bird observation system.

---

# Author

BirdPilot research prototype  
Computer Vision / Edge AI exploration project