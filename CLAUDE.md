# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BirdPilot is an edge AI project for bird species classification (524 species, Birds-525 dataset). The pipeline trains CNN classifiers in PyTorch, exports to ONNX, and targets deployment on RK3588 NPU hardware. The primary deployment model is MobileNetV3-Large (~5.4M params, 99.24% test accuracy). ResNet18 (~11M params) is the secondary/experimental model.

## Commands

### Training (full dataset, requires CUDA GPU)
```bash
python src/train_full.py --experiment-name my_exp --model-name mobilenetv3 --epochs 30 --batch-size 256
python src/train_full.py --model-name resnet18  # resnet18 or mobilenetv3
```

### Smoke test (quick validation, few steps)
```bash
python src/train_smoke.py --max-train-steps 20 --max-valid-steps 5
```

### Pilot training (small subset, runs on Mac MPS or CPU)
```bash
python src/train_pilot.py
```

### Create small dataset subset
```bash
python src/make_small_subset.py  # stratified 1% sample into data/birds_1pct/
```

### Export to ONNX
```bash
python src/export_onnx.py  # outputs/mobilenetv3_birds.onnx
```

### Test inference
```bash
python src/test_torch.py  # PyTorch inference on a single image
python src/test_onnx.py   # ONNX Runtime inference on a single image
```

### Benchmark
```bash
python src/benchmark_onxx.py              # synthetic random-input ONNX latency
python src/benchmark_real_image_onnx.py   # real image end-to-end pipeline benchmark
```

## Architecture

### Training pipeline (`src/train_full.py`)
The main training script uses a `TrainConfig` dataclass for all hyperparameters. Key design choices:
- **Dataset**: `BirdDataset` loads from CSV (`data/birds/birds.csv`) with columns `labels`, `filepaths`, `data set`. Splits are `train`/`valid`/`test`. Missing image files are auto-filtered.
- **Label mapping**: Classes are sorted alphabetically, mapped to indices. Saved as `{exp}_idx_to_label.json` in `outputs/`.
- **Optimization**: AdamW + CosineAnnealingLR + label smoothing (0.1). AMP mixed precision on CUDA with `GradScaler`.
- **Checkpoint format**: Dict with keys `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_valid_acc`, `num_classes`, `config`, `class_to_idx`.
- **Outputs per experiment**: `{exp}_best.pt`, `{exp}_last.pt`, `{exp}_metrics.csv` (in `logs/`), `{exp}_idx_to_label.json`, `{exp}_summary.json`, `{exp}.json` config (in `configs/`).

### Model selection (`build_model()`)
Supports `resnet18` and `mobilenetv3` via `cfg.model_name`. Both use torchvision pretrained weights. The final classification layer is replaced with `nn.Linear(in_features, num_classes)`.

### Data augmentation
- **Train**: RandomResizedCrop (scale 0.8–1.0), RandomHorizontalFlip, ColorJitter
- **Eval**: Resize to 224×224 only
- **Normalization**: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

### ONNX export (`src/export_onnx.py`)
Hardcoded to export the MobileNetV3 checkpoint (`models/exp005_mobilenetv3_full_best.pt`) to `outputs/mobilenetv3_birds.onnx` with opset 12 and dynamic batch axis. Input name: `"input"`, output name: `"output"`.

### Scripts relationship
- `train_pilot.py` — standalone early prototype, uses `data/birds_1pct/`, hardcoded params, no config dataclass
- `train_smoke.py` — structured like `train_full.py` but limits steps per epoch (`max_train_steps`, etc.) for quick CI-style validation
- `train_full.py` — production training script, CLI args → `TrainConfig`

## Key paths
```
data/birds/          — full dataset (train/valid/test dirs + birds.csv)
data/birds_1pct/     — 1% subset for local dev
models/              — saved .pt checkpoints
outputs/             — ONNX models, label maps, summaries
configs/             — experiment config JSON/YAML files
logs/                — per-epoch metrics CSVs
experiments/         — experiment notes/markdown
deploy/rk3588/       — RK3588 deployment (RKNN conversion, in progress)
```

## Environment notes
- Python with PyTorch, torchvision, pandas, Pillow, onnxruntime (for inference/benchmark scripts)
- CUDA training targets RTX 5090 with `torch.channels_last` memory format and persistent workers
- MPS (Apple Silicon) supported for pilot/local dev but not the primary target
- `requirements.txt` also includes `onnx` and `onnxruntime` for export and runtime validation (note: `timm` and `scikit-learn` are listed but not currently imported by any script)
