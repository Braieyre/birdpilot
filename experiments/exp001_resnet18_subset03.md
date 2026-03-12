# Experiment 001 — ResNet18 pilot training on subset03

## Basic Info

- Project: BirdPilot
- Date: 2026-03-11
- Stage: Initial pilot training on MacBook Air
- Experiment ID: exp001_resnet18_subset03

## Objective

Run the first real end-to-end pilot training on Apple Silicon MPS, verify whether the training pipeline works correctly, and observe whether the model begins to learn under a very small subset setting.

## Dataset

- Dataset: Birds 525 Species subset
- Number of classes after filtering: 524
- Train size: 2245
- Valid size: 1048
- Test size: 1048
- Missing files filtered: 8

Note:
The original subset included 525 classes, but one class was effectively removed after filtering missing image files. The removed files all belonged to the class `PARAKETT  AKULET`.

## Model

- Backbone: ResNet18
- Pretrained: Yes
- Final layer: replaced to match 524 classes

## Training Config

- Input size: 160
- Batch size: 16
- Epochs: 3
- Optimizer: Adam
- Learning rate: 1e-3
- Loss: CrossEntropyLoss
- Device: Apple MPS
- Num workers: 0
- Seed: 42

## Results

| Epoch | Train Loss | Train Acc | Valid Loss | Valid Acc | Time (s) |
|------|-----------:|----------:|-----------:|----------:|---------:|
| 1 | 6.9932 | 0.3563% | 6.6000 | 0.1908% | 16.2 |
| 2 | 6.4651 | 0.0445% | 6.5470 | 0.3817% | 14.1 |
| 3 | 6.3664 | 0.1336% | 6.1785 | 0.4771% | 14.0 |

### Final Test Result

- Test loss: 6.1618
- Test accuracy: 0.0954%

## Observations

1. The training pipeline successfully runs end-to-end on MacBook Air with Apple MPS.
2. Missing-file filtering works correctly and prevents DataLoader crashes.
3. Training loss decreases, which indicates that optimization is functioning.
4. However, the classification accuracy remains extremely low due to the tiny training subset.
5. With 2245 training images across 524 classes, the average number of images per class is too small to support meaningful fine-grained learning.

## Conclusion

This experiment is considered a successful pipeline validation rather than a meaningful model-performance evaluation.

It confirms that the following components work correctly:

- CSV loading
- image path resolution
- missing-file filtering
- dataset and dataloader construction
- MPS training
- checkpoint saving
- validation and test evaluation

However, the subset is too small to judge model quality.

## Next Step

Increase the training subset size and rerun the pilot experiment with the same pipeline to check whether the model begins to learn more meaningful bird-class features.