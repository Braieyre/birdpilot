# Experiment 002 — ResNet18 pilot training on subset10

## Basic Info

- Project: BirdPilot
- Date: 2026-03-11
- Stage: Pilot training on MacBook Air
- Experiment ID: exp002_resnet18_subset10

## Objective

Validate the training pipeline on Apple Silicon MPS using a larger pilot subset, and observe whether the model can learn meaningful bird-class features under a 524-class classification setting.

## Dataset

- Dataset: Birds 525 Species subset
- Number of classes: 524
- Train size: 8238
- Valid size: 1048
- Test size: 1048
- Missing files filtered: 8

Note:
The original subset contained 525 classes, but one class was effectively removed after filtering missing image files.

## Model

- Backbone: ResNet18
- Pretrained: Yes
- Final layer: replaced to match 524 classes

## Training Config

- Input size: 160
- Batch size: 16
- Epochs: 5
- Optimizer: Adam
- Learning rate: 1e-3
- Loss: CrossEntropyLoss
- Device: Apple MPS
- Num workers: 0
- Seed: 42

## Results

| Epoch | Train Loss | Train Acc | Valid Loss | Valid Acc | Time (s) |
|------|-----------:|----------:|-----------:|----------:|---------:|
| 1 | 6.5703 | 0.2064% | 6.3323 | 0.6679% | 49.2 |
| 2 | 5.9799 | 0.6798% | 5.6999 | 1.1450% | 48.4 |
| 3 | 5.5144 | 1.8451% | 5.3437 | 2.7672% | 49.6 |
| 4 | 5.0899 | 3.9087% | 4.7291 | 5.8206% | 50.7 |
| 5 | 4.5867 | 7.5868% | 4.3476 | 10.2099% | 53.3 |

### Final Test Result

- Test loss: 4.1806
- Test accuracy: 11.4504%

## Observations

1. The training pipeline is fully functional on MacBook Air with Apple MPS.
2. Loss decreases consistently across epochs, indicating stable optimization.
3. Validation accuracy improves from 0.6679% to 10.2099%, showing that the model is learning meaningful visual patterns.
4. Test accuracy (11.4504%) is slightly higher than validation accuracy, suggesting no obvious overfitting at this stage.
5. Compared with the previous smaller subset experiment, the larger training subset provides significantly better learning signals.

## Conclusion

This pilot experiment successfully validates the end-to-end training workflow, including:

- CSV loading
- missing-file filtering
- dataset construction
- training on MPS
- checkpoint saving
- validation and test evaluation

The current subset is large enough to demonstrate real learning behavior, though it is still not the final full-scale training setting.

## Next Step

1. Save metrics automatically to CSV during training.
2. Visualize training curves.
3. Compare different backbones or training settings.
4. Move to full-dataset training on cloud GPU.