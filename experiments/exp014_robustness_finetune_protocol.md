# exp014: Paired robustness fine-tuning protocol

## Frozen comparison

- M0: accepted exp005 checkpoint with no updates.
- C: clean continuation from M0.
- A: continuation from the same M0 weights and ordinary transform stream, with one reviewed synthetic degradation applied to 50% of training inputs.
- Validation only during training: 2,620 clean images plus 39,300 reviewed degraded variants. Test rows and files are not read by the training path.
- Initial seed: 42. Planned schedule: ten epochs. First stop: epoch 3 for feasibility review.

The paired configs differ only in `run_name` and `degradation_probability`:

- `configs/exp014_clean_seed42.json`
- `configs/exp014_augmented_seed42.json`

## GPU launch

From the repository root, in the verified CUDA environment:

```bash
python src/train_robustness.py --config configs/exp014_clean_seed42.json --stop-after-epoch 3
python src/train_robustness.py --config configs/exp014_augmented_seed42.json --stop-after-epoch 3
```

Review finite losses, actual CUDA use, elapsed time, clean accuracy, all 15 condition scores, and storage. If the gate passes, resume the same runs rather than restarting:

```bash
python src/train_robustness.py --config configs/exp014_clean_seed42.json --resume outputs/robustness_runs/exp014_clean_seed42/last.pt
python src/train_robustness.py --config configs/exp014_augmented_seed42.json --resume outputs/robustness_runs/exp014_augmented_seed42/last.pt
```

After both arms reach epoch 10, apply the frozen comparison:

```bash
python src/select_robustness_candidate.py \
  --baseline-summary outputs/exp013_m0_robustness_full_v2/summary.json \
  --clean-metrics outputs/robustness_runs/exp014_clean_seed42/metrics.jsonl \
  --augmented-metrics outputs/robustness_runs/exp014_augmented_seed42/metrics.jsonl \
  --output outputs/robustness_runs/exp014_seed42_selection.json
```

The augmented checkpoint is promoted only if its clean loss is at most 0.5 percentage points against both M0 and C, and its 15-condition mean improves by at least 2 points against both. A failed gate is a valid negative result; M0 remains deployed.

If A passes, export and evaluate the selected C and A checkpoints before test access. Use distinct, previously absent output paths:

```bash
python src/export_onnx.py \
  --checkpoint outputs/robustness_runs/exp014_clean_seed42/best.pt \
  --output outputs/robustness_runs/exp014_clean_seed42/best.onnx
python src/export_onnx.py \
  --checkpoint outputs/robustness_runs/exp014_augmented_seed42/best.pt \
  --output outputs/robustness_runs/exp014_augmented_seed42/best.onnx
```

Run `src/evaluate_robustness.py` for each ONNX/checkpoint pair against the frozen full validation manifest. Then run `src/bootstrap_robustness_difference.py` twice: A minus M0 and A minus C. It pairs predictions by source and condition, and resamples whole source clusters so the 15 variants are not treated as independent images. Only after the validation decision is locked should the reused held-out test set be evaluated.

## Resume smoke evidence

A CPU-only 32-train / 16-valid smoke run stopped after epoch 1 and resumed through epoch 2. Metrics contain exactly epochs 1 and 2, optimizer/scheduler/RNG state restored, and the learning rate continued from `0.0001` to `0.0000505`. This verifies execution and resume plumbing, not model quality or runtime budget.

## Compute boundary

Formal runs use 84,480 training images, batch 256, full clean validation, and 39,300 degraded validation images per epoch. The local CPU smoke cannot provide a credible GPU time or cost estimate. Probe the actual GPU, CUDA/PyTorch compatibility, free memory, disk space, and one-epoch elapsed time before committing to the remaining epochs.
