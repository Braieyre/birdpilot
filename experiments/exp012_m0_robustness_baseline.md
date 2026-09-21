# exp012/013: M0 controlled-degradation baseline

## Purpose

Measure the accepted exp005 MobileNetV3 model before any robustness training. These results describe a synthetic validation benchmark; they are not outdoor or camera evidence.

## Frozen inputs

- Checkpoint: `models/exp005_mobilenetv3_full_best.pt`, SHA-256 `b80b9142c1a58cfd0b76d5b2be40a2c03c09a60e10d208bafb30f74ec19615d5`.
- ONNX graph: `outputs/mobilenetv3_birds.onnx`, SHA-256 `4070a9307bd191ef167fe6f8d9a5dcf8ba2e2155dc4d3a59da94380841c21f87`.
- ONNX external data: SHA-256 `aea8b047e54424d07ddd2058a33d7bead7da37eb867e9f639086999af6428ae8`.
- Label map: 524 classes; it matches the checkpoint mapping.
- Full validation degradation manifest: 2,620 readable sources and 39,300 variants, SHA-256 `bbb534b6eb97219f76d09dcfee309c19cbb1beebcc8c9d1ada7a84831c016f84`.
- Five CSV rows for `PARAKETT  AKULET` are absent on disk and absent from the 524-class model. The same issue affects train, valid, and test; the readable split contains 524 classes.
- Cross-split file-content hash collisions: zero.

The 524-source screening set and the full validation set use one and five readable images per class respectively. The 15 degraded variants of each source are correlated observations.

## Benchmark review

All 20 pilot sources and five review pages were visually inspected. Darkening, blur, resolution loss, and JPEG severity increase monotonically. Occlusion uses a nested opaque ellipse at approximately 8.12%, 16.42%, and 28.39% actual image area. The ellipse is a controlled synthetic obstruction, not a claim about foliage. Downscale is a resolution-loss proxy, not proof about physical distance. All reviewed variants retained a visible bird subject.

Pilot manifest SHA-256: `e109f7a9f24e18557b4a91af705fac0e949ca15d2d877fa4dfd7acd5eb512225`.

## M0 results on complete readable validation split

| Condition | Top-1 | Correct / total |
|---|---:|---:|
| clean | 98.59% | 2,583 / 2,620 |
| downscale / light | 98.09% | 2,570 / 2,620 |
| downscale / medium | 96.68% | 2,533 / 2,620 |
| downscale / heavy | 94.50% | 2,476 / 2,620 |
| JPEG / light | 97.98% | 2,567 / 2,620 |
| JPEG / medium | 97.52% | 2,555 / 2,620 |
| JPEG / heavy | 96.87% | 2,538 / 2,620 |
| low light / light | 98.47% | 2,580 / 2,620 |
| low light / medium | 98.21% | 2,573 / 2,620 |
| low light / heavy | 97.98% | 2,567 / 2,620 |
| motion blur / light | 97.37% | 2,551 / 2,620 |
| motion blur / medium | 82.48% | 2,161 / 2,620 |
| motion blur / heavy | 44.62% | 1,169 / 2,620 |
| occlusion / light | 96.76% | 2,535 / 2,620 |
| occlusion / medium | 94.66% | 2,480 / 2,620 |
| occlusion / heavy | 85.69% | 2,245 / 2,620 |

Overall and macro top-1 are equal because every represented class has five readable sources. The equal-weight mean across the 15 degraded conditions is 91.8575%.

The main measured gap is motion blur: medium loses 16.11 percentage points and heavy loses 53.97 points from clean. Heavy occlusion loses 12.90 points. The other families remain between 94.50% and 98.47%. This supports a bounded paired fine-tuning experiment focused on robustness; it does not support a claim that the deployed model generally fails.

Full summary SHA-256: `8cac0ea19198f932666d57a577f475e730479a0b0d0fdb4dbf731c5964e6f53f`. Per-image predictions SHA-256: `e874a04e9f7396802aaa54441ee01a63283e48a5427222052a90ed4c8ca79d93`.

## Runtime parity checks

- PyTorch checkpoint versus ONNXRuntime: 16 validation images, zero top-1 mismatches; maximum absolute logit difference `1.33514404296875e-05`.
- FP16 RKNN on the real RK3588S board versus ONNXRuntime: 20 validation images from 20 evenly spaced classes, zero top-1 mismatches. One image is misclassified by both backends, so this check proves conversion parity rather than accuracy. Median of the 20 per-image median latencies is 21.5827 ms over three measured calls per image after warm-up.
- Board report SHA-256: `fa0c651cbc72ceb7001a75f084b6239b908e2849923159703811d2a725199964`; batch manifest SHA-256: `b0784a8163091def4952298e8c82333b78f3a000d38acca93659f2e78464c067`.

## Decision

Proceed to the predeclared paired clean-versus-augmented continuation experiment. Run three epochs per arm as the first gate, starting both arms from the exact M0 checkpoint. Do not access test images. Continue to ten epochs only if both runs are numerically sound and the augmented arm shows a plausible robustness benefit without material clean loss.
