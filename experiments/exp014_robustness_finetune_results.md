# exp014: Paired robustness fine-tuning results

## Decision

Accept the degradation-augmented seed-42 epoch-9 checkpoint for FP16 RKNN conversion. The experiment passed the predeclared three-seed promotion rule, the selected checkpoint reproduced in local ONNX Runtime, and its robustness gain persisted on the reused held-out test partition.

This result covers the frozen synthetic benchmark only. It is not camera, outdoor, or wildlife-observation evidence.

## L20 training

All runs started from M0, used ten epochs, batch 256, AdamW at `1e-4`, weight decay `1e-4`, label smoothing `0.1`, and the same full clean and degraded validation sets. C retained clean training; A applied one uniformly sampled reviewed degradation to half of training inputs.

| Seed | Selected C epoch | C clean | C robust | Selected A epoch | A clean | A robust | A-C robust |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3 | 98.5878% | 92.1094% | 9 | 98.4351% | 97.5140% | +5.4046 pp |
| 43 | 8 | 98.5496% | 92.0916% | 9 | 98.5115% | 97.5751% | +5.4835 pp |
| 44 | 1 | 98.5878% | 91.8787% | 9 | 98.4733% | 97.5725% | +5.6938 pp |

Across seeds, A improves over M0 by `5.7040` points (sample SD `0.0347`) and over C by `5.5123` points (sample SD `0.1337`). Mean clean loss against M0 is `0.1018` points. All three A-C differences are positive, so the repeated gate passed. The frozen deployment rule chooses seed 42.

## Local ONNX validation

| Model | Clean | 15-condition mean | Change vs M0 |
|---|---:|---:|---:|
| M0 | 98.5878% | 91.8575% | — |
| C42 epoch 3 | 98.5878% | 92.1043% | +0.2468 pp |
| A42 epoch 9 | 98.4351% | 97.5216% | +5.6641 pp |

The A-M0 source-clustered bootstrap interval is `[+5.3817, +5.9542]` points; A-C is `[+5.1349, +5.7023]` points (`10,000` samples, seed `20260921`). PyTorch and ONNX predictions match on all 16 parity images; A's maximum absolute logit delta is `7.62939453125e-06`.

## Reused held-out test

The validation choice and model hashes were locked before test access. The original test partition has historical use, so this is a reused held-out test rather than a never-seen blind set. It contains 2,620 readable clean images and 39,300 deterministic degraded variants.

| Model | Clean | 15-condition mean | Change vs M0 | Lowest condition |
|---|---:|---:|---:|---|
| M0 | 99.2366% | 93.3486% | — | motion blur heavy: 48.3588% |
| C42 epoch 3 | 99.3511% | 93.5420% | +0.1934 pp | motion blur heavy: 52.2901% |
| A42 epoch 9 | 99.1985% | 98.5038% | +5.1552 pp | occlusion heavy: 94.9618% |

The A-M0 source-clustered bootstrap interval is `[+4.9033, +5.4148]` points; A-C is `[+4.7023, +5.2214]` points. No tuning follows this test result.

## Frozen artifacts

- Repeated selection JSON SHA-256: `f3b02d04b6ea8d61b712c9ac3f38acc6809c3eb521b987d444fe9d3363d4c755`
- C42 checkpoint SHA-256: `02ba670a264d3403cf0511abf0afba63ac948c6d5af74d36332331e514772d19`
- A42 checkpoint SHA-256: `54dd1ea9ff4ff44dd3cdae47b311909d20638a1517a805fb1080b52b6a4f43d8`
- C42 ONNX SHA-256: `c0ca20fd81b96a62480cd0f33e37b8c2d525ea905299b34aee068f2de437a9f4`
- A42 ONNX SHA-256: `bdc31c7e6bccff37543b1507d576177c8ca540c99fb7d313c78748cbee5b01ab`
- Test degradation manifest SHA-256: `3daf8ca21b1e6fc8262a8dae28db6c0be705f537884d2d39667790a0a3712167`
- M0/C/A test summary SHA-256: `3b19b047f7ac18cd63413cefaf88f3e2b68e8048f4c5220add5a46718333aff9`, `df4a9c89b0c7124391cd4914cb5ba150f54b7977b8b2b54d173f34540eed18f2`, `a031083c57902aa0010e0e22b364c6f138a10dbaf16b8700bd60c84397d4adc3`

The server return archive SHA-256 is `9a5183d3cbbba5463dc5a644c2af9fbe279ee70281f829f5e9ffce32a6d83621`. Its source version is `8bff5977111ec8951992f508bb7d8e2d90a147e1`.

## Next acceptance gate

The accepted A ONNX was converted to FP16 RKNN with Toolkit2 `2.0.0b0+9bab5682`. RKNN SHA-256 is `1870a4185b1c080e91617f279b8d0bbe75cb8482142dd4b073a8909d8844af79`; the prepared 20-image transfer archive SHA-256 is `79104c51ce27f8ea3e596aab4a5d8fbd5c55265981e8dd847698ff4e959c6840`.

Run that fixed multi-image parity batch on the real RK3588S, record top-1 agreement, latency, and failures, and replace the board baseline only if parity passes.

## RK3588S deployment result

The fixed batch passed on the real QuarkPi-CA2: `20/20` RKNN top-1 predictions matched ONNX, with no mismatches. Across ten measured calls per image, the median of image medians was `29.6723 ms`. The board report SHA-256 is `737d1587c1034a46f6f75caae0021f6088db876f5a4a3498929d8a4e95e6df21`.

The accepted model is installed under a versioned path and selected through `/opt/birdpilot/models/current_fp16.rknn`; the prior M0 file remains for rollback. A separate 30-run local-image record correctly predicts `HAMERKOP`, with median latency `22.6942 ms` and P95 `29.9831 ms`. Its evidence SHA-256 is `4b42a8c31ebe7f5ec99403e0d84d4d4e5a24a7e705b42e3c819040ff14a5920b`.

Camera capture is not yet evidence: RKISP nodes enumerate, but streaming returns no frame and the kernel reports that no remote sensor is connected. WP-04 therefore proceeds once the intended camera hardware is attached and recognised.
