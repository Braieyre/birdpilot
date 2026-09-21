# Execution Status

## Active Package

`WP-04`: accepted model installed; camera observation loop pending.

## State

BLOCKED_BY_CAMERA_SENSOR

## Robustness decision

- The bounded three-seed L20 experiment completed all six ten-epoch runs. Every augmented-minus-clean seed difference was positive.
- Repeated augmented improvement over M0 is `+5.7040` percentage points (sample SD `0.0347`); improvement over paired clean continuation is `+5.5123` points (sample SD `0.1337`). Mean clean loss against M0 is `0.1018` points. The predeclared repeated gate passed.
- The frozen deployment rule selects seed 42. Clean checkpoint C is epoch 3, SHA-256 `02ba670a264d3403cf0511abf0afba63ac948c6d5af74d36332331e514772d19`. Augmented checkpoint A is epoch 9, SHA-256 `54dd1ea9ff4ff44dd3cdae47b311909d20638a1517a805fb1080b52b6a4f43d8`.
- A was accepted for deployment. Its ONNX SHA-256 is `bdc31c7e6bccff37543b1507d576177c8ca540c99fb7d313c78748cbee5b01ab`.

## Independent local verification

- Local ONNX evaluation reproduced the validation decision. A scores `98.4351%` clean and `97.5216%` over the 15 degraded conditions. Versus M0 the degradation gain is `+5.6641` points with source-clustered 95% bootstrap interval `[+5.3817, +5.9542]`; versus C it is `+5.4173` points with interval `[+5.1349, +5.7023]`.
- After that decision was locked, the reused held-out test partition was opened once. A scores `99.1985%` clean and `98.5038%` degraded mean. Versus M0 the degradation gain is `+5.1552` points with 95% interval `[+4.9033, +5.4148]`; versus C it is `+4.9618` points with interval `[+4.7023, +5.2214]`.
- A's lowest test condition is heavy occlusion at `94.9618%`. M0's lowest is heavy motion blur at `48.3588%`; A reaches `96.6794%` there.
- PyTorch and ONNX top-1 predictions agree on all checked samples for M0, C, and A. A's maximum checked absolute logit delta is `7.62939453125e-06`.
- These are deterministic synthetic-degradation results. They do not establish camera or outdoor performance.

## Runtime evidence

- Existing real RK3588S FP16 RKNN baseline and M0 ONNX agree on 20/20 validation images. Median per-image median latency is `21.5827 ms` after warm-up.
- INT8 remains rejected independently of robustness training.
- The accepted A ONNX has been converted to FP16 RKNN with Toolkit2 `2.0.0b0+9bab5682`; RKNN SHA-256 is `1870a4185b1c080e91617f279b8d0bbe75cb8482142dd4b073a8909d8844af79`.
- A 20-image validation parity bundle is ready. Its transfer archive SHA-256 is `79104c51ce27f8ea3e596aab4a5d8fbd5c55265981e8dd847698ff4e959c6840`.
- The new RKNN passed real-board parity on 20/20 validation images with zero mismatches. Across ten calls per image, the median of image medians is `29.6723 ms`. Board report SHA-256 is `737d1587c1034a46f6f75caae0021f6088db876f5a4a3498929d8a4e95e6df21`.
- The accepted model is installed at `/opt/birdpilot/models/exp014_augmented_seed42_epoch9_fp_rk3588_runtime200.rknn`; `/opt/birdpilot/models/current_fp16.rknn` points to it. The old M0 model remains available for rollback.
- A recorded 30-run single-image inference predicted `HAMERKOP` correctly. Median latency is `22.6942 ms` and P95 is `29.9831 ms`; evidence SHA-256 is `4b42a8c31ebe7f5ec99403e0d84d4d4e5a24a7e705b42e3c819040ff14a5920b`.

## Working State

- L20 return artifacts are present under ignored `outputs/`; the original downloaded archive remains outside the repository in `L20返回/`.
- Source changes add explicit final-test gating, partition-aware evaluation, and ONNX external-data hashes in export manifests.
- Existing untracked `docs/` is unrelated and remains untouched. No project commit has been pushed.

## Next Action

While the camera is unavailable, build the bounded exp016 licensed fixed-view proxy set and compare full-scene versus manual bird-crop inference without changing model weights. Then connect and enumerate the intended camera sensor, capture one real frame, and persist the image, timestamp, prediction, and latency record.

## Blocker

The board is online and NPU inference is accepted. RKISP device nodes exist, but one-frame capture fails and the kernel reports `get remote terminal sensor failed -19`; no usable camera sensor is currently attached to the media pipeline.
