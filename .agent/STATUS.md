# Execution Status

## Active Package

`WP-04`: classifier installed; detector/crop stage required before the camera observation loop.

## State

READY

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

## External-scene proxy evidence

- The licensed iNaturalist candidate pool contains 54 images across 18 model classes. Full-frame classification is `13/54 = 24.1%` top-1.
- A provisional YOLOv8n COCO bird detector finds birds in `46/54 = 85.2%` of those scenes. Selecting the highest-confidence crop per scene gives `35/46 = 76.1%` classification top-1; counting detector misses as failures gives `35/54 = 64.8%` end-to-end success.
- Detector crops recover 23 full-frame classification failures and break none of the 13 full-frame successes in this pool. This is directional external-proxy evidence, not a frozen benchmark or camera result.
- On 20 Wellington frames carrying an empty sequence label, full-frame classifier confidence averages `16.5%` and reaches `82.6%`; classifier confidence therefore cannot gate empty scenes. The generic detector produces one bird-like box (`1/20`), which remains a visually ambiguous label conflict because annotations are sequence-level.
- The Wellington bird-labelled sample cannot yield a valid recall number: only 6 of 20 selected frames produce a bird box, but the sequence label may not describe that individual frame.
- Decision: use `detector -> crop -> species classifier`. No classifier retraining or GPU is indicated by this gate.

## Working State

- L20 return artifacts are present under ignored `outputs/`; the original downloaded archive remains outside the repository in `L20返回/`.
- Source changes add licensed Wellington candidate collection, generic COCO bird detection/cropping, and external full-scene/crop evaluation. Network images, model binaries and generated outputs remain ignored.
- Existing untracked `docs/` is unrelated and remains untouched.

## Next Action

Freeze the detector preprocessing, decode, NMS and crop contract; select a redistribution-safe detector candidate; convert it to RKNN; and verify ONNX/RKNN box, crop and final-classification parity on the proxy batch. Then connect and enumerate the intended camera sensor, capture one real frame, and persist the timestamped end-to-end record.

## Physical-camera boundary

The board is online and NPU inference is accepted. RKISP device nodes exist, but one-frame capture fails and the kernel reports `get remote terminal sensor failed -19`; no usable camera sensor is currently attached to the media pipeline.
