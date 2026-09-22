# Execution Status

## Active Package

`WP-04`: local detector package complete; waiting for real-board detector parity.

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

- exp017 adds a distinct close-feeder-view risk check. A deterministic 24-image sample from the CC BY-NC-ND 4.0 BirdLense feeder crops produced a YOLOX-Nano bird detection on only `6/24` images. Because the source images are mostly pre-cropped detections, this is directional risk evidence rather than a recall benchmark; it shows that exp016's distant/full-bird result cannot be transferred to the intended sub-20 cm composition.
- The close-view decision is now conditional: retain YOLOX for full-bird frames, but compare it with fixed-ROI classification on the first intended-camera frames. A temporary camera jig and feeding-plane marker are sufficient; the final enclosure is not required.
- Exp018 implements that comparison as one shared schema on local ONNX and board RKNN entry points. The fixed route records normalized ROI, background hash, mean absolute change, changed-pixel fraction, crop, classification and latency. Defaults are explicitly provisional until intended-camera calibration.
- Fresh local checks pass all 8 contract tests. The accepted exp014 ONNX (`bdc31c7...`) fixed-ROI smoke opens `1/2` gates (background closed, synthetic subject open), while the unified YOLOX entry reproduces `6/24` on the fixed exp017 sample.

- YOLOX-Nano is the frozen Apache-2.0 detector. It passes all predeclared gates: `44/54 = 81.5%` bird-scene detection, `33/44 = 75.0%` crop classification, `33/54 = 61.1%` end to end, and `0/20` empty triggers.
- Toolkit2 `2.0.0b0+9bab5682` produced FP16 RKNN SHA-256 `f137e5f85ba5c4fbdc2249c116edc5dab6ff15679d0340993e4223643f78c69e`.
- Local simulator parity passes 54/54 iNaturalist and 40/40 Wellington scenes for gate and highest-score-box choice. Minimum box IoU is `0.966846`; stabilized crop classifier top-1 agrees for all 94 scenes.
- The 20-sample failure review attributes 5 to detection, 14 to classifier external-domain shift, and 1 to label/multi-bird ambiguity. No weights were updated.
- The ignored 24-image board archive is ready at `outputs/birdpilot_exp016_board_bundle.tar.gz`, SHA-256 `c718a93b5fe8e730c357a3fd0adfc8b3ac9cc0861aec5ea4bfe9a9a90b2a25cc`.

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

Attach and enumerate the intended camera, hold it approximately 20 cm from a temporary feeding plane, then capture an unchanged empty reference plus full-subject, partial-subject, occluded and lighting-change frames. Run both routes on the same frames and freeze the route, ROI and thresholds from the recorded comparison.

## Physical-camera boundary

The board is online and NPU inference is accepted. RKISP device nodes exist, but one-frame capture fails and the kernel reports `get remote terminal sensor failed -19`; no usable camera sensor is currently attached to the media pipeline.
