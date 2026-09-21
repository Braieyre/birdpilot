# Project Plan

## Product Goal

Build BirdPilot into a reproducible bird-recognition prototype that first proves robustness against controlled image degradation online, then runs the same model and observation flow on the RK3588S device, and finally undergoes real outdoor observation.

## Constraints

- The team currently collaborates online; the RK3588S device is held by the project lead.
- Full training data and model weights must not be committed to the public repository.
- Simulated-degradation results must be labelled as simulation, never as outdoor evidence.
- Training, validation, and test data must remain separated; the test set cannot guide tuning.
- Gitee remains private; GitHub remains public.

## Current Reality

- Local baseline commit is `14ea645`; it was not pushed. Remote synchronization must be checked before any future push.
- Public GitHub contains the same approved source snapshot without private data or model binaries.
- MobileNetV3 training, ONNX export, and a local Mac benchmark have historical artifacts.
- A real QuarkPi-CA2 has been identified as ARM64 Debian 11 and its development path is verified: USB OTG exposes an ADB rescue shell, phone USB tethering provides temporary networking, and cross-network Tailscale SSH works in userspace-networking mode because the current 5.10.209 kernel lacks usable TUN support.
- A version-matched FP16 RKNN baseline now runs correctly on the real board; INT8 calibration still needs separate remediation before it can replace that baseline.
- Full data, the primary `.pt` checkpoint, and ONNX outputs are intentionally excluded from Git.

## Roadmap

1. Give the two collaborating members a synchronized repository, a reproducible starter-data package, and a clear starting point.
2. Establish a board baseline using the already accepted MobileNetV3 model: local image, RKNN inference, and evidence record.
3. Establish a fixed degradation benchmark and quality-review process before changing model weights.
4. Train and compare the clean baseline and degradation-augmented model without touching the held-out test set.
5. Replace the board-baseline model only after the accepted comparison, then complete a desk-based observation loop.
6. Perform final outdoor observation and report it separately from simulated evidence.

## Work Packages

### WP-01: Shared repository and two-member starter package

- Status: DONE
- Outcome: Both repositories expose the same safe code snapshot, and the two collaborating members can understand the project and inspect a small representative dataset without receiving the full private workspace.
- Acceptance:
  - Gitee and GitHub expose the same approved source snapshot, excluding private artifacts.
- The repository contains two-member collaboration instructions and a deterministic starter-pack generator.
  - A starter-data ZIP is generated from local `data/birds`, contains 8 classes with separate train/valid/test samples, and passes manifest/file-count checks.
  - Full data, `.pt` weights, and ONNX binaries remain outside Git.
  - The working tree is clean after the synchronized commit.
- Out of scope: Full-model retraining, degradation implementation, RK3588S deployment, and outdoor collection.
- Dependencies / risks: Remote authentication must permit pushes to both Gitee and GitHub.

### WP-02: Fixed degradation benchmark v1

- Status: DONE
- Outcome: The two junior members jointly deliver a reviewed, reproducible benchmark covering a small agreed set of realistic degradations before any model tuning begins.
- Acceptance:
  - Exactly one owner is assigned to generation/code and one to visual QA/experiment evidence.
  - A 20-image pilot covers five degradation types at light, medium, and heavy levels.
  - Parameters, seeds, source-image IDs, rejection reasons, and generated-file paths are recorded.
  - The project lead accepts the pilot before larger data generation.
- Out of scope: Claiming outdoor performance or tuning on the final test set.
- Dependencies / risks: The pilot may reveal that some synthetic effects are unrealistic and require parameter revision.

### WP-02A: RK3588S local-image inference baseline

- Status: DONE
- Outcome: The real QuarkPi-CA2 runs the existing accepted MobileNetV3 model on one known local image through the RK3588 NPU and writes an evidence record that can be reproduced without camera hardware.
- Acceptance:
  - The board's NPU driver and RKNN runtime compatibility are recorded from the real device.
  - Conversion records the source ONNX, source NCHW / runtime NHWC preprocessing contract, ImageNet normalization, target platform, and FP16 choice in a model manifest.
  - Board inference saves model/input hashes, predicted class, raw score, measured-run count, median latency, and P95 latency.
  - The output agrees with the source ONNX on the selected image, or any discrepancy is preserved and explained before the package can pass review.
- Out of scope: USB camera, bird/no-bird detection, automatic triggering, storage-retention policy, power benchmarking, and any field claim.
- Dependencies / risks: The Debian 11 board image needs a compatible isolated RKNN Python runtime. The connected USB camera is currently absent, which does not block this package.

### WP-03: Measure and improve robustness with controlled fine-tuning

- Status: DONE
- Outcome: Establish whether the existing model needs robustness fine-tuning; deliver a controlled comparison, including a negative result if augmentation does not help. Improvement is not presumed.
- Training protocol: `.agent/PLAN.md` below is authoritative; it supersedes the earlier unconditional two-by-30-epoch proposal and the preliminary exp010 selection rule.

#### Phase A: measurement before GPU training

1. Audit `models/exp005_mobilenetv3_full_best.pt`: model architecture, 524-class mapping, checkpoint configuration and SHA-256. Confirm it matches the deployed ONNX on a fixed validation batch; do not infer equivalence from filenames.
2. Finish pilot QA across all 20 sources, recording accept/reject reasons. Only one source montage has previously been visually inspected. Treat ellipses as synthetic occlusion and downsampling as resolution loss, not realistic foliage or distance proof. Occlusion's current parameter is bounding-box fraction, not actual ellipse area; three severities currently use different positions. Correct/document these before freezing the benchmark, preserving prior artifacts under their old version.
3. Freeze reviewed parameters, dependency versions, source IDs, hashes and exclusion reasons. Use the existing 524-source / 7,860-variant validation set for rapid screening, and all readable validation sources with the same 15 conditions for checkpoint selection. The 15 variants per source are correlated, not 15 independent observations. Check train/valid duplicate hashes and exact label-map coverage. Record the missing CSV class explicitly.
4. Evaluate the existing model M0 on clean validation and all 15 conditions. Report overall and macro top-1 accuracy, per-family/per-level scores and predictions. No training or test-image evaluation is needed for this step. Severe synthetic failure alone does not justify retraining: inspect mild/medium failures for relevance first. If there is no practically meaningful gap, retain M0 and return to device-loop work.

#### Phase B: paired fine-tuning, only if Phase A supports it

- M0: frozen exp005 checkpoint, no updates; deployment reference.
- C: clean continuation from M0, retaining the existing crop/flip/mild color jitter.
- A: augmented continuation from the identical M0 weights, with the same ordinary transforms plus one reviewed degradation, applied with probability 0.5. Draw the five families and three severities uniformly. Put degradation after geometric transforms at 224 pixels, before tensor normalization. Retain the clean half of the inputs.
- C versus A isolates augmentation from extra training; A versus M0 tests whether deployment would benefit. Do not compare only A against an untrained control or an older score from another evaluation setup.
- Initial defaults, to freeze before launch: all layers trainable; AdamW; learning rate 1e-4; weight decay 1e-4; label smoothing 0.1; cosine schedule over 10 epochs to 1e-6; effective batch 256; 224-pixel inputs. Reset optimizer/scheduler identically for C and A, loading model weights only. These are starting engineering choices, not experimentally optimized values.
- Use seed 42 first. Serialize identical initial model state, batch order and ordinary-transform RNG streams for the pair; isolate degradation randomness so it cannot change later sample order or ordinary transforms. Seed workers explicitly; record nondeterministic operations and environment. Use the same physical batch size and accumulation policy for both arms; accumulation does not make BatchNorm equivalent to a larger physical batch.
- Pilot: run the first 3 epochs of the planned 10-epoch schedule for both arms. Check finite losses, real parameter updates, label mapping, clean/robust validation and resource use. This is a feasibility gate, not final evidence. If sound, resume both runs through epoch 10 with optimizer, RNG, sampler and scheduler state preserved. Stop on nonfinite loss, mapping errors or severe unexpected clean collapse; diagnose without consulting test results.
- If the first pair is promising, repeat the same 10-epoch protocol with seeds 43 and 44 (up to six runs total). If not, report the negative result before considering a bounded new validation-only experiment; never silently extend to 30 epochs or search indefinitely.

#### Phase C: fixed selection and final evaluation

- Evaluate clean validation and the same 15 reviewed degraded-validation conditions each epoch for both arms.
- Define R as the equal-weight mean of 15 condition accuracies. Eligible checkpoints must lose at most 0.5 percentage points of clean accuracy against M0. Select each arm's highest R among eligible epochs; ties use higher clean accuracy, then earlier epoch. If none qualify, reject that arm and retain M0.
- Engineering promotion gate: augmented R improves by at least 2 percentage points against both M0 and its paired C, with clean loss at most 0.5 points against both. These are predeclared practical thresholds, not literature-derived guarantees. Report every family and severity, including regressions.
- For repeated runs report paired seed results, mean and standard deviation; require positive A-C robustness differences in all three seeds and the mean practical gate above. Show source-clustered paired bootstrap uncertainty, keeping all variants of a source together. These results apply only to this synthetic benchmark.
- Freeze model hashes, checkpoint choices and a deployment-seed rule before test access (use seed 42 if the repeated protocol passes). Evaluate M0 and the locked C/A checkpoints in one final clean/degraded test batch. The original test set already has historical exp004/exp005 scores, so call it a reused held-out dataset, not a never-seen independent blind test. New field images will supply external validation. Do not use final test scores for another tuning round.
- Re-export any accepted candidate and verify PyTorch/ONNX/FP16 RKNN on a multi-image validation batch before device replacement. INT8 accuracy is a separate conversion/quantization issue; augmentation is not its presumed fix.

#### Execution readiness and compute budget

- Implemented before launch: explicit checkpoint-weight initialization; separate resume path; fixed-degradation validation evaluator; score-based checkpoint selection; independent random streams; a strict no-test training path; run manifests and safe output directories; candidate ONNX export; and source-clustered paired bootstrap. A stop/resume smoke and an end-to-end checkpoint-to-ONNX evaluation smoke verify the plumbing, not model quality.
- No new architecture, from-scratch retraining, paid GPU provisioning or long training is started by this planning task.
- Use the laboratory L20 through Sunlogin under the shared-server rules in `deploy/l20/L20_RUNBOOK.md`: physical GPU 3 only, the user-authorized common `stasis` environment without package changes, no global shell changes, no system-package installation, and hash-verified tar transfer because the server has no Git. Probe fresh occupancy, exact-batch memory, software compatibility and disk first. Measure one full epoch plus validation before estimating remaining time.
- Budget: initial gate = 2 × 3 epochs; complete first pair = 2 × 10 epochs total; repeated evidence = at most 3 × 2 × 10 epochs total. Resume work counts within, not in addition to, those totals. Estimate GPU hours from measured training + validation time and add transfer/setup separately; do not invent a rental cost.
- Deliverables: immutable configs/data manifests/checkpoint hashes; per-image predictions and condition scores; comparison with uncertainty and negative findings; one accepted candidate or explicit retention of M0.
- Result: all six ten-epoch L20 runs completed. The three-seed gate passed, seed 42 was selected by the frozen rule, and its augmented epoch-9 checkpoint passed local ONNX validation and the one-time reused held-out test evaluation. See `experiments/exp014_robustness_finetune_results.md`.

### WP-04: Device observation loop

- Status: IN_PROGRESS
- Outcome: The held RK3588S runs a bird-detection gate followed by the accepted species classifier and records image, time, detection, predicted class, confidence, and latency in a desk-based demonstration.
- Acceptance: The accepted classifier remains installed under a versioned rollback-safe path; the frozen detector preprocessing/decode/NMS/crop contract passes ONNX/RKNN parity on a multi-scene batch; empty scenes do not reach the closed-set classifier; and one captured camera frame produces a timestamped end-to-end record.
- Current result: Classifier ONNX/RKNN parity passed 20/20, the versioned model is installed and selected, and timestamped local-image inference passed. Exp016 shows that full-frame species classification fails on external scenes (`24.1%` top-1), while generic detector crops reach `76.1%` among detected scenes and `64.8%` end to end. Camera capture remains blocked because the current RKISP pipeline has no attached sensor.
- Out of scope: Long-duration unattended outdoor operation.
- Dependencies / risks: Classifier FP16 RKNN inference and 20-image ONNX/RKNN top-1 parity are verified. Detector RKNN parity, camera integration and the observation loop remain unverified. The provisional COCO detector has licensing and domain limits and is not yet an accepted production model. The current CA2 kernel requires Tailscale userspace networking; long-term deployment must also address the default system credential and high-privilege physical ADB access.

### WP-05: Outdoor validation

- Status: PLANNED
- Outcome: The physical prototype is placed outdoors for a bounded observation session, with real observations and failures retained as final evidence.
- Acceptance: To be defined after the device loop is stable.
- Out of scope: Retrospectively relabelling simulated evidence as outdoor evidence.
- Dependencies / risks: Team availability, site access, weather, power, mounting, and wildlife occurrence.

## Current Directive

`WP-03` and accepted-classifier board deployment are complete. Exp016 established the required architecture: detect and crop the bird before species classification, and do not use classifier softmax as an empty-frame gate. The active package is to freeze that detector/crop contract, choose a redistribution-safe detector candidate, convert it to RKNN, and verify ONNX/RKNN end-to-end parity on the proxy batch without changing classifier weights. Then connect the intended sensor, capture one real frame, and build the timestamped observation loop. INT8 remains a separate follow-up.

## Key Decisions

- Repository first, starter data second, full data/model access only after each member runs the starter workflow.
- The two members work together on one degradation outcome but have separate ownership: generation/code versus QA/evidence.
- The project lead retains model-training decisions and all device operations while the device remains in their possession.
- Treat remote access as development infrastructure only; it does not satisfy any RKNN, inference, camera, or field acceptance criterion.
