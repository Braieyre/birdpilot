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

- The committed local repository and Gitee `main` are at `25d5804`.
- GitHub `main` is older and needs synchronization.
- MobileNetV3 training, ONNX export, and a local Mac benchmark have historical artifacts.
- RKNN conversion, device inference, degradation training, and outdoor validation are not yet complete.
- Full data, the primary `.pt` checkpoint, and ONNX outputs are intentionally excluded from Git.

## Roadmap

1. Give new members a synchronized repository, a reproducible starter-data package, and a clear onboarding path.
2. Establish a fixed degradation benchmark and quality-review process before changing model weights.
3. Train and compare the clean baseline and degradation-augmented model without touching the held-out test set.
4. Deploy the accepted model to the RK3588S and complete a desk-based observation loop.
5. Perform final outdoor observation and report it separately from simulated evidence.

## Work Packages

### WP-01: Shared repository and newcomer starter package

- Status: REVIEW
- Outcome: Both repositories expose the same safe code snapshot, and new members can understand the project and inspect a small representative dataset without receiving the full private workspace.
- Acceptance:
  - Gitee and GitHub `main` resolve to the same new commit.
  - The repository contains newcomer instructions and a deterministic starter-pack generator.
  - A starter-data ZIP is generated from local `data/birds`, contains 8 classes with separate train/valid/test samples, and passes manifest/file-count checks.
  - Full data, `.pt` weights, and ONNX binaries remain outside Git.
  - The working tree is clean after the synchronized commit.
- Out of scope: Full-model retraining, degradation implementation, RK3588S deployment, and outdoor collection.
- Dependencies / risks: Remote authentication must permit pushes to both Gitee and GitHub.

### WP-02: Fixed degradation benchmark v1

- Status: PLANNED
- Outcome: The two junior members jointly deliver a reviewed, reproducible benchmark covering a small agreed set of realistic degradations before any model tuning begins.
- Acceptance:
  - Exactly one owner is assigned to generation/code and one to visual QA/experiment evidence.
  - A 20-image pilot covers five degradation types at light, medium, and heavy levels.
  - Parameters, seeds, source-image IDs, rejection reasons, and generated-file paths are recorded.
  - The project lead accepts the pilot before larger data generation.
- Out of scope: Claiming outdoor performance or tuning on the final test set.
- Dependencies / risks: The pilot may reveal that some synthetic effects are unrealistic and require parameter revision.

### WP-03: Robustness training and comparison

- Status: PLANNED
- Outcome: A degradation-augmented model improves the fixed simulated benchmark without unacceptable loss on clean images.
- Acceptance: To be made precise after WP-02 fixes the benchmark.
- Out of scope: Outdoor claims and device claims.
- Dependencies / risks: Improvement may be degradation-specific or trade off against clean accuracy.

### WP-04: Device observation loop

- Status: PLANNED
- Outcome: The accepted model runs on the held RK3588S device and records image, time, predicted class, confidence, and latency in a desk-based demonstration.
- Acceptance: To be made precise after WP-03 selects the model.
- Out of scope: Long-duration unattended outdoor operation.
- Dependencies / risks: RKNN conversion and device runtime compatibility remain unverified.

### WP-05: Outdoor validation

- Status: PLANNED
- Outcome: The physical prototype is placed outdoors for a bounded observation session, with real observations and failures retained as final evidence.
- Acceptance: To be defined after the device loop is stable.
- Out of scope: Retrospectively relabelling simulated evidence as outdoor evidence.
- Dependencies / risks: Team availability, site access, weather, power, mounting, and wildlife occurrence.

## Current Directive

Review `WP-01` only.

## Key Decisions

- Repository first, starter data second, full data/model access only after each member runs the starter workflow.
- New members work together on one degradation outcome but have separate ownership: generation/code versus QA/evidence.
- The project lead retains model-training decisions and all device operations while the device remains in their possession.
