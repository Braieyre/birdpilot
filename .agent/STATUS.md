# Execution Status

## Active Package

`WP-02`

## State

READY

## Delivered

- Lightweight Planner-Executor-Reviewer coordination initialized.
- Newcomer instructions added in `TEAM_ONBOARDING.md` and linked from `README.md`.
- Deterministic starter-pack generator added in `src/make_starter_pack.py`.
- Starter data created outside Git at `team-share/BirdPilot_starter_data_v1.zip`.
- ONNX runtime dependencies added to `requirements.txt`.

## Evidence

- Local committed `main`: `25d5804`.
- Live Gitee `main`: `25d5804`.
- Live GitHub `main`: `992589b` and therefore not synchronized yet.
- Full data is present locally but ignored by Git; primary `.pt` and ONNX artifacts are also ignored.
- Starter package: 8 classes, 160 train, 40 valid, 40 test, 240 unique images total.
- ZIP integrity: passed `unzip -t`.
- Manifest: all 240 files present and every SHA-256 matched.
- Reproducibility: two independent builds produced ZIP SHA-256 `051c0de75c897becd2343e288ec1cc09b463605b8508beecb883edabe24c33ab`.
- Visual sample: one image from each of the 8 classes inspected and correctly labelled.
- Python syntax: `src/make_starter_pack.py` and `src/recover_eval.py` compiled successfully using the bundled Python runtime.
- `src/recover_eval.py --skip-test`: not runtime-verified because the available bundled environment does not include PyTorch; PyTorch remains declared in `requirements.txt`.
- Local commits: `769b960`, `3e74a99`, `72c8f19`.
- Gitee `main`: `72c8f196937ab6bd2c6659e3c438056eae4a12ca`.
- Public GitHub `main`: `b5a664c1bd48eecd17a39e51a3ab6559e388aae4` (API-created ordinary commit from the prior public history).
- GitHub tree verified to contain onboarding and experiment evidence, with no `data/` or `models/` paths.
- The local H5 model remains on disk for private use but is no longer tracked.
- Both repositories are public-code/private-data aligned; their commit histories differ because GitHub started from an older independent history.

## Deviations

- The repository initially had no shared coordination files; they were initialized for this handoff.

## Working State

- Existing user work is preserved.
- Tracked AppleDouble metadata files are already deleted in the working tree and will be removed from the repository.
- `CLAUDE.md` and `src/recover_eval.py` were pre-existing untracked user files and are under review for inclusion.

## Next Action

Assign WP-02: jointly define a 20-image, five-degradation pilot; give one junior ownership of generation/code and the other ownership of visual QA/evidence; wait for lead acceptance before scaling or training.

## Blocker

None.
