# BirdPilot L20 training runbook

This runbook adapts the confirmed rules in Liaoning's `CLOUD_RUNBOOK.md` and `L20_STATUS.md` to BirdPilot. The L20 is reachable only through Sunlogin. Commands below run in the visible terminal on the L20; no SSH path is assumed.

## Shared-server rules

- Use physical GPU 3 only. Every new terminal must set `CUDA_VISIBLE_DEVICES=3`; never add it to `~/.bashrc`.
- Do not stop, renice, inspect private files from, or otherwise alter other users' processes.
- Do not use `sudo apt`, `conda install`, or `uv`.
- The user has explicitly authorized BirdPilot to reuse the common `stasis` environment. Activate and inspect it, but do not upgrade or install packages during an active experiment. If a dependency is missing, stop and decide the smallest `pip` change with the user first.
- There is no `git`, `tmux`, or `screen` on the server. Leave the terminal window open. A Sunlogin display disconnect does not itself stop the foreground process.
- Do not upload test images for Phase B. Training and selection use train and validation data only.

## 1. Fresh read-only probe

Before transferring or training, run Liaoning's current probe or transfer the BirdPilot source package and run:

```bash
bash deploy/l20/preflight.sh
```

Treat the fresh `nvidia-smi -i 3` output as authoritative. Historical occupancy and temperature are context only.

## 2. Transfer and verify

Use Sunlogin to copy the four files from the Mac transfer directory into an empty temporary directory on the L20:

- `birdpilot_source.tar`
- `birdpilot_train_valid_data.tar`
- `birdpilot_training_assets.tar`
- `SHA256SUMS`

Then run:

```bash
mkdir -p /mnt/sdc/Stasis/birdpilot_transfer /mnt/sdc/Stasis/birdpilot && cd /mnt/sdc/Stasis/birdpilot_transfer && sha256sum -c SHA256SUMS && tar -xf birdpilot_source.tar -C /mnt/sdc/Stasis/birdpilot && tar -xf birdpilot_train_valid_data.tar -C /mnt/sdc/Stasis/birdpilot && tar -xf birdpilot_training_assets.tar -C /mnt/sdc/Stasis/birdpilot
```

This refuses to proceed past a bad checksum because the commands are joined with `&&`.

## 3. Common environment

BirdPilot reuses the user-authorized common `stasis` environment. Inspect it without changing packages:

```bash
conda activate stasis && cd /mnt/sdc/Stasis/birdpilot && python -c 'import torch,torchvision,numpy,pandas,PIL; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torchvision.__version__,numpy.__version__,pandas.__version__,PIL.__version__)'
```

The training gate needs PyTorch, torchvision, NumPy, pandas, and Pillow. ONNX and ONNXRuntime are needed only after a candidate is selected. If an import fails, report the exact output before making any environment change.

## 4. Verify payload and batch fit

```bash
cd /mnt/sdc/Stasis/birdpilot && conda activate stasis && export CUDA_VISIBLE_DEVICES=3 && python -m py_compile src/*.py deploy/l20/*.py && python -c 'from src.train_robustness import load_config; print(load_config(__import__("pathlib").Path("configs/exp014_augmented_seed42.json")))' && bash deploy/l20/run_exp014.sh probe
```

The probe performs disposable forward/backward updates on random tensors using the frozen batch size 256. It writes GPU name, software versions, free memory, peak memory and step timing under `outputs/l20_evidence/`. It does not change a formal checkpoint.

## 5. Three-epoch gate

Only after the probe fits without OOM:

```bash
cd /mnt/sdc/Stasis/birdpilot && conda activate stasis && export CUDA_VISIBLE_DEVICES=3 && bash deploy/l20/run_exp014.sh gate
```

This runs C to epoch 3, then A to epoch 3. If interrupted, run the same command again: completed epochs are detected from `metrics.jsonl`, and an incomplete arm resumes only from its own `last.pt`. Never change the config or batch size during a resumed run.

Inspect without using the GPU:

```bash
cd /mnt/sdc/Stasis/birdpilot && conda activate stasis && bash deploy/l20/run_exp014.sh status
```

Do not continue to epoch 10 until the epoch-3 losses, clean accuracy, all condition metrics, actual elapsed time, GPU occupancy and disk growth have been reviewed.

## 6. Continue after the gate decision

```bash
cd /mnt/sdc/Stasis/birdpilot && conda activate stasis && export CUDA_VISIBLE_DEVICES=3 && bash deploy/l20/run_exp014.sh complete
```

After the seed-42 gate passes, `complete` resumes seed 42 to epoch 10, runs the paired C/A protocol for seeds 43 and 44, and writes the three per-seed selections plus the repeated-seed summary. Re-running the same command skips completed runs and resumes any run that has a valid `last.pt`. A rejection keeps M0 deployed and is a valid result.

## 7. Return to the Mac

Return these files through Sunlogin before any cleanup:

- `outputs/l20_evidence/`
- `outputs/robustness_runs/exp014_clean_seed42/{run_manifest.json,metrics.jsonl,best.pt,last.pt}`
- `outputs/robustness_runs/exp014_augmented_seed42/{run_manifest.json,metrics.jsonl,best.pt,last.pt}`
- `outputs/l20_logs/`
- `outputs/robustness_runs/exp014_seed42_selection.json`

Keep the server copy until hashes and checkpoint loading have been verified on the Mac.
