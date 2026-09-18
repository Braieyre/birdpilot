# Execution Status

## Active Package

`WP-02`

## State

READY

## Delivered

- Lightweight Planner-Executor-Reviewer coordination is initialized.
- Two-member collaboration instructions and a deterministic starter-pack generator are present.
- The approved public-code/private-data snapshot is available in the repository; full data, primary weights, and ONNX binaries remain outside Git.
- QuarkPi-CA2 remote development and recovery paths are documented in `deploy/rk3588/REMOTE_ACCESS_CN.md`.
- USB OTG/ADB rescue access and cross-network Tailscale SSH were verified on the real board.

## Current Evidence

- Repository baseline before the remote-access milestone: local and Gitee `main` at `ac8e0ab`; public GitHub uses a separate history with an equivalent approved source snapshot.
- Starter package: 8 classes, 160 train, 40 valid, 40 test, 240 unique images; ZIP integrity, manifest hashes, reproducibility, and an eight-class visual sample were previously checked.
- Historical model evidence covers training, ONNX export/consistency, and Mac ONNX Runtime benchmarking only.
- Device identity check: `quarkpi-ca2`, `aarch64`, Debian 11 (bullseye), Linux 5.10.209, approximately 15 GiB RAM and 114 GiB system storage.
- Board networking check: phone USB tethering appeared as `usb0`; Tailscale 1.102.4 authenticated and remained reachable after the network path changed.
- USB recovery check: the CA2 USB OTG port exposed a Rockchip ADB interface and allowed a root rescue shell from the Mac.
- Kernel compatibility check: the shipped kernel has no usable TUN support (`Module tun not found` and `no such device` after creating `/dev/net/tun`); `FLAGS="--tun=userspace-networking"` produced an active daemon and working local socket.
- End-to-end remote check: the Mac reached the board through Tailscale and executed `hostname`, `whoami`, and `uname -m` over SSH, returning `quarkpi-ca2`, the expected user, and `aarch64`.

## Evidence Boundary

- Remote access is development infrastructure, not model or observation evidence.
- RKNN conversion, RK3588 inference, camera integration, power benchmarking, the desk observation loop, and outdoor validation remain unverified.
- The current system still requires long-term security work before unattended deployment: replace default credentials, define restricted tailnet access, and assess the high-privilege physical ADB interface.

## Working State

- Existing user work is preserved.
- Existing untracked hardware-product documentation under `docs/` is not part of the remote-access milestone.
- The remote-access change is limited to the two READMEs, `deploy/rk3588`, and the two coordination files.

## Next Action

Execute WP-02 only: define and review the 20-image, five-degradation pilot before any model tuning or device inference work.

## Blocker

None.
