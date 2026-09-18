# QuarkPi-CA2 远程开发与设备接入

> 验证日期：2026-09-18
> 证据边界：本文只证明开发板接入、远程维护通道和联网切换已经跑通，不代表 RKNN 转换、板端模型推理、摄像头闭环或户外验证已经完成。

## 1. 已验证环境

| 项目 | 实测结果 |
|---|---|
| 开发板 | QuarkPi-CA2（RK3588S） |
| CPU 架构 | `aarch64` |
| 系统 | Debian GNU/Linux 11（bullseye） |
| 内存 | 约 15 GiB |
| 系统盘 | 约 114 GiB |
| 内核 | Linux 5.10.209 |
| 本地救援通道 | USB OTG 暴露 Rockchip ADB 接口；Mac 可执行 `adb shell` |
| 临时联网 | 手机 USB 网络共享，板端接口为 `usb0` |
| 跨网维护 | Tailscale 1.102.4 + Tailscale SSH |
| Tailscale 模式 | `userspace-networking` |

实测完成了以下链路：

```text
Mac ──USB OTG/ADB──> CA2 root shell

Mac ──任意互联网──> Tailscale ──手机 USB 网络──> CA2
                                      ↓
                                  Tailscale SSH
```

切换网络后，Mac 仍能通过 CA2 的 Tailscale 设备身份建立 SSH 会话。Tailscale 地址属于设备维护面，不写入公开文档。

## 2. 为什么没有使用标准 TUN 模式

CA2 当前内核启动 Tailscale 时出现过以下关键错误：

```text
modprobe: FATAL: Module tun not found in directory /lib/modules/5.10.209
tstun.New("tailscale0"): no such device
```

手工创建 `/dev/net/tun` 设备节点后仍返回 `no such device`，说明问题不是节点缺失，而是当前内核没有可用的 TUN 支持。因此没有刷机或替换内核，而是在 `/etc/default/tailscaled` 中使用：

```text
FLAGS="--tun=userspace-networking"
```

此后 `tailscaled` 能稳定启动并创建本地 socket，Tailscale SSH 实测可用。这个兼容结论只适用于当前 CA2 系统镜像；其他 Linux 边缘设备应先尝试标准 TUN 模式。

## 3. 日常开发方式

### 3.1 首选：Tailscale SSH

设备在线并处于同一 tailnet 时，从开发电脑连接：

```bash
ssh <device-user>@<tailscale-ip-or-magicdns-name>
```

网络切换不会改变设备的 Tailscale 身份。两端能访问互联网、Tailscale 服务正常且访问策略允许时，设备可以位于手机共享、实验室网络或其他外部网络中。

### 3.2 现场救援：USB OTG + ADB

当板子网络失效但设备在手边时：

```bash
adb devices -l
adb shell
```

当前系统的 ADB shell 具有高权限。它适合救援，但也是物理安全风险：长期部署前应确认能否关闭、限制或改为受控调试模式，不能把“需要接触设备”视为充分保护。

### 3.3 网络自检

板端：

```bash
ip -brief address
systemctl is-active tailscaled
tailscale status
```

开发电脑：

```bash
tailscale status
tailscale ping <device-name-or-ip>
```

## 4. 实验室长期部署前的安全清单

- 修改系统镜像的默认用户密码，不在文档、镜像或脚本中保存明文密码；
- 把设备从个人临时身份迁移为项目/实验室管理的设备身份；
- 使用标签和访问策略，仅允许指定维护人员或维护设备访问；
- 不需要远程 shell 时关闭 Tailscale SSH，而不是默认向整个 tailnet 开放；
- 评估并限制 USB OTG ADB 高权限入口；
- 保留本地数据存储，网络中断不能导致观测证据丢失；
- 建立设备丢失、移交和报废时的移除/注销流程；
- 不把 Tailscale 状态目录、节点私钥或个人登录状态复制到其他设备。

## 5. 从一台原型到批量设备

### 原型阶段（当前）

允许人工登录一台板子，用于验证联网、远程维护和后续板端推理。当前成果属于开发基础设施，不属于 BirdPilot 观测闭环证据。

### 小批量阶段

制作统一系统镜像和首次启动脚本。每台设备首次启动时：

1. 读取硬件序列号或分配唯一设备编号；
2. 生成唯一主机名；
3. 使用短期、受限的注册凭据加入项目 tailnet；
4. 获得独立设备身份和密钥；
5. 写入设备清单并删除临时凭据。

不得复制已经登录的 `/var/lib/tailscale`，否则会让多台设备共享身份和密钥。

### 量产阶段

使用项目或企业 tailnet、标签化设备身份、细粒度访问策略，以及由部署服务动态签发的短期注册凭据。建议的角色边界是：

```text
研发设备  ──> 允许研发人员维护
实验室设备 ──> 允许项目维护人员维护
现场设备  ──> 默认只允许运维服务和最小必要端口
设备之间  ──> 默认不互访
```

量产镜像只包含客户端和首次启动逻辑，不包含个人账号会话、长期通用密钥或某台原型机的状态目录。

## 6. 与 BirdPilot 路线的关系

本里程碑消除了“如何把代码、模型和诊断命令送到板子上”的开发阻塞。后续设备证据仍必须按顺序取得：

```text
远程开发通道（已验证）
    ↓
RKNN 环境与模型转换（未验证）
    ↓
本地图像推理与结果记录（未验证）
    ↓
USB 摄像头接入（未验证）
    ↓
桌面观测闭环（未验证）
    ↓
户外验证（未验证）
```
