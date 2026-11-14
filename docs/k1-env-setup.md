# MUSE Pi Pro（SpacemiT K1）开发环境搭建实录

板子：MUSE Pi Pro，SpacemiT K1（8 核 RISC-V，riscv64），Bianbu 系镜像，内核 6.6.63。
目标：跑通"MIPI 相机 + ONNX 推理 + Web 预览"（代码在 `apps/yolo11seg-web/`）。
一句话：**能全走 apt 就别走 pip**，Bianbu 是 PEP668 externally-managed 环境，
AI 相关的包 apt 源里都有现成的（且是 riscv64 原生编译过的）。

## 1. 拿系统、拿权限

- 官方手册：<https://www.spacemit.com/community/document/info?lang=zh&nodepath=ecohardware/k1_series/k1_muse_pi_pro/pi_pro_user_guide.md>
- 默认从板载存储启动进 Bianbu desktop；首次开机有配置向导（需要 HDMI + 键盘鼠标）。
  开机 3 秒内按 F2 进 UEFI 选启动介质。
- 联网：`ip addr`（接口名一般是 `end*`）；Wi-Fi 用 `nmcli device wifi connect <SSID> password <pw>`。
- SSH 默认开启（`systemctl status ssh`）。**强烈建议一开始就配公钥免密**，
  后面的自动化脚本才不会到处塞密码：

  ```bash
  ssh-keygen -t ed25519                       # PC 端
  ssh-copy-id <user>@<板子IP>                 # 或手工把 pub 追加到 ~/.ssh/authorized_keys
  ```

  密码只放在 PC 端**不入库**的文件里（自己的凭据管理工具，或同步仓库里被 gitignore 的机器专属文件），不要提交进本仓库。

- **默认 shell 是 zsh**：远程执行多条命令请用 `ssh <user>@<ip> bash -s <<'EOS' ... EOS`，
  否则 `[[ ]]`、数组、`shopt` 之类会静默踩差异；`scp`/`rsync` 的路径通配也按 zsh 规则展开。

## 2. 装依赖（全部 apt）

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-opencv python3-numpy python3-flask python3-onnx \
    python3-pip gcc libopencv-dev \
    onnxruntime python3-spacemit-ort
```

| 包 | 作用 | 备注 |
|---|---|---|
| `onnxruntime` | ORT 主库 | 装完在 `/usr/lib/libonnxruntime.so.1.18.1` |
| `python3-spacemit-ort` | SpaceMIT EP 的 python 注入（`import spacemit_ort`） | 版本 1.2.2 系 |
| `python3-opencv` / `libopencv-dev` | 图像前后处理 | 别 pip 装 opencv-python，riscv64 上没有合适 wheel |
| `python3-onnx` | 改 opset / 看图结构 | |
| `gcc` | 编 `k1cam.c` | 板上编即可，几十行代码 |

验证：

```bash
python3 -c "import cv2,numpy,onnx,onnxruntime; print(cv2.__version__, onnxruntime.__version__)"
python3 -c "import spacemit_ort, onnxruntime as ort; print(ort.get_available_providers())"
```

### 2.1 pip 相关

- `pip install xxx` 会报 `externally-managed-environment`（PEP668）。三种选择：
  1. 先看 apt 有没有（99% 有，见上表）；
  2. 真要 pip：`pip install --break-system-packages xxx`（只在自测环境用）；
  3. 干净做法：`python3 -m venv ~/work/venv`，但 **EP 的 `spacemit_ort` 是系统包**，
     venv 里要么 `--system-site-packages`，要么把 `PYTHONPATH` 指到系统 site-packages，
     否则会出现"ORT 能导入但 provider 列表里没有 SpaceMIT"这种半残状态。
- 板子可以直连 PyPI 与 `archive.spacemit.com`（本次开发下模型/工具都是板上直接 `curl` 下来的），
  需要 HuggingFace 时用 `HF_ENDPOINT=https://hf-mirror.com`。

### 2.2 找不到 python 包时的兜底

镜像里预装了一些自带依赖的应用（例如语音相关的包会把 flask/cv2 装在应用自己的目录下）。
缺哪个包可以先找：

```bash
python3 - <<'PY'
import os, glob
hit=[p for p in ("/opt","/usr/share") for pat in ("flask","cv2") for x in glob.glob(os.path.join(p,"**",pat)) ]
print("\n".join(sorted(set(hit))[:20]))
PY
```

找到后 `PYTHONPATH=<其父目录> python3 ...` 临时救急。**不要**把它当成常规方案，
`setup-board.sh` 走的是 apt 正路。

## 3. 相机侧只需要一件事

`apt` 装完后，相机点亮与校验请用 `apps/yolo11seg-web/cam-start.sh`（原因见
`docs/k1-mipi-camera.md` —— 板子重启后不会有任何服务去起 `cam-test`）。

## 4. 模型侧只需要两件事

1. opset ≤ 21（ORT 1.18.1 的上限；官方 yolo11n-seg 是 22，会加载失败）——
   `apps/yolo11seg-web/export_onnx.py` / `setup-board.sh` 第 4 步自动处理。
2. 想要快就得给**激活为 int8 的量化模型**（`apps/yolo11seg-web/quant/run-quant.sh` 一条命令产出）——
   判据与账单见 `docs/k1-aipu-onnxruntime.md` §3.4；另注意 K1 上没有独立 NPU，省的是 CPU 工作量（同篇 §8）。

## 5. 服务化（可选但推荐）

```bash
cd ~/work/yolo11seg-web && ./setup-board.sh --service
systemctl status muse-camtest muse-yolo11seg-web
journalctl -u muse-yolo11seg-web -n 50
```

两个 unit 的依赖关系是 `web Requires=camtest`，因为 `/dev/video50` 只允许一个采集客户端，
而帧必须由 cam-test 产生的 pipeline 供出来。

## 6. 这份环境里已知的坑（速查）

| 症状 | 真因 | 去处 |
|---|---|---|
| `Opset 22 is under development ... support ... till opset 21` | 官方 ONNX 的 opset 太高 | `docs/k1-aipu-onnxruntime.md` §4 |
| EP 在 `get_providers()` 里但速度没变 | float Conv 拿不到 int8 kernel（预期行为） | 同上 §2 / §3.4 |
| 日志 `Attempt to use DefaultLogger but none has been registered` | 没设 `LD_PRELOAD` | 同上 §1.1 |
| 采到的帧全 0 / 满屏噪声 | cam-test 没起或多开、dma-buf flags | `docs/k1-mipi-camera.md` §2 |
| `error: externally-managed-environment` | PEP668 | 本文 §2.1 |
| 远程脚本行为诡异 | 默认 zsh | 本文 §1 |
