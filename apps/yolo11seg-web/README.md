# MUSE Pi Pro（SpacemiT K1）上跑 YOLO11-seg 实时分割 + Web 预览

一条链路：OV5647(MIPI) → `/dev/video50`(spacemit vivi 虚拟相机) → `k1cam` 采 NV12 →
ONNX Runtime(+SpaceMIT EP) 推理 YOLO11n-seg → Flask 出 MJPEG 网页。
浏览器打开 `http://<板子IP>:8000` 就能看到带 mask 的实时分割画面。

## 现状（都是板上实测数字，不是估计）

| 项目 | 结果 |
|---|---|
| 采集 | 1920x1080 11.8 fps / 1280x720 12.9 fps（`k1cam` 单进程），web 侧稳定 ~15 fps |
| 推理（float yolo11n-seg @640，CPU 后端） | 单帧 ~968 ms ≈ **1.0 fps**，20 核全占用 |
| 推理走 SpaceMIT EP（智算核） | **float 模型不加速**，EP 一个 Conv 都不接管（见 `docs/k1-aipu-onnxruntime.md`） |
| 量化后（int8 QDQ） | yolov8n-seg 实测 204 ms/帧 ≈ 4.9 fps；yolo11n-seg 量化模型已生成，待板上复验 |
| 结果正确性 | `bus.jpg` 上 CPU 与 EP 输出完全一致：bus 0.89 / person 0.88 / 0.85 / 0.82 / 0.37 |

> 摄像头在暗环境（夜里/无光）下采到的帧 Y 均值 <4，画面近乎全黑，此时 `dets` 为空是正常的，
> 不是链路坏了。没有光/不想接摄像头时用 `STILL=<图片> ./run-web.sh` 演示分割链路。

## 验证矩阵（哪条结论是在哪儿跑出来的）

| 项 | 在哪验证 | 证据 |
|---|---|---|
| `k1cam` 采集 1080p/720p 帧率、NV12 内容 | **板子**（reboot + 单实例 cam-test 后） | 与官方 `v4l2_test_spacemit` 同帧像素统计一致，见 `docs/k1-mipi-camera.md` §3 |
| float yolo11n-seg CPU 968 ms / EP 1060 ms、检出的 5 个目标 | **板子** | `yolo_seg.py --prefer spacemit`，两后端输出逐目标相同 |
| `server.py` 全接口（/health /stats /video /snapshot.jpg /config） | 本机(aarch64, ORT 1.32)冒烟 + 板上跑过 | `/stats` 有 5 个目标、`/video` 5 s 出流、`POST /config {"conf":0.5}` 后低分框掉出 |
| int8 QDQ 模型精度（只量 Conv / 全图+排除 Concat） | **仅 PC**（ORT CPU EP，bus.jpg） | `0.89/0.86/0.85/0.81/0.36` vs float `0.89/0.88/0.85/0.82/0.37` |
| int8 QDQ 模型能否真的进 AIPU、耗时 | **待板子复验**（开发板 2026-11-14 00:49 起失联，未能上电） | 用 `SPACEMIT_EP_DUMP_SUBGRAPHS=1` 看有没有 SpineSubgraph 产出 |
| `setup-board.sh` 全流程 | 分步在板上手工验证过，脚本整体串跑待复验 | 各步命令与板上实测一一对应 |

## 目录结构

```
apps/yolo11seg-web/
├── setup-board.sh      # 板端一键：apt 依赖 → 编 k1cam → 配相机 → 下模型/降 opset → 单帧自检
├── cam-start.sh        # 点亮相机：保证只有一个 cam-test + 用像素统计校验"真的出图了"
├── run-web.sh          # 启动 web（处理 LD_PRELOAD，参数用环境变量覆盖）
├── export_onnx.py      # PC 端：下载官方 yolo11n-seg.onnx 并降到板端能加载的 opset
├── cam/k1cam.c         # V4L2(m2m)+dma-buf 采集器，往 stdout 吐 NV12 裸帧
├── web/server.py       # Flask 后端：采集线程 + 推理线程(跳帧不排队) + MJPEG
├── web/yolo_seg.py     # 预处理/后处理(NMS、mask 合成)/绘制，可单独当命令行自测工具
├── web/index.html      # 前端：img 显示 /video，侧栏显示 stats、调 conf/掩码透明度
└── quant/quant_qdq.py  # PC 端 int8 QDQ 量化（走智算核的唯一路）
```

## 5 分钟跑通

### 1. PC 端准备模型（可选，板子能直连 GitHub 时跳过）

```bash
pip install onnx
python3 export_onnx.py --opset 17          # 生成 models/yolo11n-seg_opset17.onnx
```

官方导出的是 opset 22，板端 onnxruntime 1.18.1 只支持到 opset 21，**加载就会报错**，所以必须降。
降 opset 与"能不能上智算核"无关，别指望它提速。

### 2. 拷到板子上

```bash
rsync -az --exclude '*.onnx' apps/yolo11seg-web/ <user>@<板子IP>:work/yolo11seg-web/
# 模型单独拷（板子自己下载也行）
scp models/yolo11n-seg_opset17.onnx <user>@<板子IP>:work/yolo11seg-web/models/
```

### 3. 板端一键准备

```bash
cd ~/work/yolo11seg-web
./setup-board.sh                 # 依赖 + 编译 + 相机 + 模型 + 自检
sudo ./setup-board.sh --service  # 需要开机自启时（装 muse-camtest / muse-yolo11seg-web 两个 unit）
```

`MUSE_SUDO_PASS='xxx' ./setup-board.sh` 可以免交互输 sudo 密码；密码不要写进脚本或仓库。

### 4. 启动 + 看

```bash
./cam-start.sh        # 显示 [ok] 出图正常（或提示"画面极暗"）后再往下
./run-web.sh          # 前台；或 PORT=8080 CONF=0.3 ./run-web.sh
```

浏览器 `http://<板子IP>:8000`。接口：`/video`(MJPEG)、`/snapshot.jpg`、`/stats`、
`/config`(GET/POST 调 conf/iou/mask 透明度)、`/health`。

## 可调参数

`run-web.sh` 的环境变量：`PORT` `MODEL` `CONF` `WIDTH`(+`HEIGHT`) `PREFER`(auto|spacemit|cpu)
`STILL`(静态图) `VIDEO`(mp4)。等价的 `server.py` 参数见 `--help`
（另有 `--cam --dev --cam-log --iou --mask-mode --quality --host`）。

## 想上智算核（把 1 fps 变成 5 fps）

```bash
pip install onnx onnxruntime numpy opencv-python-headless    # PC 端，不需要 torch
python3 quant/quant_qdq.py models/yolo11n-seg_opset17.onnx yolo11n-seg_qdq_conv.onnx --images bus.jpg
scp yolo11n-seg_qdq_conv.onnx <user>@<板子IP>:work/yolo11seg-web/models/
MODEL=../models/yolo11n-seg_qdq_conv.onnx PREFER=spacemit ./run-web.sh
```

两种配方（只量 Conv / 全图 QDQ）的区别和坑写在 `quant/quant_qdq.py` 文件头。
量化后务必先跑 `python3 web/yolo_seg.py --model <量化模型> --image bus.jpg` 对精度，
再上摄像头。

## 遇到问题先看这三篇

| 症状 | 去处 |
|---|---|
| 装不上包 / EP 库找不到 | `docs/k1-env-setup.md` |
| 采到的帧全黑、全噪声、只出 1~2 帧、重启后没画面 | `docs/k1-mipi-camera.md` |
| 模型加载失败、provider 里没有 SpaceMIT、EP 不提速 | `docs/k1-aipu-onnxruntime.md` |
