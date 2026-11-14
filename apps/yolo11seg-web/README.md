# MUSE Pi Pro（SpacemiT K1）上跑 YOLO11-seg 实时分割 + Web 预览

一条链路：OV5647(MIPI) → `/dev/video50`(spacemit vivi 虚拟相机) → `k1cam` 采 NV12 →
ONNX Runtime(+SpaceMIT EP) 推理 YOLO11n-seg → Flask 出 MJPEG 网页。
浏览器打开 `http://<板子IP>:8000` 就能看到带 mask 的实时分割画面。

## 现状（都是板上实测数字，不是估计）

| 项目 | 结果 |
|---|---|
| 采集（1280x720 NV12） | web 侧 `cam.fps` 稳定 **15.3~15.4**，72s 无重启无黑帧；`cam-start.sh` 校验 371 帧/24s（13.7 fps，差值是校验脚本自己逐帧做 numpy 统计） |
| 推理（float yolo11n-seg @640，**当前默认**） | 单帧 968 ms（纯推理基准）/ 1400~1570 ms（web 里和采集线程抢核时），**≈0.58~0.65 fps** |
| 推理走 SpaceMIT EP（智算核） | **float 模型不加速**：Placement 日志显示 97 个 Conv 全在 CPU（判据与矩阵见 `docs/k1-aipu-onnxruntime.md` §1.2/§2.1） |
| 官方 int8 `yolov8n-seg.q.onnx`（v8，13 输出） | **单帧 408 ms（EP）vs 1371 ms（CPU）**、`bus.jpg` 精度对得上（bus 0.87 / person 0.87/0.85/0.82），解码器已支持。崩不崩几乎由线程数决定：`EP_THREADS=4` 且 ORT 线程池也收到 4 → **32 min / 4559 次 0 崩（2.43 fps）**；线程池留在 8（旧默认）→ **5 min 崩 6 次**。`_tie_intra()` 已自动配平，但样本是一块板一次长跑，常驻仍建议配看门狗 |
| 自产 int8（QDQ / 改写成 QLinearConv） | 3067~3176 ms，**比 float 更慢**：QLinearConv 仍落 CPU |
| 画面出流（MJPEG 1280x720 q80） | 约 6 帧/s，瓶颈是 K1 上的 JPEG 编码；`/video` 3s 实测 19 帧 |
| 结果正确性 | `bus.jpg` 上 CPU 与 EP 输出完全一致：bus 0.89 / person 0.88 / 0.85 / 0.82 / 0.37（板上 `/stats` 复核过） |
| 崩溃自愈 | `supervise-web.sh` 看门狗：web 进程消失（含 SIGSEGV）2s 后重开，实测 **32s 崩 → 2s 恢复**，`/stats` 重新出数 |

所以现在的"实时"是这样的（默认 float 模型）：**浏览器看到的画面 15 fps，mask 大约每 1.6s 更新一次**
（HUD 上 `infer ... fps` 是真实分割帧率，`view ... fps` 是画面帧率，不假装实时）。
`--still` 模式实测 infer 0.86 fps（970 ms/帧）、5 个目标全部正确，说明低的是算力不是链路。

> 摄像头在暗环境下 Y 均值 <4 时画面近乎全黑、`dets` 为空是正常的。但**先分清"没光"和"假帧"**：
> `curl -s localhost:8000/stats` 看 `cam.lum`（Y 的 std），`lum<2 且 lum_mean<4` 是假帧
> （有多发 cam-test 在抢 pipeline），有噪声纹理的黑才是真没光。详见 `docs/k1-mipi-camera.md` §3。

## 效果参考（都是板上 `/snapshot.jpg` 原图，未二次加工）

|  |  |
|---|---|
| ![live](docs/web-live-720p.jpg) | ![still](docs/web-still-bus.jpg) |
| 摄像头实时帧 1280x720（室内偏暗，Y p50=17/p99=91，无目标 → `dets` 为空） | 同一套 web、同一份模型，`STILL=bus.jpg` 时的分割叠加（bus 0.89 / person 0.88/0.85/0.82/0.37） |

## 验证矩阵（哪条结论是在哪儿跑出来的）

| 项 | 在哪验证 | 证据 |
|---|---|---|
| 相机零干预出图（reboot 后不手动起任何 cam-test/服务） | **板子** 2026-11-14 10:13（`up 0 min`，`muse-camtest` 已 not-found） | `cam-start.sh`：`frames=308 真帧=308 黑帧=0 零帧=0`、首张真帧 0.4s、退码 0；另一次 09:47 为 `371/370/1/0` |
| `k1cam` 采集 1080p/720p 帧率、NV12 内容 | **板子** | 与官方 `v4l2_test_spacemit` 同帧像素统计一致，见 `docs/k1-mipi-camera.md` §3/§6 |
| web 端到端（采集+推理+MJPEG） | **板子** 2026-11-14 09:48~10:03 | `/stats`: `cam.fps 15.33 / lum 11.3 / black_s 0 / restarts 0`；`/video` 3s 抓到 19 帧；`/snapshot.jpg` 拉回 PC 目视确认（`docs/web-live-720p.jpg`） |
| 模组装反的 180° 纠偏 | **板子** 10:14 | `ROTATE=180 ./run-web.sh`，`/stats` 里 `rotate:180`，截图方向正常 |
| 退出收尾不留 D 状态采集进程 | **板子** | `kill -TERM <web>` 后 `pgrep -x k1cam` 为空；反例（父进程被 kill 不管子进程）实测会把 k1cam 钉成 D，只能靠重拉 pipeline 或重启板子解开 |
| float yolo11n-seg CPU 968 ms / EP 1060 ms、检出的 5 个目标 | **板子** | `yolo_seg.py --prefer spacemit`，两后端输出逐目标相同 |
| `server.py` 全接口（/health /stats /video /snapshot.jpg /config） | 本机(aarch64, ORT 1.32)冒烟 + 板上跑过 | `/stats` 有 5 个目标、`/video` 5 s 出流、`POST /config {"conf":0.5}` 后低分框掉出 |
| int8 QDQ 模型精度（只量 Conv / 全图+排除 Concat） | **仅 PC**（ORT CPU EP，bus.jpg） | `0.89/0.86/0.85/0.81/0.36` vs float `0.89/0.88/0.85/0.82/0.37` |
| int8 各形态能否进 AIPU、耗时 | **板子** 2026-11-14 08:4x | Placement 直方图 + 端到端耗时，6 个模型矩阵见 `docs/k1-aipu-onnxruntime.md` §2.1（`SPACEMIT_EP_DUMP_SUBGRAPHS` 这条判据被证伪，别再用） |
| 官方 int8 13 输出解码正确性 | **板子** | bus.jpg：bus 0.87 + person 0.87/0.85/0.82；单帧 infer 418 ms（float 1371 ms） |
| EP 崩溃矩阵（线程数 x 内容 x 第几次崩） | **板子** 2026-11-14 09:5x~11:3x | `docs/k1-aipu-onnxruntime.md` §7（8 线程 4 轮都崩在 i=74~76；web 并发下 32s 崩） |
| 看门狗自愈 | **板子** 11:36 | q 模型崩后 `logs/web-supervise.log`: `web 退出（存活 32s，累计第 1 次）` → 2s 后第 2 次启动、`/stats` 恢复出数 |
| `import onnxruntime` 与 `spacemit_ort` 的顺序 | **板子** 10:4x | 反序 + int8 → EP 报 DefaultLogger 未注册并回退 CPU（`/stats` 的耗时 x10） |
| `setup-board.sh` 全流程 | 分步在板上手工验证过，脚本整体串跑待复验 | 各步命令与板上实测一一对应 |
| 「有没有用到智算核」这条判据本身 | **板子** 2026-11-14 13:42 | `./epcheck/run-epcheck.sh`：float 卷积类 CPU 101 个/EP 0 个；官方 int8 卷积类 EP 75 个/CPU 1 个 |

## 目录结构

```
apps/yolo11seg-web/
├── setup-board.sh      # 板端一键：apt 依赖 → 编 k1cam → 配相机 → 下模型/降 opset → 单帧自检
├── cam-start.sh        # 点亮相机：清理多余 cam-test + 用像素统计校验"真的出图了"（不再需要常驻服务）
├── run-web.sh          # 启动 web（处理 LD_PRELOAD，参数用环境变量覆盖）
├── supervise-web.sh    # 看门狗：web 退出/卡死就重开（量化模型走 EP 必崩时靠它兜底）
├── export_onnx.py      # PC 端：下载官方 yolo11n-seg.onnx 并降到板端能加载的 opset
├── cam/k1cam.c         # V4L2(m2m)+dma-buf 采集器，往 stdout 吐 NV12 裸帧
├── web/server.py       # Flask 后端：采集线程 + 推理线程(跳帧不排队) + MJPEG
├── web/yolo_seg.py     # 预处理/后处理(NMS、mask 合成)/绘制，可单独当命令行自测工具
├── web/index.html      # 前端：img 显示 /video，侧栏显示 stats、调 conf/掩码透明度
├── quant/quant_qdq.py  # PC 端 int8 QDQ 量化（走智算核的唯一路）
└── epcheck/            # 板端一条命令出"卷积落在哪个 EP"的账单（判断有没有吃到智算核）
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
ROTATE=180 sudo ./setup-board.sh --service   # 要开机自启时只装 web 一个 unit（相机不需要服务，见下）
```

`MUSE_SUDO_PASS='xxx' ./setup-board.sh` 可以免交互输 sudo 密码；密码不要写进脚本或仓库。
`--service` 故意**不装 cam-test 的 systemd 服务**：驱动会在 `open()` 时自己拉一发，再起一个
就是两发抢 pipeline → 恒黑帧（`docs/k1-mipi-camera.md` §2.1）。脚本会顺手把历史遗留的
`muse-camtest.service` 禁用删除。

### 4. 启动 + 看

```bash
./cam-start.sh        # 显示 [ok] 出图正常: 100% 为真帧 之后再往下
ROTATE=180 ./run-web.sh    # OV5647 模组装反（画面上下颠倒）时加 ROTATE=180
# 其它：PORT=8080 CONF=0.3 ./run-web.sh   STILL=/tmp/bus.jpg ./run-web.sh
```

浏览器 `http://<板子IP>:8000`。接口：`/video`(MJPEG)、`/snapshot.jpg`、`/stats`、
`/config`(GET/POST 调 conf/iou/mask 透明度)、`/health`。

## 可调参数

`run-web.sh` 的环境变量：`PORT` `MODEL` `CONF` `WIDTH`(+`HEIGHT`) `PREFER`(auto|spacemit|cpu)
`STILL`(静态图) `VIDEO`(mp4) `ROTATE`(0/90/180/270) `EP_THREADS`(SpaceMIT EP 线程数，默认 4)。
等价的 `server.py` 参数见 `--help`
（另有 `--cam --dev --cam-log --iou --mask-mode --quality --host --full-range`）。
采集端 ioctl 等 cam-test 注册格式的超时用 `K1CAM_WAIT`（秒，默认 20）。

`supervise-web.sh` 另有 `LOG`（默认 `logs/web-supervise.log`）与 `STALL_KILL`（秒，默认 120；
进程活着但 `infer.runs` 不再增长就主动杀掉重开，0=关）。

> 板上 `/tmp` 是 tmpfs，**重启就空**：`STILL=/tmp/bus.jpg` 重启后会报
> `can't open/read file`（实测踩过）。测试图放 `~/work/pics/` 这种持久目录。

## 排障速查（按现象）

| 现象 | 先查 | 处置 |
|---|---|---|
| `S_FMT capture: Resource temporarily unavailable` 后退出 | 是否已有别的进程占着设备：`pgrep -x k1cam` | 设备是独占的。停掉另一个采集端；本仓库的 k1cam 已自带 20s EAGAIN 重试 |
| 一帧都没有、`ps` 里也没有 cam-test | `cam-start.sh` 会打这条提示 | 内核 open() 自拉的那发握手失败；`cam-start.sh` 会在"窗口内确实 0 帧且 cam-test=0"时补拉一发（一次性，不是常驻循环） |
| 帧一直在走但画面纯黑、`/stats` 里 `cam.lum≈0.5` | `ps -eo pid,ppid,etime,comm \| grep cam-test` | 有多发 cam-test 抢 pipeline。设备空闲时清到只剩 1 发；**正在推流时绝对不要 `pkill cam-test`**（会把采集端钉成 D 状态，只能重启板子） |
| 采集进程 `kill -9` 杀不掉、状态 D | `ps -eo pid,stat,comm \| grep k1cam` | 先 `sudo setsid nohup /usr/bin/cam-test /root/svivi_cam1.json &` 恢复出帧再杀；不行就 reboot |
| 画面上下颠倒 | 模组装反，subdev 无翻转控件 | `ROTATE=180 ./run-web.sh` |
| 推理只有 1 fps 左右 | `/stats` 的 `model.providers` + `docs/k1-aipu-onnxruntime.md` | float 模型不吃 AIPU 是预期行为；要提速只能官方 int8，但它对线程数配置很敏感，看 §"想上智算核" |
| web 跑着跑着整个进程消失、`/stats` 变 HTTP=000、日志里没有 traceback | `dmesg` 也没有 segfault 记录，`logs/web-supervise.log` 会记"存活 Ns" | 是 SpaceMIT EP 在 int8 模型上 SIGSEGV（Python 层抓不到）。用 `supervise-web.sh` 兜底，或换回 float 模型 |
| `/stats` 里 `infer.fps` 远低于 `1/ms.total` | `ms.wait`（等采集帧的毫秒数）与 `cam.fps` | wait 大 → 卡在采集侧（`docs/k1-mipi-camera.md`）；wait≈0 但 fps 低 → 卡在推理线程自身的调度 |

## 想上智算核（0.6 fps → 2.4 fps，长跑要靠线程数配平）

先明确一句：**当前默认的 float yolo11n-seg 没有用到 K1 的智算核（AIPU/NPU）**，
101 个卷积类算子全在 RISC-V CPU 上；要用核只能换官方工具链产的 int8 模型。
板子上跑 `./epcheck/run-epcheck.sh` 就能自己出这张账单（判据与踩坑见
`docs/k1-aipu-onnxruntime.md` §1.2）。

板上实测过 6 个模型（`docs/k1-aipu-onnxruntime.md` §2.1），结论很硬：
**只有官方工具链（PPQ / ai_toolkit）产出的 int8 图才会真正落到 AIPU**；
自己用 onnxruntime 量化器产的 QDQ（哪怕 `--format qoperator` 改写成 100 个 QLinearConv）
全部留在 CPU，还比 float 慢 2~3 倍。所以这条路只对"用官方工具重量化"有效：

```bash
# 官方 int8 yolov8n-seg.q.onnx：解码器已支持（13 输出 / DFL，见 yolo_seg.py 的 _classify_outputs）
MODEL=models/yolov8n-seg.q.onnx PREFER=spacemit ./supervise-web.sh            # 单帧 ~408ms
```

启动日志里必须出现这一行，否则线程没配平、几十秒就会崩：

```
[info] intra_op_num_threads=4 (ep=True quant=True ep_threads=4)
```

- 精度核对（板上 bus.jpg）：bus 0.87 / person 0.87 / 0.85 / 0.82，与 float 的
  0.89/0.88/0.85/0.82 同级，int8 量化噪声最大 0.06。
- **代价：崩溃由线程数配平决定**。`SPACEMIT_EP_INTRA_THREAD_NUM` 设 6/8 时"跑到第 75 次
  推理"必 SIGSEGV；设 4 而 ORT 线程池留在 8 时 web 里 20~94s 崩一次（5 min 6 次）；
  **两者都取 4 → 32 min / 4559 次 0 崩**。崩溃发生在 `sess.run` 内部，没有 Python 异常、
  RSS 不涨（不是泄漏），native 栈落在 `libspacemit_ep.so.1.2.2` + ORT
  `ThreadPoolTempl::WorkerLoop`（`docs/k1-aipu-onnxruntime.md` §7.1）。
  **结论：要速度就用 int8（+看门狗），但它不是 yolo11 且没证明根治，默认仍是 float yolo11。**
- 想让它"崩了也能继续演示"就挂看门狗：
  `MODEL=models/yolov8n-seg.q.onnx PREFER=spacemit ./supervise-web.sh`（`EP_THREADS` 保持默认 4，
  别改成 6/8：那一档实测崩溃率反而更高，见 docs §7.1）。
- 为什么 import 顺序是 `onnxruntime` 先、`spacemit_ort` 后：反序时 int8 模型会让 EP 报
  `Attempt to use DefaultLogger but none has been registered` 然后**静默回退 CPU**
  （比 float 还慢 10 倍）。`make_session()` 里已按正确顺序写死，别调换。

自产 QDQ 的两种配方（只量 Conv / 全图 QDQ）区别和坑写在 `quant/quant_qdq.py` 文件头，
精度在 PC 上验过（`0.89/0.88/0.85/0.79/0.36` vs float `0.89/0.88/0.85/0.82/0.37`），
但**别指望它提速**。任何量化模型上摄像头前务必先跑
`python3 web/yolo_seg.py --model <量化模型> --image ~/work/pics/bus.jpg` 对精度。

## 遇到问题先看这三篇

| 症状 | 去处 |
|---|---|
| 装不上包 / EP 库找不到 | `docs/k1-env-setup.md` |
| 采到的帧全黑、全噪声、只出 1~2 帧、重启后没画面 | `docs/k1-mipi-camera.md` |
| 模型加载失败、provider 里没有 SpaceMIT、EP 不提速 | `docs/k1-aipu-onnxruntime.md` |
