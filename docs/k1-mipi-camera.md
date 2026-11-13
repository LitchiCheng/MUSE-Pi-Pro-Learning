# MUSE Pi Pro / SpacemiT K1 点亮 MIPI 相机（OV5647）—— 踩坑实录

适用：MUSE Pi Pro（SpacemiT K1，riscv64，Bianbu/ROS2_LXQT 镜像），摄像头 OV5647 排线接 CSI3。
结论先放前面：**照官方文档做完流程，仍然采不到图是常态**。官方流程缺了 3 个软件细节和
1 个运维前提，全部是"看起来像驱动坏了"的类型。下面每条都附了实测数据。

相关代码：`apps/yolo11seg-web/cam/k1cam.c`（采集器）、`apps/yolo11seg-web/cam-start.sh`（拉起 + 校验）。

## 0. 链路长什么样

```
OV5647 --(MIPI CSI3)--> ISP -->  /dev/video50  (spacemit "vivi" 虚拟相机, m2m)
                                   ↑
                    /usr/bin/cam-test <svivi.json>  常驻，负责 sensor/ISP pipeline
```

`/dev/video50` 只是一个"把 ISP 出的帧以 V4L2 capture 形式吐出来"的虚拟节点，**它自己不会
去开 sensor**。没有 cam-test 在跑时，节点存在、ioctl 全都成功、帧也"采得到"，但内容是
全 0 或上一段内存的残留 —— 这是最容易骗人的地方。

## 1. 官方流程（照做，但不够）

```bash
# 1) 探测 sensor（会占住 CSI 口约 20s）
sudo cam-test /usr/share/camera_json/csi3_camera_detect.json
#    日志里应有：detect xxx sensors in csi3: success
#    并生成 /tmp/csi3_camera_auto.json
sudo cp /tmp/csi3_camera_auto.json /root/svivi_cam1.json
# 2) 在顶层加 "use_v4l": 1     ← 开启 Virtual Camera 模式，必须带逗号
# 3) 跑 cam-test /root/svivi_cam1.json
# 4) 用官方调试工具验证出图
wget -q https://archive.spacemit.com/ros2/code/v4l2_test_spacemit.tar.gz   #  archive 可直连
tar xzf v4l2_test_spacemit.tar.gz
gcc v4l2_capture.c v4l2_main.c v4l2_output.c v4l2_stream.c v4l2_common.c -o v4l2_test_spacemit --static
./v4l2_test_spacemit --device /dev/video50 \
    --set-fmt-video width=1920,height=1080,pixelformat=NV12 \
    --verbose --stream-dmabuf --stream-loop --stream-save 10 --stream-to=test.yuv
```

官方文档里最大分辨率 1920x1080、格式只支持 NV12、内存必须是 dmabuf 这三条是真的。
> 提示：这条命令是判断"到底是驱动问题还是我的代码问题"的分水岭，先跑通它再写代码。
> 它也能在 PC(x86/aarch64) 上编译后拷进板子，省一轮板端工具链。

## 2. 四条铁律

### 2.1 cam-test 必须常驻，而且只能有 1 个实例；板子重启后没人启动它

- `camera.service`（`/etc/init.d/camera.sh`）**只做 chmod**，不负责拉 cam-test。
  也就是说：重启之后 `/dev/video50` 在、权限对、ioctl 不报错，但**读不到真帧**。
  → 我们用 `cam-start.sh`（或 `--install` 装的 `muse-camtest.service`）补这一环。
- 多开的表现：后起实例日志里
  `Bind TCP server socket failed ... error(98)`（抢 tuning server 端口 12218），
  然后 `v4l2_single_online_test: wait kernel stream on ... Resource temporarily unavailable`，
  最后自己 `finish power off` —— 它把 pipeline 关了，于是**先起来的那个也采不到帧了**。

拉起方式（不依赖终端，退出 ssh 不会被杀）：

```bash
sudo sh -c 'setsid nohup /usr/bin/cam-test /root/svivi_cam1.json > /tmp/svivi_cam1.log 2>&1 < /dev/null &'
sleep 8
```

- `/dev/video50` 同时只允许**一个**采集客户端：第二个进程 `S_FMT` 会返回
  `Resource temporarily unavailable`。web 服务里那个"采集线程重启"逻辑就是为独占这个 fd 设计的。

### 2.2 capture buffer 要用 `heap_flags=0xf0` 申请，且必须按真实大小 mmap

官方文档被注释掉的那段（`使用 /dev/dma_heap/ 申请 DMABUF，heap_flags = 0xf0`）是**关键信息**：

| 申请方式 | 实测结果 |
|---|---|
| `/dev/dma_heap/system` + flags=0 | ioctl 成功、能 mmap、能 DQBUF，但**驱动根本不往里写**：Y 全 0、UV 全 0 |
| `/dev/dma_heap/system` + flags=0xf0 | 正常出帧 |

但 0xf0 有两个额外坑（K1 内核 6.6.63）：
1. **返回的 buffer 真实大小可能小于或大于请求值**（实测请求 1MB → filesz 520192 或 4423680）。
   按请求长度 mmap 会 `EINVAL`。必须 `lseek(fd, 0, SEEK_END)` 拿真实大小再 mmap。
2. 反复 `kill -9` 采集进程之后，0xf0 会开始稳定报 `EINVAL`（CMA/page-pool 状态被搞坏），
   换 `linux,cma` 或 flags=0 也可能失败 —— **reboot 后恢复**。所以：采集进程要能被
   SIGTERM/SIGINT 正常收尾（STREAMOFF + REQBUFS(0) + munmap + close），不要动辄 kill -9。

`k1cam.c` 里 `dmabuf_alloc()` 按 `[system → linux,cma]` x `heap_flags=0xf0` 重试最多 16 轮
（**不拿 flags=0 兜底**：那种 buffer 能建立流但内容为 0，静默出黑帧比直接失败更难查；
`--hf` 可以手工改这个值做实验），并总是以 lseek 的真实大小 mmap；`-v` 会打印每个 plane 的 len/used/data_offset。

### 2.3 取帧时每个 plane 的起点是它自己的 `planes[p].data_offset`，不能累加

V4L2 多平面 buffer 里 plane1 的 `data_offset` **已经是相对 buffer 起始的绝对偏移**
（NV12 下 plane0 off=0 len=W*H，plane1 off=W*H）。若按"上一个 plane 末尾 + 本 plane offset"
累加，色度就读到 buffer 外面去了 —— 表现就是**满屏彩色噪声/花屏**，非常像 ISP 没调好。
（官方 FAQ 里"画面花屏/噪点/偏紫偏绿"那一条把人往硬件方向带，实际这里是纯软件 bug。）

### 2.4 重新 QBUF 会 EAGAIN，必须重试

fd 是非阻塞的 + m2m 队列偶发忙，`VIDIOC_QBUF` 返回 EAGAIN 是正常态。直接放弃的话，
4 个 buffer 用完就再也不会有新帧 —— 症状是"只出 1~2 帧然后卡住/退出"。

## 3. 症状 → 原因 对照（含判据）

拿一帧 NV12（W*H*3/2 字节），算 Y 的 mean/std、UV 的 mean/std：

| 现象 | Y | UV | 结论 |
|---|---|---|---|
| 全 0 | mean≈0, std≈0 | mean≈0, std≈0 | 驱动没写进你的 buffer：cam-test 没跑 / 多开 / heap_flags 不是 0xf0 |
| 彩色噪声花屏 | 抖动大 | std 远大于 10 | 读的是未初始化内存，或 plane 偏移算错（见 2.3） |
| 全黑但有细节 | std 很小(约 0.5)、max 只有个位数 | mean≈128, std≈0.3 | **链路完全正常，只是环境没光**（UV=128 是灰/黑的中性色度，正好证明帧是真的） |
| 只出 1~2 帧 | 前几帧正常 | | QBUF EAGAIN 没重试（2.4）或 buffer 泄漏，reboot 再来 |

实测样本（00:47，reboot + 单实例 cam-test 后，1920x1080，环境全黑）：

```
f0  Y mean 1.07  std 0.52  min 0  max 2 | chroma mean 127.9  std 0.3     # 官方 v4l2_test_spacemit
    同一帧经 k1cam：1920x1080 40帧/3.4s = 11.8 fps，1280x720 40帧/3.1s = 12.9 fps
```

`cam-start.sh` 把这套判据写成了脚本，退出码分别是 2(没整帧)/3(全 0)/4(噪声)/0(正常或极暗)。

## 4. 硬件与编号细节

- MUSE Pi Pro 上有两路 CSI：**22pin/4lanes** 与 **15pin/2lanes**。15pin 那一路的丝印信号名是
  `MIPI_CSI3_*`，所以 OV5647（15pin 模组）走 `csi3_camera_detect.json` 是对的。
- **排线方向**：官方 FAQ 明确 OV5647 用"铜色面（金手指朝下）"的接法，IMX219 用白色排线接法；
  接反可能永久损坏（I2C/电源错位），通电前务必对着丝印确认。CSI/DSI 排线**禁止热插拔**。
- `csiN_camera_detect.json` 里的 N 与物理口编号在部分批次上并不严格对应；
  认不出sensor时把 csi1~csi3 三个探测脚本各跑一遍再下"硬件坏了"的结论。
- 排线过度弯折会造成间歇性识别失败（差分线对形变敏感）——遇到"昨天好今天不行"先换排线。

## 5. 其他细节

- OV5647 挂在 **CSI3**（`/usr/share/camera_json/csi3_camera_detect.json`）。接 CSI1/CSI2 就换对应编号。
- sensor subdev **不暴露任何 v4l2 控件**：`v4l2-ctl -d /dev/v4l-subdev1 --list-ctrls` 输出为空，
  所以外部调不了曝光/增益，只能在 svivi json 的 `isp0` 里写死；日志会提示
  `no isp setting file, use settings from user or sensor`。白天嫌暗/夜里全黑都属于这一类限制。
- `isp0` 里写 `1920x1080 fps30`，实际出流约 **15 fps**；任何分辨率都能出（非原生分辨率由 vivi 缩放），
  所以 web 端用 1280x720 是安全的。
- 采集不需要 sudo（`/dev/video50`、`/dev/dma_heap/*` 权限已放开）；只有起 cam-test 需要 sudo。
- 板子默认 shell 是 **zsh**，远程批量命令建议 `ssh <user>@<ip> bash -s <<'EOS' ... EOS`，
  否则 `shopt`/数组/`[[ ]]` 之类会踩到 zsh 的差异。
