# MUSE Pi Pro / SpacemiT K1 点亮 MIPI 相机（OV5647）—— 踩坑实录

适用：MUSE Pi Pro（SpacemiT K1，riscv64，Bianbu/ROS2_LXQT 镜像），摄像头 OV5647 排线接 CSI3。
结论先放前面：**照官方文档做完流程，仍然采不到图是常态**。官方流程缺了 3 个软件细节和
1 个运维前提，全部是"看起来像驱动坏了"的类型。下面每条都附了实测数据。

相关代码：`apps/yolo11seg-web/cam/k1cam.c`（采集器）、`apps/yolo11seg-web/cam-start.sh`（拉起 + 校验）。

## 0. 链路长什么样

```
OV5647 --(MIPI CSI3)--> ISP -->  /dev/video50  (spacemit "vivi" 虚拟相机, m2m)
                                   ↑
                        open() 时内核 call_usermodehelper 自动拉起
                                   ↓
                    /usr/bin/cam-test <svivi.json>  负责 sensor/ISP pipeline
```

`/dev/video50` 只是一个"把 ISP 出的帧以 V4L2 capture 形式吐出来"的虚拟节点，**它自己不会
去开 sensor**。没有 cam-test 推流时，节点存在、ioctl 不报错、帧也"读得到"，但内容是全 0
或恒定的假帧 —— 这是最容易骗人的地方。

> **2026-11 订正**：本文早期版本写过"cam-test 必须常驻 / 跑约 10s 自己退出 / 需要 supervisor
> 循环或开机自启服务"。那是错的（而且按它做会稳定得到黑帧）。真实机制是 **open() 触发内核
> 自拉 cam-test**，帧流起来之后可以持续几十秒不重启（见 2.1）。

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

## 2. 五条铁律

### 2.1 `open("/dev/video50")` 会让内核自己拉起一发 cam-test；客户端要重试 EAGAIN

这是本轮最关键的一条，推翻"必须常驻/必须有 supervisor"的旧结论。三条实测证据：

1. 板上什么都没跑（`pgrep -c -x cam-test` = 0、没有自启服务）时，只要有一个进程
   `open()` 了这个节点，`ps` 里就会多出一发：
   `/bin/bash -c /usr/bin/cam-test /root/svivi_cam1.json > /tmp/svivi_cam1.log 2>&1`，
   **PPid = 2（kthreadd）** —— 这是 `call_usermodehelper` 的典型特征，不是任何服务干的。
2. cam-test 从被拉起到向内核注册格式要 1~2s，这期间客户端的 `VIDIOC_S_FMT`/`REQBUFS`/
   `QUERYBUF`/`QBUF`/`STREAMON` 会直接返回 **EAGAIN**（`Resource temporarily unavailable`）。
   官方示例代码一次 ioctl 失败就退出，所以表现为"打不开相机"。
   → `k1cam.c` 里这些 ioctl 统一走 `ioctl_wait()`：只对 EAGAIN/EBUSY/ETIMEDOUT 以 200ms
   间隔重试，默认上限 20s（`-w` 可改），其它 errno 立即失败。加上这个改动后，
   **reboot 之后零人工干预**就能出图（10:13，`uptime` 显示 up 0 min、`muse-camtest` 已
   not-found、`pgrep -c -x cam-test` = 0）：`cam-start.sh` 实测 `frames=308 真帧=308
   黑帧=0 零帧=0`、首张真帧 t=0.4s、退码 0；另一次（09:47）是 `371/370/1/0`。
3. 格式注册成功之后帧流是**持续**的：同一进程连续跑 72s，`cam-test` 数恒为 1、
   采集端 `restarts=0`、`black_s=0`，web 侧 `cam.fps=15.4`。所以不需要任何 supervisor。

由此得到两条运维规则：

- **只能有 1 发 cam-test**。2~3 发并存会互相抢 pipeline，症状是**恒定黑帧**
  （Y mean≈1.0 / std≈0.5 / max=2 / UV=128，实测多实例时帧帧如此）。上一版为了"自愈"
  留的 supervisor 循环 + 内核自拉的那发正好是两个实例 —— 黑帧就是这么制造出来的。
  `cam-start.sh` 现在只做一件事：在**确认没有采集端在推流**时把多余实例（含历史遗留的
  supervisor shell）清掉，然后交给 open() 自拉。
- **绝不要在采集端已经 open + STREAMON 之后 `pkill cam-test`**。实测帧数会立刻冻结
  （2450 之后不再涨），`k1cam` 进入 `kill -9` 也杀不掉的 **D 状态**并继续占着设备，
  之后任何进程 open 都只会 EAGAIN，只能重启板子。
  救法（本轮实测有效，比重启快）：`sudo setsid nohup /usr/bin/cam-test <json> &`
  把 pipeline 重新跑出帧，卡住的那发 k1cam 就能被 kill 掉了。
  所以 web 进程退出时必须**正常收尾**自己的 k1cam 子进程：`server.py` 里挂了
  `atexit` + SIGTERM/SIGINT 处理，实测 `kill -TERM <web>` 之后 `pgrep -x k1cam` 为空。

顺带修正 2.1 的旧说法："`/dev/video50` 同时只允许一个采集客户端"仍然成立 —— 第二个
reader 的 `S_FMT` 会一直 EAGAIN 到超时（实测 20s 打满后报
`等 20s 仍然 EAGAIN/EBUSY（cam-test 没起来？）`），所以 `cam-start.sh` 在校验前会先
`pgrep -x k1cam`，发现占用就直接报错退出而不是继续"验证"。

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

### 2.5 多个采集实例并发 `open()` 会打出 kernel Oops，之后谁都起不来（只能重启）

2026-11-14 11:5x，两批实验脚本各自拉起一套 web（3 发 `server.py` 并发 `open`），现场是：
`ps` 里三个进程全停在 `wchan = misc_open` 的 **DN** 状态，`kill -9` 无效、8000 端口不监听、
之后再没有任何进程能出图。`dmesg` 给出了原因 —— **open 路径上内核自己 Oops 了**：

```
Unable to handle kernel paging request at virtual address 0000004c00000033
Oops [#1]
CPU: 4 PID: 10782 Comm: python3 Not tainted 6.6.63 #2.2~rc3.2
Hardware name: spacemit k1-x MUSE-Pi-Pro board (DT)
 epc : misc_open+0x4a/0x132   ra : misc_open+0x38/0x132
[<ffffffff8072cb3c>] misc_open+0x4a/0x132
[<ffffffff802319e2>] chrdev_open+0x9c/0x1fa
[<ffffffff80228026>] do_dentry_open+0x184/0x382
...
[<ffffffff80fe64ea>] ret_from_exception+0x0/0x6e
---[ end trace 0000000000000000 ]---
```

同一分钟的 dmesg 里还有配套的握手错乱（**推断**：`/dev/video50` 那发由
`call_usermodehelper` 拉起的 cam-test 只认"第一个来握手的 pid"，第二个 reader 的
netlink 请求把内核里记的那个 pid 冲掉了，于是 open 永远等不到应答）：

```
vcam_wrn: fill_recv_msg_by_nlmsg_data 220: other pid:0 != 10752, don't respond netlink:17
vcam_wrn: netlink_rcv_msg 295: unexpected resv happen!
cam_not:vi (spm_vdev_stop_streaming 1197): aout0 total_frm(317) ok_frm(165)   # 165/317，丢一半
```

运维结论（比"§2.1 只能有 1 发 cam-test"更硬）：

- **`/dev/video50` 同时只允许 1 个 open 者，而且这条要靠流程保证，不是靠"第二个会 EAGAIN 退出"**：
  实测并发的第二次 open 不总是干净地 EAGAIN，会走进上面那条 Oops 分支，代价是**重启整块板子**。
- 任何做对照实验的脚本，启动 web 之前必须先 `pgrep -f "web/server[.]py"` 确认为 0；
  `pkill` 之后至少等 3s 再拉起下一发（老 web 实例的 k1cam 收尾、驱动停流都要时间）。
- 已经进了 D 状态就别救了吧：`setsid nohup sudo cam-test <json> &`（§2.1 那条救法）
  对"被 `pkill cam-test` 冻住的流"有效，对这种 **open 里 Oops 掉的任务无效**，
  只能 `sudo reboot`。reboot 要留够时间：板子上挂着 D 任务时 systemd 停服务会走满超时，
  本轮实测从下 reboot 到 ssh 可用约 **8 分钟**（不是 1 分钟）。

## 3. 症状 → 原因 对照（含判据）

拿一帧 NV12（W*H*3/2 字节），算 Y 的 mean/std、UV 的 mean/std：

| 现象 | Y | UV | 结论 |
|---|---|---|---|
| 全 0 | mean≈0, std≈0 | mean≈0, std≈0 | 驱动没写进你的 buffer：cam-test 没跑 / 多开 / heap_flags 不是 0xf0 |
| 彩色噪声花屏 | 抖动大 | std 远大于 10 | 读的是未初始化内存，或 plane 偏移算错（见 2.3） |
| 黑且**没有任何噪声纹理** | mean≈1.0, std≈0.5, max=2 | mean≈128, std≈0.3 | **假帧**：有第二发 cam-test 在抢 pipeline（见 2.1）。清到只剩 1 发立刻变真帧 |
| 黑但有噪声颗粒 | mean 个位数、std>=1 | mean≈128, std≈1~3 | 链路正常，**真的没光**。盖住/照亮镜头数值会变，假帧不会 |
| 只出 1~2 帧 | 前几帧正常 | | QBUF EAGAIN 没重试（2.4）或 buffer 泄漏，reboot 再来 |
| web 起不来、`ps` 里进程停在 `misc_open` 且杀不掉 | — | | 并发 open 打出 kernel Oops（2.5），只能 reboot |

```
# 本轮（09:47，reboot 后零干预，单实例，1280x720，室内偏暗）
frames=371 | 真帧=370 黑帧=1 零帧=0 | 首张真图 t=0.4s      -> [ok] 100% 为真帧 13.7 fps
# 同一时刻 web 侧 /stats：cam.fps 15.33, lum(Y std) 11.3, lum_mean 19.4, black_s 0, restarts 0
# 多实例对照（09:2x，supervisor + 内核自拉 = 2~3 发并存）
每帧 Y mean≈1.0 std≈0.5 max=2, UV=128                    -> 恒黑帧，与"没光"完全不同
```

> 判黑帧时 **std 比 mean 可靠**：真暗场景的 Y 也有 1~3 的噪声，假帧的 std 只有 0.5 左右。
> 光看"画面全黑"就下结论"夜里没光"是踩过两次的坑。

`cam-start.sh` 把这套判据写成了脚本，退出码 2(没整帧)/3(全 0 帧)/4(噪声或真帧占比过低)/0(正常)。

另外一条**新踩的坑**：暗场景下 `cv2.cvtColor(plane, COLOR_YUV2BGR_NV12)` 看着比原始 Y 暗得多
（原始 Y mean=19.4 → BGR 灰度 mean 只有 3.8）。不是 bug：cv2 一律按 **limited-range**
（`R=1.164*(Y-16)+…`）换算，Y=19 会被压到 4。判自己的板子是哪种量程就看 Y 的下界：
`np.percentile(Y,(0,1,50,99,100))` 实测 `[17,17,17,89,116]`，地板正好停在 16 附近
→ **本板 ISP 输出是 limited-range，cv2 的默认转换是对的，不要开 `--full-range`**。
（若哪天拿到 0..255 全量程的流，`server.py --full-range` 会对 BGR 结果套一次
`x/1.164+16` 的 LUT，等价还原成 full-range，色度误差 2%。）

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
- **模组装反 → 画面上下颠倒 180°**，而 sensor/ISP subdev 一个 v4l2 控件都不暴露
  （`v4l2-ctl -d /dev/v4l-subdev1 --list-ctrls` 输出为空），没有任何"翻转"开关可走。
  所以在采集源发布前统一旋转：`server.py --rotate 180`（0/90/180/270，
  `run-web.sh` 用 `ROTATE=180` 传）。放在发布前是为了**让推理和绘制看同一份像素**，
  否则框对、mask 错。
- 在板上用 shell 数进程有两个坑：
  `pgrep -c cam-test` 在**没匹配时也打印 0 但退码 1**，写 `$(pgrep -c -x cam-test || echo 0)`
  会拿到两行 0，`[ 0\n0 -gt 1 ]` 直接报 integer expression expected；
  `pgrep -f`/`pkill -f` 会匹配到**执行这条命令的 shell 自己**（cmdline 里含同样的字符串），
  ssh 远程执行时会把会话自己杀掉且没有任何报错。统一用 `-x 进程名`，
  非要按 cmdline 匹配就写 `pkill -f "server[.]py"`。
- 探针脚本别用 `-n 0` 往 `/tmp` 写原始 NV12：1280x720 一帧 1.38MB、15fps = 20MB/s，
  跑十分钟就是 13GB。用完立刻 `pkill -x k1cam` + 删文件（本轮就把板上 `/tmp` 撑到 2.1G）。

## 6. 采集侧性能实测（1280x720 NV12）

| 场景 | 数字 |
|---|---|
| 纯采集（`k1cam -n 0` 只读不算） | ~15.4 fps（web `cam.fps` 72s 稳定） |
| 采集 + python 侧逐帧 numpy 统计（`cam-start.sh`） | 13.2~13.7 fps |
| 1920x1080（早期实测，同一采集器） | 11.8 fps |
| web 出流（1280x720 JPEG q80，`fps_out=15`） | 约 6 帧/s（3s 抓到 19 个 boundary），瓶颈是 K1 上的 JPEG 编码，不是相机 |

`isp0` 写 30fps 也只出 15fps（见 5 节），这是 sensor/ISP 侧的上限，和软件无关。
