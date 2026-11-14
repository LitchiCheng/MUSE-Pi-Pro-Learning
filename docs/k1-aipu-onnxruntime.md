# K1 上用 ONNXRuntime + SpaceMIT EP 跑 YOLO：加速边界到底在哪

对象：MUSE Pi Pro（SpacemiT K1，riscv64）+ `python3-spacemit-ort` + `onnxruntime 1.18.1`。
模型：ultralytics 官方 `yolo11n-seg.onnx`（float）及其 int8 QDQ 变体。

TL;DR：**float 模型在 K1 上永远跑 CPU，智算核只吃官方工具链产出的 int8 图。**
把 640x640 的 float yolo11n-seg 降到低 opset、开 EP、加线程，全都无效；
换成量化模型才能进 AIPU（实测 yolov8n-seg 量化版 sess.run 155 ms，端到端 408 ms/帧）。

但 libspacemit_ep 1.2.2 跑 int8 会在连续推理时随机 SIGSEGV，Python 侧完全抓不到（第 7 节）。
**崩不崩和线程数配置强相关**：`SPACEMIT_EP_INTRA_THREAD_NUM=4` 且把 ORT 的
`intra_op_num_threads` 也收到 4（两个不一致时最脆），实测 web 常驻连跑
**32 min / 4559 次推理 0 崩**（§7.1）；老配置（EP 4 线程 + ORT 8 线程）是 **5 min 崩 6 次**。
代码已按这条自动收线程（`yolo_seg._tie_intra()`）。即便如此板端默认仍是
"float yolo11n-seg + CPU + 0.6 fps"——int8 那份是 v8 不是 v11，而且长跑样本只有一块板一次，
要 2.4 fps 就照 §7 末尾显式换模型并挂上看门狗。

---

## 1. 装与不装的区别

```bash
sudo apt-get install -y spacemit-onnxruntime python3-spacemit-ort python3-opencv libopencv-dev
```

（这行来自官方 onnxruntime 文档；`python3-spacemit-ort` 依赖 `onnxruntime`(1.18.1)。）

### 1.1 `LD_PRELOAD` 不是可选项

直接 `python3 server.py` 会看到 EP 在日志里抱怨

```
Attempt to use DefaultLogger but none has been registered.
```

然后**静默把整张图交给 CPU**，一点提示都没有。必须：

```bash
export LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1
```

`web/server.py` 里的 `ensure_ld_preload()` 会在检测不到该变量时自动 re-exec 自己，
所以正常用 `run-web.sh` 启动不会踩到；排查时用 `MUSE_NO_LD_PRELOAD=1` 可以禁用这个行为。

### 1.2 判断"有没有真的用上 EP"

```python
import spacemit_ort, onnxruntime as ort
print(ort.get_available_providers())     # 不 import spacemit_ort 时列表里没有 SpaceMIT
sess = ort.InferenceSession(m, providers=["SpaceMITExecutionProvider", "CPUExecutionProvider"])
print(sess.get_providers())              # 只有这里能看到实际生效顺序
```

**但 `get_providers()` 里有 SpaceMIT ≠ 算子在智算核上跑**（EP 只做子图划分，
float 模型也会给你列出这个 provider）。

> **订正**：网上/文档里流传的 `SPACEMIT_EP_DUMP_SUBGRAPHS=1` 在 K1 这版 runtime 上
> **不能用** —— 连确定吃到加速的官方 `yolov8n-seg.q.onnx`（实测 163ms vs CPU 1371ms）
> 也是 `spine_subgraphs=0`、不落任何 `*_SpineSubgraph_*.onnx`。按它下"没进 AIPU"的结论会
> 把所有模型都误判成没加速。`SPACEMIT_EP_DEBUG_PROFILE` 同理没给出可用信息。

能落地的判据是**逐算子的 Placement 日志**：开最啰嗦的日志跑一次，统计
`Placement:'XxxExecutionProvider'` 出现次数，按 (op 类型, EP) 做直方图。

```python
so = ort.SessionOptions()
so.log_severity_level = 0        # VERBOSE
so.log_verbosity_level = 3
sess = ort.InferenceSession(m, sess_options=so,
                            providers=["SpaceMITExecutionProvider", "CPUExecutionProvider"])
```

```bash
# 板端 /tmp/epctl/{ep.py,place.py}：ep.py 跑一次并把 stderr 全存下来，
# place.py 正则提 (op, Placement) 直方图
python3 ep.py 模型.onnx > run.log 2>&1 && python3 place.py run.log
```

`place.py` 输出的关键就是 `Conv`/`QLinearConv` 落在哪：全在 `CPUExecutionProvider`
就等于没加速，跟耗时数字必须对得上（下面 2.1 的表）。

## 2. 实测：float 模型不吃 AIPU

`/tmp/bus.jpg`，conf=0.25，同一份 `yolo11n-seg_opset17.onnx`，板上跑：

| 后端 | 平均耗时 | 检出 |
|---|---|---|
| CPUExecutionProvider | **967.8 ms** | bus 0.89 / person 0.88 / 0.85 / 0.82 / 0.37 |
| SpaceMITExecutionProvider | **1059.9 ms** | 完全同上 |

耗时反而更高、结果一模一样 → EP 只是把 CPU 计算多包了一层。旁证：
官方 `ai_tool/aiLib/yolo11n.onnx`（他们自己给的 float 模型）在 EP 下同样整图回 CPU。
另外试过 opset 14/15/17/18 四个版本，**都不改变这一结论**。

### 2.1 六个模型的完整实测矩阵（同一块板、640x640、bus.jpg）

| 模型 | EP 下耗时 | CPU 下耗时 | 关键算子 Placement | 结论 |
|---|---|---|---|---|
| `yolo11n-seg_opset17.onnx`（float，本项目用） | 1060 ms | 968 ms | **97 个 Conv 全在 CPU** | 不吃 AIPU，EP 只是白包一层 |
| 官方 `yolov8n-seg.q.onnx`（int8） | **163 ms** | 1371 ms | 75 `QLinearConv` + 66 `QLinearSwish` 在 EP | **8.4x，目前唯一能实时的那条路** |
| 官方 `yolo11n.q.onnx`（int8, det） | LOAD_FAIL | — | — | `SpaceMITExecutionProvider_QLinearConv_72` 返回 4 |
| `resnet18.q.onnx`（int8） | 19 ms | — | 在 EP | 参照点，证明板子本身没问题 |
| 我们的整图 QDQ | LOAD_FAIL | — | — | `EP does not support Compile`（见 3.x） |
| 我们的 QDQ→`--format qoperator` | 3067 ms | — | 100 个 `QLinearConv` **仍在 CPU** | 光是把 QDQ 改写成 QLinearConv 不会上智算核 |

（`yolo11n-seg_qdq_conv`（只量 Conv）端到端 3176 ms，比 float 还慢 2~3 倍 —— 量化参数
在 CPU 侧来回 Q/DQ 的开销全暴露出来了。）

> **算子名分两层，别混**（本轮在板上 `grep -a -o` 二进制复核）：官方 `yolov8n-seg.q.onnx`
> 的**图里**一个 `QLinearConv` 都没有，只有 251 个 `QuantizeLinear` + 252 个
> `DequantizeLinear` + 69 个 `Sigmoid` —— 它是 **QDQ 形态**的图；上表"75 个 QLinearConv 在 EP"
> 是 **Placement 日志里 EP 内部 kernel** 的名字（EP 自己把 QDQ 折成 int8 kernel）。
> 反过来，我们自己 `--format qoperator` 产的图**层面**就是 QLinearConv 却全落 CPU。
> 所以"图上写什么算子"不是判据，session 建起来之后 `providers`/Placement 才是。
> 代码里判"是不是 int8 模型"（只为决定线程池）因此找的是 `QuantizeLinear` 子串，
> 见 `web/yolo_seg.py` 的 `QUANT_OP_TAGS`。

一句话结论：**K1 的 EP 只认官方工具链（PPQ / ai_toolkit）产出的 int8 形态**，包括自研的
融合算子（`QLinearSwish` 这种）和配套的 Transpose 布局处理。自己用 onnxruntime 量化器产的
QDQ/QLinearConv 图，无论怎么写都留在 CPU 上。

## 3. 要进 AIPU：整图 int8 QDQ

### 3.1 什么格式算"对"

```
float 输入 → QuantizeLinear → DequantizeLinear → Conv(int8 权重) → Q → DQ → 后续
```

即**激活和权重都要量化**。只把权重换成 int8 + DequantizeLinear（激活保持 float）不够，
Conv 依然留在 CPU。

关于 `QLinearConv`（"整算子量化"写法）要说准确：**它本身不是禁词** —— 官方
`yolov8n-seg.q.onnx` 里的 75 个 `QLinearConv` 就实实在在落在 EP 上。但**我们自己图里**的
QLinearConv 建不出来 / 建出来也在 CPU：

```
create_operator 'SpaceMITExecutionProvider_QLinearConv_72'(QLinearConv_qc8) returned 4   # 官方 yolo11n.q.onnx 也这样
```

也就是说决定成败的不是"QDQ 还是 QLinearConv"，而是**这张图是不是官方工具链按它认识的
pattern 产出的**。这一点在本轮用 `--format qoperator` 把 QDQ 全图改写成 100 个
QLinearConv 之后被反证：算子类型对了，Placement 仍然全是 CPU。

参考数据（同一块板、640x640）：`resnet18.q.onnx` 19 ms、`yolov8n-seg.q.onnx` 204 ms，
说明格式对了提速是实打实的。官方 modelzoo 给的口径：resnet18 int8 12.99 ms、
yolov8n int8@640 73.74 ms（K1 条目里没有 yolo11）。

### 3.2 用 ONNX Runtime 自带量化器就够（不需要 torch）

`apps/yolo11seg-web/quant/quant_qdq.py`（本次开发用，纯 numpy 标定）：

```bash
python3 quant_qdq.py yolo11n-seg_opset17.onnx yolo11n-seg_qdq_conv.onnx --images bus.jpg --ops Conv
```

### 3.3 一个新坑：量化最后的 Concat 会让模型"检不出任何东西"

全图 QDQ 时若把 `output0` 的那个 Concat 也量化了，结果**不报错但 cls 全 0**：
`output0 = Concat(box 坐标 0~637, cls 概率 0~1, mask coef 0~2.5)`，
一个张量的值域跨三个数量级，uint8 量化步长直接被 box 撑到 2.5，cls 概率全被截成 0。

| 配方 | bus.jpg 结果（float 基准 0.89/0.88/0.85/0.82/0.37） |
|---|---|
| 只量 Conv（`--ops Conv`） | 0.89 / 0.86 / 0.85 / 0.81 / 0.36 ✅ |
| 全图 QDQ，量化 Concat_6 | **一个目标都没有**（cls max = 0.0）❌ |
| 全图 QDQ + `--exclude /model.23/Concat_6` | 0.78 / 0.78 / 0.78 / 0.78 / 0.38（可用，精度略降） |

结论：优先"只量 Conv"，需要更高覆盖率时**必须把输出侧那个跨值域的 Concat 排除掉**。
量化模型上板前先在 PC 上跑一遍 `web/yolo_seg.py --image bus.jpg` 对精度，别直接上摄像头。

## 4. opset：只是"能不能打开"，和加速无关

官方 `yolo11n-seg.onnx`（torch 新版 dynamo 导出）是 **opset 22 / ir_version 10**，
板端 onnxruntime 1.18.1 建会话直接失败。在任意 aarch64 机器上装同版本就能复现：

```
pip install onnxruntime==1.18.1
FAIL: Load model ... ValidateOpsetForDomain ... ONNX Runtime only *guarantees* support for
      models stamped with official released onnx opset versions. Opset 22 is under development...
      Current official support for domain ai.onnx is till opset 21.
```

**所以降到 ≤21 就行**，本仓库默认 17。两种降法：

- 直接改版本号 `m.opset_import[0].version = 17`：YOLO 用到的算子
  （Conv/Relu/Add/Mul/Sigmoid/Concat/Resize/MatMul/Split(≥13)/Softmax(≥13)/…）
  在 13~22 语义一致，实测 14/15/17/18 都能加载并得到相同结果 → 这就是 `setup-board.sh` 的做法。
- `onnx.version_converter` 逐级转：**不要用**，在 22→17 会
  `No Adapter From Version $18 for Split` 直接抛错（除非目标 ≥13 且模型里没有别的版本敏感算子）。
  真要降到 ≤12，得手工把 `Split` 的第 2 输入搬回 attribute、`Softmax` 的 `axis` 换成 `axes`，
  `export_onnx.py --strict` 就是干这个的。

> 曾经有个错误结论流传："opset 太高导致 EP 回退 CPU"。不成立 —— 回退 CPU 的原因是
> 第 2 节（float Conv），跟 opset 无关。

## 5. 后处理侧的对齐要点（float/int8 通用）

`web/yolo_seg.py` 有两处必须这么写，否则"模型明明是对的但检不出东西"：

1. 预处理要 `letterbox` + BGR→RGB + **乘 1/255**（漏掉归一化会让分数全部乱套）。
2. ultralytics 的 seg 导出里 `output0[4:84]` 的 cls **已经过 Sigmoid**，
   不要再套一层；自己写稳健版 Sigmoid 时要先 `np.clip(x,-30,30)`，
   因为 `np.where(x>=0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))` 两个分支都会被求值，
   `exp(+大值)` 会出 inf/inf = NaN（NaN 会让后面的阈值筛选全灭）。
3. 输出布局：`output0 [1,116,8400]`（4 box，输入尺度 cx,cy,w,h + 80 cls + 32 mask coef）、
   `output1 [1,32,160,160]`（protos）。mask 用"只在检测框覆盖的 protos 子块上线性组合"
   的 crop 写法，比全图合成快一个量级（`--mask-mode full` 可切回标准做法核对精度）。
4. 官方 int8 模型是**另一种布局**：不做层融合，检测头按层摊成 13 个输出
   （3 尺度 x 的 box[1,64,H,W] DFL logits / cls[1,80,H,W] 已 sigmoid / 系数[1,32,H,W]，
   再加 3 个 [1,1,H,W] 的 cls-max（忽略）+ proto[1,32,160,160]）。`_classify_outputs()`
   按 shape 认布局，`_decode_raw()` 解码成同一个 `[1,116,8400]`。DFL 的通道排布是
   **head-major**：`reshape(4,16,H,W)`，head = left/top/right/bottom、bin 在最后一维
   （写成 `reshape(16,4,..)` 会解出乱七八糟的框）。解码要点：
   ```
   d = softmax(dist.reshape(4,16,H,W), axis=2) @ arange(16)   # 每个 head 的期望距离(单位: stride)
   w = (d[l] + d[r] + 1) * stride;  cx = (gx + 0.5 + (d[r] - d[l]) / 2) * stride   # 同理 h/cy
   ```
   另外那 3 个 [1,1,H,W] 的 cls-max 在 int8 图上是**量化过的概率**（实测与 float cls 的
   max 平均绝对误差 1.2e-4、最大 0.06，即 int8 噪声），可以直接忽略不读。

## 6. 抓官方文档正文的技巧

`spacemit.com/community/document/...` 页面是前端渲染的，抓不到正文。正文其实是静态 md：

```
https://cdn-resource.spacemit.com/ai/compute_stack/ai_compute_stack/onnxruntime.md
https://cdn-resource.spacemit.com/ai/compute_stack/ai_compute_stack/onnxruntime_ep_ops.md
https://cdn-resource.spacemit.com/ai/compute_stack/ai_compute_stack/model_zoo.md
```

把 nodepath 的最后一段换成文件名即可；不存在时返回 OSS 的 `NoSuchKey` XML（用这个判断路径写错）。

---

## 7. int8 + EP 能跑，但长跑必崩（SIGSEGV 实测矩阵）

`yolov8n-seg.q.onnx`（官方 int8，13 输出，结构见 §2.1）单次推理确实快：
`sess.run` 155 ms、`_decode_raw` 54 ms、`postprocess` 44 ms、`draw` 87 ms，端到端 ~420 ms。
但只要连着跑就会在 `sess.run` 里段错误。**线程数**指
`SPACEMIT_EP_INTRA_THREAD_NUM`（代码里的 `EP_THREADS`）：

> 下表是早期"单进程死循环"那批数据的汇总；把两个线程变量摊开固定时长对比的是 §7.1。

| EP 线程 | 跑的活 | 结果 |
|---|---|---|
| 8 | bus.jpg + 每 5 帧换随机噪声图，单进程死循环 | **4 轮全部崩在 i=74 / 75 / 76 / 32** |
| 6 | 同上 | 崩在 i=75 |
| 4 | 同上（死循环） | 一次 400 帧不崩；**换到 web（带采集线程并发）里 i≈26~50 就崩** |
| 2 / 1 | 同上（死循环） | 600 帧不崩（th=1 端到端 405 ms/帧） |
| 4 | **web 常驻**（ROTATE=180 + 摄像头 15 fps），ORT 线程池仍是当时的默认 8 | 300s 内崩 6 次（20~94s 一次）；历史最长一次 2035 s / 423 次 |
| 8 + `enable_cpu_mem_arena=False` 或 `enable_mem_pattern=False` | 死循环 | 仍崩（i=75 / 99）→ 不是 SessionOptions 的内存策略 |
| 8 + 每 60 次推理重建 session | 死循环 | 仍崩 → 不是"跑久了才坏"，是计数到了某个深度 |
| 8 + 只 fetch 3/13 个输出 | 死循环 | 跑到 i=50 未崩（样本不足，不能当结论） |

崩溃现场的特征（这些排除了几条常见解释）：
- 栈顶永远是 `onnxruntime_inference_collection.py:220 run`，开了 `PYTHONFAULTHANDLER`
  也只有这一帧 —— **没有 Python 异常、没有可捕获的错误码**，进程直接没了。
- `dmesg` 里没有 segfault / OOM 记录；RSS 稳定在 239 MB 不涨（不是内存泄漏）。
- 崩完之后 `atexit` 不执行，但因为 k1cam 的 stdout 是父进程的管道，管道一关它自己收到
  EPIPE 退出，所以**不会**留下 D 状态进程霸占 `/dev/video50`（`ps -eo stat,args | grep k1cam`
  为空，重启 web 后 8s 内恢复出图）。这一条只在"web 直接父进程被 SEGV 打死"时成立；
  如果是 `kill -9` 了 python 而 k1cam 还活着，仍会变 D（见 `docs/k1-mipi-camera.md`）。

### 7.1 有一个配置明显更耐跑：ORT 线程池 == EP 线程数（且 <=4）

同一块板、同一份 `yolov8n-seg.q.onnx`、都在 web 里常驻（摄像头 15 fps + `ROTATE=180`），
每档跑 300s，只看"进程消失了几次"。取数方式是"每档单实例串行"，但**要打折看**：
本轮末尾发现两批实验脚本的看门狗会并存（`pkill` 后 1s 就重开），并发实例既能把崩溃率
放大、又有把内核 `misc_open` 打出 Oops 的风险（`docs/k1-mipi-camera.md` §2.5），
所以矩阵里"崩 N 次"是上界；干净的单实例长跑见下面 §7.2。

| `EP_THREADS`(=SPACEMIT_EP_INTRA_THREAD_NUM) | `intra_op_num_threads`(ORT 线程池) | 300s 内崩溃 |
|---|---|---|
| 4 | 8（本项目原默认） | **6 次**（20~94s 一次，均值 ~45s） |
| 4 | 4 | **0 次** |
| 8 | 8 | **9 次** |
| 4 | 1 | **0 次** |

gdb 抓到的崩溃现场支持"线程 id 越界"这个猜测（**这是推断，不是结论**）：崩的那一帧是
EP 在自己的回调里被 ORT 的线程池调起来的 ——

```
Thread 15 "python3" received signal SIGSEGV, Segmentation fault.
#0  ??? () at .../spacemit_ort/libspacemit_ep.so.1.2.2      # 符号被 strip，只能给库内偏移
#1  ??? () at .../libspacemit_ep.so.1.2.2
#2  ??? () at .../libspacemit_ep.so.1.2.2
#3  ??? () at /usr/lib/libonnxruntime.so.1.18.1
#4  ??? () at /usr/lib/libonnxruntime.so.1.18.1
#5  std::_Function_handler<void (), onnxruntime::concurrency::ThreadPoolTempl<Env>::
    ScheduleOnPreferredWorkers(...)::{lambda()#1}>::_M_invoke()   at libonnxruntime.so.1.18.1
#6  onnxruntime::concurrency::ThreadPoolTempl<Env>::WorkerLoop(int)  at libonnxruntime.so.1.18.1
#7/#8  ... / start_thread / clone3
```
同一时刻另一个 ORT 线程停在 `ThreadPoolTempl::RunInParallel(...)` 上，也就是
**EP 是用 ORT 的 parallel-for 跑分片的**。如果它按 `worker id` 去索引一个长度只有
`SPACEMIT_EP_INTRA_THREAD_NUM` 的数组，而 ORT 起了 8 个 worker，id 4~7 就是越界写。
这解释了 `ep=4/intra=8` 为什么比 `ep=4/intra=4` 脆得多；但解释不了 `ep=8/intra=8`
（对齐了照样 9 次），所以 EP 里至少还有第二种崩法。复现代码：
`gdb -q -batch -ex "set pagination off" -ex run -ex "bt 25" -ex "info threads"
--args python3 web/server.py --model models/yolov8n-seg.q.onnx --prefer spacemit ...`
（要带 `LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1`，否则静默回退 CPU，永远不崩）。

### 7.2 单实例长跑复验：`ep=4 / intra=4` 连跑 32 min 不崩

**长跑复验（同一份 q 模型、单实例 + `supervise-web.sh`，12:22:13 起连跑 32 min）：
0 次崩溃、0 次看门狗重启**，`infer.runs` 累计 4559、`infer.fps` 稳定 2.43
（`ms` = infer 283 / post 90 / pre 34 / total 408 / wait 12），`cam.fps` 15.6、
`cam.restarts` 0、`black_s` 0，这 32 min 里 `dmesg` 没多出任何 Oops。
同一块板同一份模型，把 intra 改回 8 就是上表的"5 min 崩 6 次"。
所以 **`ep=4 / intra=4` 是目前唯一可演示的智算核配置**；但它只是一次 32 min 的样本，
不能说"修好了"，看门狗照挂。

> 补一条"为什么矩阵必须单实例串行跑"的实测：上面 float 的 web 还在跑（8 核几乎占满）时，
> 另开一发 q 模型做静态图推理，日志照旧是 `intra_op_num_threads=4 (ep=True quant=True ep_threads=4)`、
> providers 也确实带 SpaceMIT EP，但 `infer` 从 283 ms 涨到 797~1055 ms（5 次均值 1258 ms/帧、
> 检测框仍是 bus 0.87 / person 0.87/0.85/0.82）。也就是**耗时数字只在单实例下成立**，
> 两发 web 并存既会让 q 慢到不如 float 单跑，又有把内核打 Oops 的风险（§7.1 末尾）。

代码里已经把这条固化成默认：`make_session()` 不再固定 `intra_op_num_threads=8`，
改由 `_tie_intra()` 决定 —— **session 真的挂了 SpaceMIT EP 且图里有量化算子**
（`QuantizeLinear`/`QLinearConv`/`QLinearMatMul`/`ConvInteger`，见 `QUANT_OP_TAGS`）才把 ORT
线程池收到 `EP_THREADS`，float 模型（含"int8 但 EP 没接住"的自产 QDQ）照旧开满 8 核。
启动时会在日志里打一行 `[info] intra_op_num_threads=4 (ep=True quant=True ep_threads=4)`，
要手工对照用 `INTRA_THREADS=<n> ./run-web.sh`。

**能用的处置**：
1. 默认（本项目就是这么配的）用 float 模型：97 个 Conv 全在 CPU，等于不碰智算核，
   连跑几小时无崩，0.6 fps。要"跑通 yolo11-seg"也只能用它（官方那份 int8 是 v8）。
2. 要速度就用官方 int8 + `EP_THREADS=4`：线程池由 `_tie_intra()` 自动配平到 4，
   实测 32 min 无崩；同时**务必挂 `supervise-web.sh`**（崩了 2s 重开，实测 32s 崩 → 2s 恢复，
   页面会黑一下），并且**同一时刻只允许一个 web 实例**（并发 open 会把内核打出 Oops）。
3. 想追根因就往这几个方向试：同一进程里只跑 EP 不跑采集线程能否撑到 1000 次；
   换 libspacemit_ep 的更新版本；`gdb -batch -ex run -ex bt` 抓 native 栈（本节已抓过一份）。

> 顺带一个坑：`import onnxruntime` 必须在 `import spacemit_ort` **之前**。反序时 float 模型
> 一切正常，但 int8(QLinearConv) 模型会让 EP 打 `Attempt to use DefaultLogger but none has
> been registered` 然后**静默回退 CPU**，耗时从 163 ms 变成 ~1371 ms 且没有任何报错提示。
