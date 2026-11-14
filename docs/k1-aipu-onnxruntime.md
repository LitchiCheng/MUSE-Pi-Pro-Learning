# K1 上用 ONNXRuntime + SpaceMIT EP 跑 YOLO：加速边界到底在哪

对象：MUSE Pi Pro（SpacemiT K1，riscv64）+ `python3-spacemit-ort` + `onnxruntime 1.18.1`。
模型：ultralytics 官方 `yolo11n-seg.onnx`（float）及其 int8 QDQ 变体。

TL;DR：**float 模型在 K1 上永远跑 CPU；能不能落到 SpaceMIT EP 的 int8 kernel，只看激活张量
是不是 `tensor(int8)`**（§3.4 修正版；早先写的"只吃官方工具链产出的图"是被 ORT 默认的
`QUInt8` 激活误导出来的错结论）。把 640x640 的 float yolo11n-seg 降到低 opset、开 EP、加线程，
全都无效；换成 int8 激活的量化图才快：自产 `--act int8` 421 ms/帧、官方 yolov8n-seg.q.onnx
214 ms/帧（同一份 float 是 1162 ms）。

但 libspacemit_ep 1.2.2 跑 int8 会在连续推理时随机 SIGSEGV，Python 侧完全抓不到（第 7 节）。
**崩不崩和线程数配置强相关**：`SPACEMIT_EP_INTRA_THREAD_NUM=4` 且把 ORT 的
`intra_op_num_threads` 也收到 4（两个不一致时最脆），实测 web 常驻连跑
**32 min / 4559 次推理 0 崩**（§7.1）；老配置（EP 4 线程 + ORT 8 线程）是 **5 min 崩 6 次**。
代码已按这条自动收线程（`yolo_seg._tie_intra()`）。即便如此板端默认仍是
"float yolo11n-seg + CPU + 0.6 fps"——int8 那份是 v8 不是 v11，而且长跑样本只有一块板一次，
要快就显式换量化模型并挂上看门狗：自产 int8 yolo11（§3.4，1.1 fps）或官方 int8 v8（§7.1，2.4 fps）。

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

能落地的判据是**逐算子的节点归属清单**：开最啰嗦的日志建一次 session，ORT 会打
`VerifyEachNodeIsAssignedToAnEp`，先按 EP 给一行 "Node(s) placed on [X]. Number of nodes: N"，
随后逐行列 `OpName (节点名)`。仓库里已经做成一条命令（`apps/yolo11seg-web/epcheck/`）：

```python
so = ort.SessionOptions()
so.log_severity_level = 0        # VERBOSE，节点清单只在这个级别输出
so.log_verbosity_level = 3
sess = ort.InferenceSession(m, sess_options=so,
                            providers=["SpaceMITExecutionProvider", "CPUExecutionProvider"])
```

```bash
# 在板子上跑，默认对比 float yolo11 与官方 int8 v8 两份；也可以指名别的模型
cd ~/work/yolo11seg-web && ./epcheck/run-epcheck.sh
#   ep.py 建 session（带 LD_PRELOAD）把 VERBOSE 日志落盘，place.py 按 EP 分段的清单做直方图
```

`place.py` 输出的关键就是**卷积落在哪个 EP 名下**：全在 `CPUExecutionProvider` 就等于没加速，
必须和耗时数字对得上（下面 2.1 的表）。2026-11-14 13:42 在同一块板上的实际账单：

```
== yolo11n-seg_opset17.onnx          (float，本项目默认)
   [SpaceMITExecutionProvider] 自报 191 个节点： Mul x88, Sigmoid x87, Add x13, Sub x2, Div x1
   [CPUExecutionProvider]      自报 158 个节点： Conv x97, Concat x24, Split x12, Reshape x11,
                                                MaxPool x3, FusedConv x3, Softmax x2, Resize x2,
                                                FusedMatMul x2, Transpose x1, ConvTranspose x1
   Conv 类: CPU 101 个 -> 卷积在 CPU，等于没吃到智算核
== yolov8n-seg.q.onnx                (官方 int8)
   [SpaceMITExecutionProvider] 自报 164 个节点： QLinearConv x75, QLinearSwish x66,
                                                DequantizeLinear x15, QLinearSigmoid x3,
                                                MaxPool x3, QuantizeLinear x2
   [CPUExecutionProvider]      自报  55 个节点： Transpose x22, QLinearConcat x13, Split x8,
                                                QLinearAdd x6, ReduceSum x3, Resize x2, ConvTranspose x1
   Conv 类: EP 75 个 / CPU 1 个 -> 卷积真的落在智算核的 int8 kernel 上
```

> **一条订正**（比"float 在 EP 上是白包一层"更准确）：float 模型并不是一个节点都没进 EP ——
> 191 个逐元素算子（Mul/Sigmoid/Add/Sub/Div）确实归了 SpaceMIT EP，但**全部 101 个卷积类算子
> 留在 CPU**，而卷积占了 yolo 的几乎全部算力，所以耗时只增不减（1060 ms vs 968 ms）。
> 判断时**只看卷积**，别被 EP 的节点数唬住。
> 另外这里出现的 `QLinearConv` 是 **EP 内部的 kernel 名**，不等于图上写了这个算子
> （官方 q 模型的图层面是 251 QuantizeLinear + 252 DequantizeLinear 的 QDQ 形态，见 §2.1 末尾）。

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
| `yolo11n-seg_opset17.onnx`（float，本项目用） | 1060 ms | 968 ms | **卷积类 101 个全在 CPU**（EP 只接走 191 个逐元素算子） | 不吃 AIPU：没有卷积进核 |
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

一句话结论（2026-11-14 上午版，**本轮已被部分推翻，见 §3.4**）：K1 的 EP 只认官方工具链
（PPQ / xquant）产出的 int8 形态。自己用 onnxruntime 量化器产的 QDQ/QLinearConv 图全留在 CPU 上。

> 这个结论当时对所有测过的模型都成立，所以看起来像"格式派"问题。真正的开关其实是**激活的
> 数据类型**：我们的量化脚本用了 ORT 默认的 `activation_type=QUInt8`，而这个 EP 的 int8
> kernel 只注册了 `tensor(int8)` 的激活。换成 `QInt8` 以后，**同一套 onnxruntime 量化器、
> 同一张图**，100 个卷积里 99 个直接落到 SpaceMIT EP（§3.4 有账单）。跟"必须用官方工具"无关。

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

也就是说决定成败的**不是**"QDQ 还是 QLinearConv"这种表面形态 —— 用
`--format qoperator` 把 QDQ 全图改写成 100 个 QLinearConv，Placement 仍然全是 CPU。

> 当时从这一行往下推的结论是"那就是官方工具链的 proprietary pattern 了"，**这个推断错了**：
> 两组实验的激活都是 ORT 默认的 `uint8`，共同变量找错了。真正的开关见 §3.4
> （`--act int8`，同一套 onnxruntime 量化器就能让 99/101 个卷积落 EP）。
> 形态实验本身仍然有效：它排除了"改个算子名字就能上核"这条路。

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

### 3.4 真正的开关：激活必须是 **INT8**，不是 ORT 默认的 UINT8（自产模型上核实测）

EP 二进制里把收图规矩写得很直白（`libspacemit_ep.so.1.2.2`，`strings` 可查）：

```
' in SpaceMIT EP, we have FLOAT|INT8, but got          ← 类型判据只有 FLOAT 和 INT8
per-channel uint8 quantization isn't supported
of scale quantization parameters for UINT8 tensor
only support 0 as zero point for per-channel quantization
N11onnxruntime21QDQ14NodeGroupSelectorE / SpaceMITQuantizeNodeCapability
```

也就是说 `QDQNodeGroupSelector` 在 capability 判定阶段按**张量元素类型**筛卷积：激活写成
`tensor(uint8)` 直接不合格 → 该 Conv 不进 EP → 图优化阶段 CPU EP 把 Q/DQ 折成
`QLinearConv_token_*` 自己吃掉（于是 Placement 里看到"100 个 QLinearConv 全在 CPU"）。
`quant/quant_qdq.py` 之前用的正是 ORT 的默认 `QuantType.QUInt8`；现在加了 `--act {uint8,int8}`。

同一份 `yolo11n-seg_opset17.onnx`，只差这一个参数，板上的账单（`./epcheck/run-epcheck.sh`，
2026-11-14 19:0x）：

| 模型 | SpaceMIT EP 拿到的算子 | Conv 类落点 | 干净单帧 | bus.jpg 分数 |
|---|---|---|---|---|
| float（参照） | Mul×88, Sigmoid×87, Add×13, Sub×2, Div×1（191 个，无卷积） | **CPU 101 / EP 0** | 1162 ms | 0.89/0.88/0.85/0.82/0.37 |
| 自产 QDQ `--act uint8` | Mul×88, Sigmoid×87, Add×15（193 个，无卷积） | **CPU 101 / EP 0** | — | 0.87/0.86/0.83/0.79/0.38 |
| **自产 QDQ `--act int8`** | **QLinearConv×99**, DequantizeLinear×218, QuantizeLinear×88, Mul×88, Sigmoid×87（598 个） | **CPU 2 / EP 99** | **421 ms** | 0.89/0.86/0.83/0.81/0.38 |
| 官方 int8 `yolov8n-seg.q.onnx`（参照） | QLinearConv×75, QLinearSwish×66, MaxPool×3（164 个） | CPU 1 / EP 75 | 214 ms | 0.87/0.87/0.85/0.82 |

- 上表的"干净单帧"是**停掉 web 后**用 `sess.run` 计时的 `wall`（8 次均值，随机噪声输入）；
  和 web 并存时同一发会是 1245~1547 ms，别看串了。
- 精度：**int8 自产量化相对 float 最大掉 0.03、平均掉 0.01**（0.89/0.86/0.83/0.81/0.38 vs
  0.89/0.88/0.85/0.82/0.37），可以直接用。
- 留在 CPU 的那 2 个卷积类算子是 1 个 `ConvTranspose`（neck 的 upsample 分支，EP 的
  ConvTranspose 只收 float/float16）和 1 个被 `nodes_to_exclude` 保精度而排除的 Conv。

一条命令复现（PC 侧，任意架构，不需要 torch）：

```bash
python3 quant/quant_qdq.py models/yolo11n-seg_opset17.onnx out/y11seg_int8.onnx \
    --images "corpus/*.jpg" --n 36 --ops Conv --exclude /model.23/Concat_6 \
    --format qdq --act int8
```

参数里三个都是必要的，少一个就退化成上表的 uint8 行：`--act int8`（激活有符号）、
`--ops Conv`（只量卷积，逐元素算子留给 EP 的 float kernel，精度最稳）、
`--exclude /model.23/Concat_6`（§3.3 那条坑）。

校准集：本轮用的是 **9 张这块板上摄像头拍的实拍图**（`--images corpus/*.jpg`）。
`load_corpus()` 会把每张图做 9 个平移块（±160 像素的 roll）再 letterbox 到 640，
所以实际喂进校准器的是 81 条样本 —— `--n 36` 是**下限**而不是上限（真实块多于 36 就全用），
只有真实图不够时才会拿随机噪声/高斯模糊图补位。换别的场景部署时，把这批图换成你实际会遇到的画面，
比调任何参数都重要。

### 3.5 官方的量化工具是 xquant（不是 PPQ 裸用）

官方文档 `software/SDK/ros/k1/07_Advanced_development/7.1_Model_Quantization.md`
（版本 2025-09-22）给的正规路子：

```bash
pip install xquant --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
python3 -m xquant --config yolov8/data/yolov8_xquant_config.json     # 产出 yolov8n.q.onnx + .q_report.md
```

`xquant` 是 SpacemiT 基于 **PPQ v0.6.6+** 做的封装（最新 2.0.4，`py3-none-any` 纯 Python 包，
依赖 torch/torchvision）。官方另一篇 `4.4 YOLOv8 训练部署全流程` 明确写了
**「量化操作需在 x86 平台进行」**，且要求用 `truncate_var_names` 在后处理节点前截断 ——
和 §3.3 那条"别量化输出 Concat"是同一件事的两种说法。

把 wheel 拆开能直接看到它对"什么形态能上核"的官方答案（`xquant/optimizer/refine.py`）：

```python
class AsymmetricaUnsignlAlignSign(QuantizationOptimizationPass):   # "XQuant AsymmetricalAlignS8 Pass"
    if config.policy.has_property(ASYMMETRICAL) and config.quant_min == 0:
        config.quant_min = -(2 ** (num_of_bits - 1)); config.quant_max = 2 ** (num_of_bits - 1) - 1
        config.offset = config.offset + config.quant_min
```

即 **xquant 专门有一个 pass 把"无符号非对称"的激活改成"有符号 int8"**，
`XQuantizer` 的 `target_platform` 也写死 `TargetPlatform.XQUANT_INT8`、
Conv 用 `SYMMETRICAL + PER_CHANNEL`。这从上游工具的角度反向坐实了 §3.4 的判据：
官方 int8 模型里 251 个 QuantizeLinear 的 zero_point **全是 INT8**（本轮用
`quant/onnxlite.py` 直接解析 protobuf 数出来的），而不是 ORT 量化器默认给的那套 uint8。

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

`spacemit.com/community/document/info?nodepath=...` 这个 URL 现在**只会返回 1223 字节的 Vue
壳子**（`GET` 任何 `?lang=zh&nodepath=*.md` 变体都是同一份 HTML，200 + text/html）。
正文不在页面里，走的是"文档树 API + CDN 静态 md"两步（本轮从 `assets/js/index-*.js` 里挖出来的）：

```bash
# 1) 全量文档树：每个叶子带 namePath / path / url(GitHub) / storgeAddress(CDN)
curl -s "https://www.spacemit.com/api-server/document/get-doc-tree?language=zh" -o tree.json
#    注意 base 必须是 /api-server，直接拼 /api 或 /document 会回那个 HTML 壳

# 2) 用树里的 storgeAddress 取正文（path 里有中文，要 percent-encode）
curl -sL "$(python3 -c "import json;d=json.load(open('tree.json'));..." )"
```

`grep` 树里的 `path` 找文档比翻导航快得多。本轮定位到的几篇（`tree.json` 里的字段值）：

| 正文 | storgeAddress 去掉域名的前缀 |
|---|---|
| ONNXRuntime EP 加速算子（**每个算子的 capability 限制，含 QDQ**） | `/ai/docs-ai/zh/compute_stack/ai_compute_stack/onnxruntime_ep_ops.md` |
| SpacemiT-ONNXRuntime（**全部 ProviderOption 的官方说明**） | `/ai/docs-ai/zh/compute_stack/ai_compute_stack/onnxruntime.md` |
| K1 模型量化开发（xquant） | `/software/SDK/ros/docs-ros/zh/k1/07_Advanced_development/7.1_Model_Quantization.md` |
| K1 YOLOv8 训练部署全流程 | `/software/SDK/ros/docs-ros/zh/k1/04_Model_deployment/4.4_Training_and_Deployment_Pipeline.md` |

`url` 字段那侧是 `github.com/spacemit-com/docs-ai|docs-ros|docs-courses|docs-events`，
`raw.githubusercontent.com/...` 也能拿到（本轮实测 `docs-ai` 的 FAQ 200、其余仓库的同名文件 404，
仓库名和目录层级和 `path` 不完全对得上，所以**优先用 `storgeAddress`**）。

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

---

## 8. 先校准用词：K1 上没有独立 NPU，"上核"= 上 SpacemiT 优化的 int8 CPU kernel

`docs/k1-env-setup.md` 和本篇前面都顺着官方宣传写了"智算核 / AIPU / NPU"。板子上查完硬件，
**这块板上并不存在一个独立的神经网络加速单元**，三条互相独立的证据：

1. **设备树里没有 NPU 节点**。`/proc/device-tree/` 全量遍历，和算力沾边的只有
   `tcm@0xd8000000`（TCM，片上紧耦合内存，驱动 `d8000000.tcm`）、`pdma@d4000000`、
   `adma@C0883800`；`find /proc/device-tree -name compatible | xargs grep -l npu\|aipu`
   是**空**，`dmesg | grep -iE 'aipu|npu'` 也是空。`lscpu`：8× `SpacemiT(R) X60`，
   614.4~1600 MHz，没有第二个计算簇。
2. **`/dev/udma`、`/dev/tcm` 被打开不等于被计算**。`libspacemit_ep.so.1.2.2` 里确实写着
   `/dev/udma`、`/dev/tcm`、`tcm_malloc`，`strace` 也看到会话建起来时
   `openat("/dev/tcm", O_RDWR|O_SYNC) = 5` / `openat("/dev/udma", ...) = 6`；
   但整发推理里这两个 fd 上**一个设备 ioctl 都没有**（只有 184 条 ioctl，全是 `TCGETS`/
   `TIOCGWINSZ` 这类 isatty 探测）。而且 `fuser /dev/udma` 显示采集进程 `k1cam` 也持有它 ——
   它是 SoC 通用 DMA 引擎，不是 NPU 的门。
3. **线程数扫描证明计算在 CPU 核上**。同一份 int8 模型（§3.4 的 `--act int8` 产物）单实例、
   只改 `EP_THREADS`：

   | `EP_THREADS` | `sess.run` wall | 进程 CPU 时间（`getrusage`） | CPU/wall |
   |---|---|---|---|
   | 1 | 621 ms | 620 ms | **1.00** |
   | 2 | 606 ms | 1211 ms | **2.00** |
   | 4 | 416 ms | 1655 ms | **3.97** |
   | 8 | 1143 ms | 4525 ms | 3.96（过订阅，反而慢） |

   **CPU 时间恰好 = wall × 线程数**，即每颗核都被这段计算实打实占满；真要有硬件核在算，
   CPU 时间应该远低于 wall。float 模型同测是 1162 ms wall / 8395 ms CPU（7.22 核）。

所以正确的说法是：**int8 的收益来自"每帧要做的 CPU 工作量变小"**。按 CPU 时间算最干净 ——
float 8395 ms/帧 vs int8(4 线程) 1655 ms/帧，**量化把单位帧的计算量砍到 1/5.1**；
单核口径更明显：int8 单核 620 ms 就能跑完一帧，float 摊到 8 核反而要 1049 ms/核。
这正是 SpacemiT 官方对 K1 的定名 —— 竞赛赛题那篇就叫
**《基于 K1 **AI CPU** 大模型部署落地》**，正文写的是"增加了对进迭时空 **AI 扩展指令**的支持"，
全程不出现 NPU/AIPU 字样。

> 结论到文档里的落点：以后看到"EP 75 个 QLinearConv 在核上"这类描述，按本节理解为
> "在 SpaceMIT EP 的 int8 CPU kernel 上（跑在 X60 核 + 向量/AI 扩展指令，4 线程）"。
> `epcheck` 的判据本身不变 —— 它回答的是"这些算子有没有交给 SpacemiT 的优化 kernel"，
> 这仍然是决定快慢的关键。

## 附：本轮（2026-11-14）改动清单

| 文件 | 改了什么 |
|---|---|
| `apps/yolo11seg-web/quant/quant_qdq.py` | 加 `--act {uint8,int8}`（默认 uint8 保持向后兼容），文件头写清"想吃 EP 必须 int8" |
| `apps/yolo11seg-web/quant/onnxlite.py` | 新增：不依赖 onnx/torch 的 protobuf 级模型结构速览（`§2.3` 那些数字就是它数出来的） |
| `apps/yolo11seg-web/quant/run-quant.sh` | 新增：PC 侧一条命令出 int8 模型 + 上板 epcheck 的提示 |
| `apps/yolo11seg-web/epcheck/` | 新增（上一轮）：一条命令出"卷积落在哪个 EP"的账单 |

