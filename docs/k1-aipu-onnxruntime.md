# K1 上用 ONNXRuntime + SpaceMIT EP 跑 YOLO：加速边界到底在哪

对象：MUSE Pi Pro（SpacemiT K1，riscv64）+ `python3-spacemit-ort` + `onnxruntime 1.18.1`。
模型：ultralytics 官方 `yolo11n-seg.onnx`（float）及其 int8 QDQ 变体。

TL;DR：**float 模型在 K1 上永远跑 CPU，智算核只吃 int8 QDQ。**
把 640x640 的 float yolo11n-seg 降到低 opset、开 EP、加线程，全都无效；
换成量化模型才能进 AIPU（实测 yolov8n-seg 量化版 204 ms/帧）。

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

**但 `get_providers()` 里有 SpaceMIT ≠ 算子在智算核上跑**（EP 只做子图划分）。
最硬的判据是导出子图：

```python
provider_options = [{"SPACEMIT_EP_DUMP_SUBGRAPHS": "1"}, {}]
```

工作目录会出现 `SpaceMITExecutionProvider_SpineSubgraph_*.onnx`：
- 有 float yolo11n-seg 的 0 个/极小子图 → Conv 全在 CPU；
- int8 QDQ 模型里 Conv 出现在子图内 → 真的进了 AIPU。
（`SPACEMIT_EP_DEBUG_PROFILE="demo"` 可以进一步导出每段耗时。）

## 2. 实测：float 模型不吃 AIPU

`/tmp/bus.jpg`，conf=0.25，同一份 `yolo11n-seg_opset17.onnx`，板上跑：

| 后端 | 平均耗时 | 检出 |
|---|---|---|
| CPUExecutionProvider | **967.8 ms** | bus 0.89 / person 0.88 / 0.85 / 0.82 / 0.37 |
| SpaceMITExecutionProvider | **1059.9 ms** | 完全同上 |

耗时反而更高、结果一模一样 → EP 只是把 CPU 计算多包了一层。旁证：
官方 `ai_tool/aiLib/yolo11n.onnx`（他们自己给的 float 模型）在 EP 下同样整图回 CPU。
另外试过 opset 14/15/17/18 四个版本，**都不改变这一结论**。

## 3. 要进 AIPU：整图 int8 QDQ

### 3.1 什么格式算"对"

```
float 输入 → QuantizeLinear → DequantizeLinear → Conv(int8 权重) → Q → DQ → 后续
```

即**激活和权重都要量化**。只把权重换成 int8 + DequantizeLinear（激活保持 float）不够，
Conv 依然留在 CPU。反例：`QLinearConv` 这种"整算子量化"写法，K1 的 EP 直接建不出来：

```
create_operator 'SpaceMITExecutionProvider_QLinearConv_72'(QLinearConv_qc8) returned 4
```

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

## 6. 抓官方文档正文的技巧

`spacemit.com/community/document/...` 页面是前端渲染的，抓不到正文。正文其实是静态 md：

```
https://cdn-resource.spacemit.com/ai/compute_stack/ai_compute_stack/onnxruntime.md
https://cdn-resource.spacemit.com/ai/compute_stack/ai_compute_stack/onnxruntime_ep_ops.md
https://cdn-resource.spacemit.com/ai/compute_stack/ai_compute_stack/model_zoo.md
```

把 nodepath 的最后一段换成文件名即可；不存在时返回 OSS 的 `NoSuchKey` XML（用这个判断路径写错）。
