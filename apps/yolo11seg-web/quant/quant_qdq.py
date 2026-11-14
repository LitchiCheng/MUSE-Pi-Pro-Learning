#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 PC 上把 YOLO11-seg ONNX 量化成 K1 智算核(AIPU)能接管的 int8 模型。

板上实测（docs/k1-aipu-onnxruntime.md 有 kernel Placement 直方图作为证据）：
  * float yolo11n-seg：97 个 Conv 全在 CPU，EP 只捡走 Mul/Sigmoid/Add 这类逐元素算子 → 不提速；
  * 官方 int8 yolov8n-seg.q.onnx：75 个 **QLinearConv** + 66 个 QLinearSwish 在 EP 上，
    163 ms/帧（同一模型强制 CPU 是 1371 ms）→ 这才是 AIPU 吃的格式；
  * --format qdq + **--act uint8**（ORT 的默认激活类型）在 K1 上**不会被 EP 接管**：
    ORT 的图优化会把 Q/DQ+Conv 折成 `QLinearConv_token_*` 落在 CPU，
    还额外背了 388 DequantizeLinear + 273 Transpose + 258 QuantizeLinear，
    实测比 float 还慢（EP 2868 ms / CPU 1664 ms vs float CPU 968 ms）。
  * 上面这条**不是 QDQ 形态的锅，是激活类型的锅**：EP 的 int8 kernel 只注册了
    `tensor(int8)` 激活（二进制里的判据 `we have FLOAT|INT8, but got ...`）。
    同一张图改成 **--act int8** 以后，100 个卷积里 **99 个直接落到 SpaceMIT EP**，
    干净单帧 421 ms（float 1162 ms），精度 0.89/0.86/0.83/0.81/0.38 对比 float
    0.89/0.88/0.85/0.82/0.37 —— 账单见 ../../../docs/k1-aipu-onnxruntime.md §3.4。
  → 想吃 EP 的 int8 kernel：`--format qdq --act int8`（默认推荐，见 run-quant.sh）。
    `--format qoperator` 是当年为验证"是不是算子名字的锅"做的对照组，结论是**不是**，
    100 个 QLinearConv 照样全在 CPU。

已验证的精度基准（板上 bus.jpg，float 基准 0.89/0.88/0.85/0.82/0.37）：
  A. --format qdq --ops Conv --act uint8 ：0.89/0.86/0.85/0.81/0.36  精度好，但不提速（卷积在 CPU）
  A'.--format qdq --ops Conv --act int8  ：0.89/0.86/0.83/0.81/0.38  **99 个卷积上 EP，2.8x 提速**
  B. --format qdq 全图 + --exclude 输出 Concat：0.78 x4 / 0.38
  C. --format qoperator（对照组）         ：100 个 QLinearConv 仍在 CPU，3067 ms —— 形态不是判据

⚠️ 无论哪种格式，**都要 --exclude 产出 output0 的那个 Concat**（配方 A 因为只量 Conv 才躲过）：
   output0 = Concat(box 0~637, cls 概率 0~1, mask coef 0~2.5)，一个张量跨三个数量级，
   量化它会把 cls 概率全压成 0 —— 模型"能跑但一个目标都检不出来"，且不报任何错。

依赖：pip install onnx onnxruntime numpy opencv-python-headless（不需要 torch）
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import onnx


def letterbox(img, size=640):
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def load_corpus(paths, size=640, n=20):
    import cv2
    xs = []
    real = [p for p in paths if os.path.isfile(p)]
    for p in real:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        for dy in (-160, 0, 160):
            for dx in (-160, 0, 160):
                tile = np.roll(np.roll(bgr, dy, 0), dx, 1)
                xs.append(letterbox(tile, size))
    rng = np.random.default_rng(0)
    while len(xs) < n:      # 真实图不够就用结构图补，保证每个通道都覆盖到动态范围
        im = rng.integers(0, 256, (size, size, 3), np.uint8)
        im = cv2.GaussianBlur(im, (5, 5), 0) if len(xs) % 2 else im
        xs.append(im)
    xs = (xs * ((n + len(real) * 9) // max(len(xs), 1)))[:max(n, len(real) * 9)]
    arr = np.stack(xs)[:max(n, 1)]
    return (arr.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)


class NumpyReader:
    def __init__(self, name, data):
        self.name, self.data, self.i = name, data, 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        d = {self.name: self.data[self.i:self.i + 1]}
        self.i += 1
        return d

    def rewind(self):
        self.i = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="float ONNX（建议已降到 opset<=21）")
    ap.add_argument("dst")
    ap.add_argument("--images", default="bus.jpg", help="标定图，逗号分隔（可用 glob）；越接近实际场景越好")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--obs", type=int, default=10, help="percentile 裁剪分位数")
    ap.add_argument("--ops", default="Conv", help="要量化的算子白名单，逗号分隔；只量 Conv 最稳")
    ap.add_argument("--exclude", default="", help="不做量化的节点名，逗号分隔（如产出 output0 的 Concat：它的输入张量值域跨 3 个数量级，量化会把 cls 概率压成 0）")
    ap.add_argument("--format", default="qdq", choices=["qdq", "qoperator"],
                    help="qdq=QuantizeLinear/DequantizeLinear 对；qoperator=直接产出 QLinearConv")
    ap.add_argument("--act", default="uint8", choices=["uint8", "int8"],
                    help="激活的量化数据类型。K1 的 EP 里写死了 FLOAT|INT8 "
                         "(`strings libspacemit_ep.so | grep 'we have FLOAT'`)，"
                         "uint8 激活会被判成不支持 → 想吃智算核就用 --act int8")
    args = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    sess = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    paths = [p for g in args.images.split(",") for p in glob.glob(g)]
    data = load_corpus(paths, sess.get_inputs()[0].shape[-1], args.n)
    print("calibration data:", data.shape)

    pre = args.dst + ".pre.onnx"
    try:
        quant_pre_process(args.src, pre, skip_symbolic_shape=True, auto_manage=False)
        src = pre
    except Exception as e:
        print("pre_process skipped:", repr(e)); src = args.src

    quantize_static(
        src, args.dst, NumpyReader(iname, data),
        op_types_to_quantize=args.ops.split(","),
        quant_format=QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator,
        per_channel=True, reduce_range=False,
        activation_type=QuantType.QUInt8 if args.act == "uint8" else QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        calibration_providers=["CPUExecutionProvider"],
        nodes_to_exclude=[x for x in args.exclude.split(",") if x] or None,
        extra_options={"PercentileRange": [args.obs, 100 - args.obs],
                       "OpTypesToExclude": ["ConvTranspose", "Sigmoid", "Softmax", "MaxPool"],
                       **({} if args.format == "qdq" else {"MakeDynamicQuantizer": False})},
    )
    m = onnx.load(args.dst)
    from collections import Counter
    c = Counter(n.op_type for n in m.graph.node)
    print("written %s  nodes=%d  %s" % (args.dst, len(m.graph.node), dict(c)))
    print("opset", [(o.domain, o.version) for o in m.opset_import])
    for i in m.graph.input:
        t = i.type.tensor_type
        print("IN", i.name, [d.dim_value for d in t.shape.dim], t.elem_type)


if __name__ == "__main__":
    sys.exit(main())
