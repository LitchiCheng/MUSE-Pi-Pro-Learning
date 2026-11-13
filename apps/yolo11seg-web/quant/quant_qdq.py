#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 PC 上把 YOLO11-seg ONNX 量化成 K1 AIPU 能接管的 int8 QDQ 模型。

为什么必须量化（板上实测，详见 docs/k1-aipu-onnxruntime.md）：
  SpaceMIT EP 不会接管 float 的 Conv —— 连官方 aiLib 里的 yolo11n.onnx 也是整图回 CPU；
  只在权重侧套 DequantizeLinear 也不够，**激活也要是 QDQ**，图才会被编进智算核。
  参考点：量化后的 yolov8n-seg 在 EP 上 204ms，float yolo11n-seg 在 CPU 上 ~968ms。

两条已验证可用的配方（bus.jpg 上的精度对比，float 基准 0.89/0.88/0.85/0.82/0.37）：
  A. 只量 Conv（默认，精度最好）：0.89/0.86/0.85/0.81/0.36
       python3 quant_qdq.py ../models/yolo11n-seg_opset17.onnx yolo11n-seg_qdq_conv.onnx
  B. 全图 QDQ（EP 覆盖率更高，精度略降）：0.78/0.78/0.78/0.78/0.38
       python3 quant_qdq.py ../models/yolo11n-seg_opset17.onnx yolo11n-seg_qdq_full.onnx \
           --ops Conv,MatMul,Add,Mul,Sub,Div,Concat,Resize,Sigmoid,Softmax,MaxPool \
           --exclude /model.23/Concat_6
     --exclude 不能省：output0 由 Concat_6 拼出，box 坐标(0~637)、cls 概率(0~1)、
     mask coef(0~2.5) 值域跨三个数量级，量化这**一个**张量会把 cls 概率全部压成 0，
     模型"能跑但一个目标都检不出来"。配方 A 不量 Concat 所以没这个问题。

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
        quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        calibration_providers=["CPUExecutionProvider"],
        nodes_to_exclude=[x for x in args.exclude.split(",") if x] or None,
        extra_options={"PercentileRange": [args.obs, 100 - args.obs],
                       "OpTypesToExclude": ["ConvTranspose", "Sigmoid", "Softmax", "MaxPool"]},
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
