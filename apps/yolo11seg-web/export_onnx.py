#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 PC 上准备 K1 能加载的 YOLO11-seg ONNX（下载官方 float 模型 + 降 opset 版本号）。

为什么需要降 opset：
  ultralytics 官方 yolo11n-seg.onnx 由新版 torch 导出，实测 opset=22 / ir_version=10 / 355 节点；
  板子上 python3-spacemit-ort(1.2.2) 依赖的 onnxruntime 是 1.18.1，它的算子 schema
  表覆盖不到这么高的 opset，建会话直接抛错（详见 docs/k1-aipu-onnxruntime.md 的复现记录）。
  YOLO 图里用到的算子（Conv/Relu/Add/Mul/Sigmoid/Split/Concat/Resize/MatMul/Softmax/…）
  在 opset 13~22 之间语义一致，所以只要把 opset 版本号改低就能被 1.18 接受，
  不需要真正的逐算子版本转换（onnx.version_converter 反而会因为没有 Split 18→x 的
  adapter 而报错，见 --strict 的说明）。

⚠️ 常见误解：降 opset 只是"让模型能在板子上打开"，跟 AIPU 加速没有任何关系。
  SpaceMIT EP 不会因为 opset 变低就接管 Conv —— float 权重的 Conv 一律留在 CPU。
  要吃到智算核，必须做整图 int8 QDQ 量化，见 quant/quant_qdq.py。

用法：
    python3 export_onnx.py --opset 17               # 下载官方模型并生成 yolo11n-seg_opset17.onnx
    python3 export_onnx.py --input x.onnx --opset 17 --opset 14
    python3 export_onnx.py --opset 11 --strict      # 降到 <13 时按语义逐算子改写
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("已存在 %s (%d bytes)" % (dst, os.path.getsize(dst)))
        return dst
    print("下载 %s" % url)
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", "0"))
        got = 0
        while True:
            blk = r.read(1 << 20)
            if not blk:
                break
            f.write(blk)
            got += len(blk)
            print("\r  %d/%d MB" % (got >> 20, total >> 20), end="")
    print()
    os.replace(tmp, dst)
    return dst


def _fix_split(g, node, opset):
    """Split: opset>=13 用第 2 个输入给切分长度，<=12 必须是 attribute。"""
    import onnx
    from onnx import helper
    if opset > 12 or node.op_type != "Split" or len(node.input) != 2:
        return
    name = node.input[1]
    val = None
    for init in g.initializer:
        if init.name == name:
            val = onnx.numpy_helper.to_array(init)
            break
    if val is None:
        for n in g.node:
            if n.op_type == "Constant" and name in n.output:
                for a in n.attribute:
                    if a.name == "value":
                        val = onnx.numpy_helper.to_array(a.t)
    if val is None:
        raise RuntimeError("Split 节点 %s 的 split 输入不是常量，无法降到 opset<=12" % node.name)
    node.attribute.append(helper.make_attribute("split", [int(v) for v in val.flatten()]))
    del node.input[1]


def _fix_softmax(node, opset):
    """Softmax: opset>=13 用 axis，<=12 用 axes（列表）。"""
    from onnx import helper
    if opset > 12 or node.op_type != "Softmax":
        return
    axis = [a.i for a in node.attribute if a.name == "axis"]
    if axis:
        del node.attribute[:]
        node.attribute.append(helper.make_attribute("axes", axis))


def downgrade(model, opset, strict=False):
    """把模型降到 opset。

    strict=False：只改 opset 版本号（opset>=13 时足够，实测 yolo11n-seg 到 14/15/17/18 都能加载）。
    strict=True ：额外按语义改写 Split / Softmax，用于目标 opset<=12；
                  不要走 onnx.version_converter —— 它在 22→17 这一步会因缺少
                  "Adapter From Version 18 for Split" 直接抛 RuntimeError。
    """
    from onnx import helper

    g = model.graph
    if strict:
        for n in list(g.node):
            _fix_split(g, n, opset)
            _fix_softmax(n, opset)
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", opset))
    model.ir_version = 9            # onnxruntime 1.18 的读取端更稳
    return model


def convert(src, dst, opset, strict=False):
    import onnx
    m = onnx.load(src)
    old = m.opset_import[0].version
    print("源: %s  opset=%s  ir_version=%s  nodes=%d" %
          (src, old, m.ir_version, len(m.graph.node)))
    m = downgrade(m, opset, strict)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print("写出 %s  (%.1f MB, sha256=%s…)" %
          (dst, os.path.getsize(dst) / 1e6, sha256(dst)[:16]))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="已有的 ONNX 文件；不给则下载官方 yolo11n-seg.onnx")
    ap.add_argument("--opset", type=int, action="append", default=None,
                    help="目标 opset，可多次指定（默认 17，对齐板端 onnxruntime 1.18.1）")
    ap.add_argument("--strict", action="store_true", help="降到 opset<=12 时逐算子改写语义")
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    opsets = args.opset or [17]
    os.makedirs(args.outdir, exist_ok=True)
    src = args.input or download(URL, os.path.join(args.outdir, "yolo11n-seg.onnx"))
    stem = os.path.splitext(os.path.basename(src))[0]
    for n in opsets:
        convert(src, os.path.join(args.outdir, "%s_opset%d.onnx" % (stem, n)), n, args.strict)
    print("完成。scp 到板子的 models/ 下；要 AIPU 加速还得再走 quant/quant_qdq.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
