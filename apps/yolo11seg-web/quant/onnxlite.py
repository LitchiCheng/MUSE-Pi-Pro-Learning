#!/usr/bin/env python3
"""极简 ONNX 读取器：不装 onnx / 不装 torch 也能看 ModelProto 的结构。

板子上只有 onnxruntime、没有 onnx 模块，想在板上直接对比"官方 int8 图"和"自产 QDQ 图"
的差别（尤其是 Q/DQ 的 zero_point 到底是 uint8 还是 int8），就靠这个。
ONNX 就是 protobuf，这里只做 wire 层解析（varint / length-delimited），
够拿到：opset、节点 (op_type, inputs, outputs, domain)、initializers 的 dtype/dims，
以及 Q/DQ 的类型签名 / scale 秩（per-channel 判据）/ DQ 出边共享数。

    python3 onnxlite.py models/yolov8n-seg.q.onnx models/y11seg_qdq_actint8.onnx

结论性数字（docs/k1-aipu-onnxruntime.md §3.5）就是这么数出来的：
官方 int8 的 251 个 QuantizeLinear 的 zero_point 全是 INT8、scale 全是 0 维（per-tensor），
75 个权重的 DequantizeLinear 的 scale 是 1 维（per-channel）。
"""
import sys, collections


def fields(buf, i, end):
    while i < end:
        key, i = varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(buf, i); yield fno, "v", v
        elif wt == 1:
            yield fno, "f", buf[i:i + 8]; i += 8
        elif wt == 2:
            n, i = varint(buf, i); yield fno, "b", (i, i + n); i += n
        elif wt == 5:
            n, i = varint(buf, i); yield fno, "d", (i, i + n); i += n
        else:
            raise ValueError("wiretype %d @%d" % (wt, i))


def varint(buf, i):
    sh = val = 0
    while True:
        b = buf[i]; i += 1
        val |= (b & 0x7F) << sh
        if not (b & 0x80):
            return val, i
        sh += 7


def s(buf, span):
    return buf[span[0]:span[1]]


DT = {1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16", 6: "INT32", 7: "INT64", 9: "BOOL", 11: "DOUBLE"}


def parse_tensor(buf, span):
    dims, dtype, name, raw = [], None, "", b""
    for fno, kind, val in fields(buf, *span):
        if fno == 1:
            if kind == "b":
                j = val[0]
                while j < val[1]:
                    v, j = varint(buf, j); dims.append(v)
            else:
                dims.append(val)
        elif fno == 2 and kind == "v":
            dtype = val
        elif fno == 8 and kind == "b":
            name = s(buf, val).decode("utf-8", "replace")
        elif fno == 9 and kind == "b":
            raw = s(buf, val)
    return dict(name=name, dtype=DT.get(dtype, dtype), dims=dims, nbytes=len(raw), raw=raw)


def parse_node(buf, span):
    ins, outs, op, nm, dom, attrs = [], [], "", "", "", []
    for fno, kind, val in fields(buf, *span):
        if fno == 1 and kind == "b":
            ins.append(s(buf, val).decode("utf-8", "replace"))
        elif fno == 2 and kind == "b":
            outs.append(s(buf, val).decode("utf-8", "replace"))
        elif fno == 3 and kind == "b":
            nm = s(buf, val).decode("utf-8", "replace")
        elif fno == 4 and kind == "b":
            op = s(buf, val).decode("utf-8", "replace")
        elif fno == 7 and kind == "b":
            dom = s(buf, val).decode("utf-8", "replace")
        elif fno == 5 and kind == "b":
            attrs.append(val)
    return dict(op=op, name=nm, inputs=ins, outputs=outs, domain=dom, attrs=attrs)


def load(path):
    buf = open(path, "rb").read()
    graph = None
    opsets, ir = [], None
    for fno, kind, val in fields(buf, 0, len(buf)):
        if fno == 7 and kind == "b":
            graph = val
        elif fno == 8 and kind == "b":
            op = {}
            for a, k, v in fields(buf, *val):
                if a == 1 and k == "b":
                    op["domain"] = s(buf, v).decode("utf-8", "replace")
                elif a == 2 and k == "v":
                    op["version"] = v
            opsets.append(op)
        elif fno == 1 and kind == "v":
            ir = val
    nodes, inits = [], []
    for fno, kind, val in fields(buf, *graph):
        if fno == 1:
            nodes.append(parse_node(buf, val))
        elif fno == 5:
            inits.append(parse_tensor(buf, val))
    return dict(buf=buf, nodes=nodes, inits=inits, opsets=opsets, ir=ir)


def report(path):
    m = load(path)
    print("== %s" % path)
    print("   ir_version=%s opset=%s nodes=%d inits=%d" % (m["ir"], m["opsets"], len(m["nodes"]), len(m["inits"])))
    print("   op 直方图:", dict(collections.Counter(n["op"] for n in m["nodes"])))
    # 每个 QuantizeLinear 的 (scale dtype, zp dtype)；DQ 同理
    tq = collections.Counter()
    byname = {t["name"]: t for t in m["inits"]}
    prod = {}
    for n in m["nodes"]:
        for o in n["outputs"]:
            prod[o] = n
    qdq_perchannel = collections.Counter()
    for n in m["nodes"]:
        if n["op"] in ("QuantizeLinear", "DequantizeLinear"):
            sig = tuple(byname.get(i, {}).get("dtype", "n/a") if i in byname else prod.get(i, {}).get("op", "?") for i in n["inputs"][1:3])
            tq[(n["op"], sig)] += 1
            dqt = byname.get(n["inputs"][1], {})
            qdq_perchannel[(n["op"], len(dqt.get("dims", [1])))] += 1
    print("   Q/DQ 输入类型签名 (op,(scale_dtype 或 产生者op, zp_dtype)):", dict(tq))
    print("   scale 张量秩分布 (op, ndim) [ndim>1 即 per-channel]:", dict(qdq_perchannel))
    # DQ 的出边数（EP 要求"单条出边到 target node"）
    fan = collections.Counter()
    for n in m["nodes"]:
        for i in n["inputs"]:
            fan[i] += 1
    dq_out = [n["outputs"][0] for n in m["nodes"] if n["op"] == "DequantizeLinear"]
    multi = [t for t in dq_out if fan[t] > 1]
    print("   DQ 张量共 %d 个，其中被多个节点共享(出边>1)的 %d 个" % (len(dq_out), len(multi)))
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
