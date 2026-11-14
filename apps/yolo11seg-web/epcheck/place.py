#!/usr/bin/env python3
"""把 ep.py 的日志统计成"每个 EP 上有哪些算子、各几个"。这就是"有没有吃到智算核"的判据。

    python3 place.py yolo11n-seg_opset17.onnx.log yolov8n-seg.q.onnx.log

日志里的节点行长这样（EP 自己折出来的融合 kernel 会带 EP 前缀的新名字）：
    ... VerifyEachNodeIsAssignedToAnEp]  Node(s) placed on [SpaceMITExecutionProvider]. Number of nodes: 164
    ... VerifyEachNodeIsAssignedToAnEp]   QLinearConv (SpaceMITExecutionProvider_QLinearConv_0)
    ... VerifyEachNodeIsAssignedToAnEp]   Conv (/model.2/conv/Conv)
所以：先按 "Node(s) placed on [X]" 切段，段内的 op 名就归到 X。
"""
import collections
import re
import sys

HDR = re.compile(r"Node\(s\) placed on \[(\w+)\]\. Number of nodes: (\d+)")
NODE = re.compile(r"VerifyEachNodeIsAssignedToAnEp\]\s{2,}(\w+) \(")


def parse(path):
    placed, cur = {}, None
    hist = collections.Counter()
    with open(path, "r", errors="ignore") as f:
        for line in f:
            h = HDR.search(line)
            if h:
                cur = h.group(1)
                placed[cur] = int(h.group(2))
                continue
            if cur:
                n = NODE.search(line)
                if n:
                    hist[(cur, n.group(1))] += 1
    return placed, hist


def main(paths):
    for p in paths:
        placed, hist = parse(p)
        print("== %s" % p)
        if not placed:
            print("   [no] 日志里没有节点清单：ep.py 是否带 LD_PRELOAD / log_severity_level=0？")
            continue
        for ep, total in placed.items():
            ops = sorted([(n, o) for (e, o), n in hist.items() if e == ep], reverse=True)
            got = sum(n for n, _ in ops)
            print("   [%s] 自报 %d 个节点，清单列出 %d 个" % (ep, total, got))
            print("        %s" % (", ".join("%s x%d" % (o, n) for n, o in ops[:14]) or "-"))
        # 一句话判据：卷积在谁名下。float yolo11 是 CPU 101 / EP 0，官方 int8 是 EP 75 / CPU 1
        conv = collections.Counter()
        for (e, o), n in hist.items():
            if "Conv" in o:
                conv[e] += n
        print("   Conv 类（含 QLinear/Fused/Transpose）: " +
              " / ".join("%s %d 个" % (e, n) for e, n in conv.items()) +
              ("  -> 卷积在 CPU，等于没吃到智算核" if not conv.get("SpaceMITExecutionProvider") else
               "  -> 卷积落在 SpaceMIT EP，走的是 int8 kernel"))


if __name__ == "__main__":
    main(sys.argv[1:] or ["/dev/null"])
