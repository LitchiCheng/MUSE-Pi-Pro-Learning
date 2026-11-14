#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO11-seg 的 ONNX 推理封装：预处理 / 网络 / 后处理(NMS + mask 合成) / 叠加绘制。

模型约定（ultralytics 官方 yolo11n-seg.onnx，dynamic=False，不含 NMS 子图）：

    images    : [1,3,640,640] float32         letterbox + BGR->RGB + /255
    output0   : [1,116,8400]  float32         4(cx,cy,w,h) + 80(cls) + 32(mask coef)
    output1   : [1,32,160,160] float32        mask protos

另一种导出（SpacemiT 官方 int8 模型 yolov8n-seg.q.onnx）不做层融合，检测头按层摊开成
13 个输出：每层 box[1,64,H,W](DFL logits) / cls[1,80,H,W] / mask系数[1,32,H,W]（H=80,40,20）
+ 1 个 [1,1,H,W](=cls 的 max，代码忽略) + proto[1,32,160,160]。这种布局由
_classify_outputs() 按 shape 自动识别，_decode_raw() 解码成上面同一个 [1,116,8400]。

执行后端：优先 SpaceMITExecutionProvider（K1 智算核），不可用时回退 CPUExecutionProvider。

每个检测结果 dict：
    bbox   (x1,y1,x2,y2)  原图像素坐标
    mbox   (x1,y1,x2,y2)  mask 数组在原图上的对齐区域（crop 模式小于 bbox）
    mask   uint8 (h,w) 0/1，仅覆盖 mbox 区域
    score  float, cls int, name str
"""
from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np

NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

PALETTE = np.random.RandomState(0).randint(40, 255, size=(80, 3), dtype=np.uint8)
MASK_DIM = 32
NCls = 80

# SpaceMIT EP(libspacemit_ep 1.2.2) 的工作线程数 = provider option
# SPACEMIT_EP_INTRA_THREAD_NUM。只有真正落到 AIPU 上的 int8 模型（官方 yolov8n-seg.q.onnx）
# 才吃得到，float 模型在 EP 上没有算子、这个值无所谓。实测它救不了崩溃：
#   6/8 线程 -> 单进程死循环里"跑到第 75 次推理"必 SIGSEGV（4 轮都是 i=74~76）；
#   1~4 线程 -> 死循环能撑 400~600 次，但放进 web（和采集线程并发）里 20~95s 就崩。
# 耗时对它不敏感：yolov8n-seg.q.onnx 单次 sess.run th=4 155ms / th=8 150ms。
# 矩阵与 native 栈见 docs/k1-aipu-onnxruntime.md §7。
EP_THREADS = int(os.environ.get("EP_THREADS", "4"))

# 常驻服务默认用 float，它 97 个 Conv 全在 CPU，线程池要开满（K1 8 核）。
# 换成官方 int8 时线程池必须收到和 EP_THREADS 一样，否则 web 里几十秒就崩，见 _tie_intra()。
CPU_THREADS = 8
INTRA_THREADS = int(os.environ.get("INTRA_THREADS", "0"))   # 0 = 自动；显式设数字做对照实验


# 序列化 protobuf 里算子类型是明文字符串节，直接 grep 二进制就够了（板子上没有 onnx 模块，
# 不想为"决定线程池"这种非正确性判断引入解析依赖）。"QuantizeLinear" 同时命中
# DequantizeLinear（子串），QDQ 图和 QLinear* 图都能覆盖。
QUANT_OP_TAGS = (b"QuantizeLinear", b"QLinearConv", b"QLinearMatMul", b"ConvInteger")


def _has_quant_ops(path):
    """粗判模型是不是 int8 图（QDQ 或 QLinear 写法都算）。

    只用来决定线程池，判错不影响正确性。
    注意别用 "QDQ" 当特征：float 的 yolo11n-seg_opset17.onnx 里恰好也有 1 处 QDQ 字样。
    """
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError:
        return False
    return any(tag in blob for tag in QUANT_OP_TAGS)


def _tie_intra(model_path, use_ep, ep_threads):
    """ORT 的 intra_op_num_threads 该设几。

    实测（同一块板、yolov8n-seg.q.onnx、web 常驻单实例、摄像头 15fps，每档 300s，
    完整矩阵见 docs/k1-aipu-onnxruntime.md §7.1）：
        ep=4 intra=8（老默认） -> 崩 6 次（20~94s 一次）
        ep=4 intra=4           -> 崩 0 次
        ep=4 intra=1           -> 崩 0 次
        ep=8 intra=8           -> 崩 9 次
    即"ep 调到 4 并不够，还得把 ORT 线程池收到和它一样"。推断（未证实）：EP 在 ORT 的
    RunInParallel 回调里按 worker id 索引自己的 per-thread 资源，id 超过
    SPACEMIT_EP_INTRA_THREAD_NUM 就是越界写。float 模型不吃这套（EP 上没算子），照旧开满 8 核。
    配平后的单实例长跑（32 min / 4559 次推理 0 崩）见 docs/k1-aipu-onnxruntime.md §7.2。
    """
    if INTRA_THREADS:
        return INTRA_THREADS
    if use_ep and _has_quant_ops(model_path):
        return min(ep_threads, CPU_THREADS)
    return CPU_THREADS


def make_session(model_path, prefer="auto", intra_threads=None, ep_threads=EP_THREADS):
    """创建 InferenceSession -> (session, providers_used, load_seconds)。

    prefer: auto(有 EP 就用) | spacemit(强制 EP，失败抛错) | cpu
    intra_threads: ORT 线程池；None = 交给 _tie_intra() 按模型自动定
    ep_threads:    传给 SpaceMIT EP 的 SPACEMIT_EP_INTRA_THREAD_NUM，见 EP_THREADS 注释
    """
    import onnxruntime as ort                    # 顺序很重要：必须先 ORT、后 spacemit_ort
    if prefer != "cpu":
        # 反序 import 会让 SpaceMIT EP 在 int8(QLinear) 模型上报
        #   "Attempt to use DefaultLogger but none has been registered"
        # 然后静默回退 CPUExecutionProvider（int8 仿真比 float 还慢 10 倍）。
        # float 模型不会触发（EP 上没有算子，走不到那段日志），所以坑只在量化模型上暴露。
        try:
            import spacemit_ort  # noqa: F401  向 onnxruntime 注册 SpaceMIT EP
        except Exception as exc:
            if prefer == "spacemit":
                raise
            print("[warn] spacemit_ort 不可用（按 CPU 跑）: %s" % exc)

    avail = ort.get_available_providers()
    if prefer == "spacemit" or (prefer == "auto" and "SpaceMITExecutionProvider" in avail):
        providers = ["SpaceMITExecutionProvider", "CPUExecutionProvider"]
        options = [{"SPACEMIT_EP_INTRA_THREAD_NUM": str(ep_threads)}, {}]
    else:
        providers, options = ["CPUExecutionProvider"], [{}]

    so = ort.SessionOptions()
    # prefer=auto 但 EP 没注册成功时 providers 只有 CPU，这时不该收线程池（会白慢 2 倍）
    use_ep = providers[0].startswith("SpaceMIT")
    n_intra = (_tie_intra(model_path, use_ep, ep_threads)
               if intra_threads is None else intra_threads)
    so.intra_op_num_threads = n_intra
    print("[info] intra_op_num_threads=%d (ep=%s quant=%s ep_threads=%d)"
          % (n_intra, use_ep, _has_quant_ops(model_path), ep_threads))
    so.log_severity_level = 3
    t0 = time.time()
    sess = ort.InferenceSession(model_path, sess_options=so,
                                providers=providers, provider_options=options)
    return sess, sess.get_providers(), time.time() - t0


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)      # 截断避免 inf/inf 产生 NaN（两个分支都会被求值）
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _softmax(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _classify_outputs(outs):
    """识别 ONNX 输出布局。

    -> ("ultralytics", None)      2 输出：[1,116,8400] + [1,32,ph,pw]
    -> ("raw", plan)              多输出：每层单独给 box(64)/cls(80)/mask系数(32) 图 + 1 个 proto
       plan = ([(box_name, cls_name, coef_name, H), ... 按 H 降序], proto_name)

    64/80/32 通道数是 Ultralytics seg 检测头的固定写法（4x16 DFL / 80 类 / 32 mask 维），
    据此按 shape 匹配而不是按张量名字匹配 —— 名字在不同导出工具里会变。
    """
    def dims(o):
        return [int(d) if isinstance(d, int) else 0 for d in o.shape]

    if len(outs) == 2:
        return "ultralytics", None
    box, cls, cmap, other = {}, {}, {}, []
    for o in outs:
        d = dims(o)
        if len(d) != 4 or d[0] != 1 or d[2] != d[3]:
            continue
        c, h = d[1], d[2]
        if c == 4 * 16:
            box[h] = o.name
        elif c == NCls:
            cls[h] = o.name
        elif c == MASK_DIM:
            cmap[h] = o.name
        else:
            other.append((o.name, d))
    if not box or set(box) != set(cls):
        raise RuntimeError("无法识别的多输出布局: %s" % [(o.name, dims(o)) for o in outs])
    hs = sorted(box, reverse=True)                      # 80,40,20 -> stride 8,16,32
    protos = [h for h in cmap if h not in box]
    if len(protos) != 1:
        raise RuntimeError("mask proto 不唯一/缺失: %s" % ([ (h, cmap[h]) for h in protos ],))
    proto_h = protos[0]
    coef = [(cmap.get(h), h) for h in hs]
    if any(c is None for c, _ in coef):
        raise RuntimeError("缺少某层的 mask 系数图: %s" % sorted(cmap))
    plan = ([(box[h], cls[h], c, h) for (c, h) in coef], cmap[proto_h])
    return "raw", plan


class YoloSeg:
    def __init__(self, model_path, conf=0.25, iou=0.60, max_det=20,
                 prefer="auto", mask_mode="crop"):
        self.path = os.path.abspath(model_path)
        self.sess, self.providers, load_s = make_session(model_path, prefer)
        ins, outs = self.sess.get_inputs(), self.sess.get_outputs()
        if len(ins) != 1:
            raise RuntimeError("需要单输入模型，实际 in=%s" % ([i.name for i in ins],))
        self.format, self.raw_plan = _classify_outputs(outs)
        self.out_names = [o.name for o in outs]
        self.cls_prob = None          # cls 是概率域(True)还是 logit 域(False)，首帧自动判定
        self.input_name = ins[0].name
        self.input_dtype = np.float16 if "float16" in ins[0].type else np.float32
        shp = [int(d) if isinstance(d, int) else 640 for d in ins[0].shape]
        self.in_h, self.in_w = shp[2], shp[3]
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.mask_mode = mask_mode
        self.load_seconds = load_s
        self.ms = {"pre": 0.0, "infer": 0.0, "post": 0.0, "total": 0.0}

    # ---------------- 预处理 ----------------
    def letterbox(self, bgr):
        h0, w0 = bgr.shape[:2]
        r = min(self.in_h / h0, self.in_w / w0)
        nw, nh = max(1, int(round(w0 * r))), max(1, int(round(h0 * r)))
        px, py = (self.in_w - nw) / 2.0, (self.in_h - nh) / 2.0
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        left, top = int(round(px - 0.1)), int(round(py - 0.1))
        canvas[top:top + nh, left:left + nw] = img
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        blob *= 1.0 / 255.0                                       # ultralytics 预处理：/255 归一化
        return np.ascontiguousarray(blob, dtype=self.input_dtype), r, (px, py), (w0, h0)

    # ---------------- 多输出（未融合）布局解码 ----------------
    def _decode_raw(self, outs):
        """13 输出（每层 box/cls/mask系数 分开）-> 拼成 [1,116,N] + protos。

        box 图是 DFL 的 4x16 分布 logits，需 softmax 后按 bin 序号求期望得到
        (left,top,right,bottom) 四个距离，再以 anchor 中心 (gx+0.5, gy+0.5) 换算成
        cxcywh（输入尺度）。层顺序必须按 H 降序（80,40,20），与 mask 系数拼接顺序一致。
        """
        tensors = dict(zip(self.out_names, outs))
        plan, proto_name = self.raw_plan
        B, C, M = [], [], []
        for (bn, cn, mn, H) in plan:
            b = np.asarray(tensors[bn], np.float32)[0]           # [64,H,H]
            d = _softmax(b.reshape(4, 16, H, H), axis=1)          # 通道 = head*16 + bin
            dist = (d * np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)).sum(1)
            stride = self.in_w / float(H)
            gy, gx = np.mgrid[0:H, 0:H].astype(np.float32)
            cx0, cy0 = gx + 0.5, gy + 0.5
            x1, y1 = (cx0 - dist[0]) * stride, (cy0 - dist[1]) * stride
            x2, y2 = (cx0 + dist[2]) * stride, (cy0 + dist[3]) * stride
            B.append(np.stack([0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1])
                     .reshape(4, H * H))
            C.append(np.asarray(tensors[cn], np.float32)[0].reshape(NCls, H * H))
            M.append(np.asarray(tensors[mn], np.float32)[0].reshape(MASK_DIM, H * H))
        n = sum(x.shape[1] for x in B)
        p = np.zeros((4 + NCls + MASK_DIM, n), np.float32)
        off = 0
        for i, arrs in enumerate(zip(B, C, M)):
            k = arrs[0].shape[1]
            p[:4, off:off + k] = arrs[0]
            p[4:4 + NCls, off:off + k] = arrs[1]
            p[4 + NCls:, off:off + k] = arrs[2]
            off += k
        return p[None], tensors[proto_name]

    # ---------------- 后处理 ----------------
    def postprocess(self, output0, protos, r, pad, orig_wh):
        w0, h0 = orig_wh
        p = np.asarray(output0, np.float32)
        if p.ndim == 4:                                       # [1,C,84,84] -> [1,C,8400]
            p = p.reshape(p.shape[0], p.shape[1], -1)
        p = p[0]                                              # [116, N]
        box_in = p[:4].T                                      # [N,4] cx,cy,w,h（输入尺度）
        cls_in = p[4:4 + NCls].T                               # [N,80]
        coef_in = p[4 + NCls:4 + NCls + MASK_DIM].T            # [N,32]
        if self.cls_prob is None:      # sigmoid 是否已在图内：概率域必落在 [0,1]
            self.cls_prob = bool(cls_in.min() >= -1e-6 and cls_in.max() <= 1.0 + 1e-6)
        if not self.cls_prob:
            cls_in = _sigmoid(cls_in)
        score = cls_in.max(1)
        cid = cls_in.argmax(1)
        k = np.flatnonzero(score > self.conf)
        if k.size == 0:
            return []
        cxy = box_in[k]
        rect = np.empty((k.size, 4), np.float32)               # 原图尺度 x,y,w,h
        rect[:, 0] = (cxy[:, 0] - cxy[:, 2] / 2 - pad[0]) / r
        rect[:, 1] = (cxy[:, 1] - cxy[:, 3] / 2 - pad[1]) / r
        rect[:, 2] = cxy[:, 2] / r
        rect[:, 3] = cxy[:, 3] / r
        rect[:, 0] = rect[:, 0].clip(0, w0 - 1)
        rect[:, 1] = rect[:, 1].clip(0, h0 - 1)
        rect[:, 2] = rect[:, 2].clip(1, w0 - rect[:, 0])
        rect[:, 3] = rect[:, 3].clip(1, h0 - rect[:, 1])

        picked = cv2.dnn.NMSBoxes([[float(a) for a in q] for q in rect],
                                  [float(s) for s in score[k]], self.conf, self.iou)
        if len(picked) == 0:
            return []
        picked = np.array(picked).reshape(-1)[: self.max_det]
        proto = np.asarray(protos, np.float32)[0]              # [32,ph,pw]
        ph, pw = proto.shape[1], proto.shape[2]
        dets = []
        for i in picked:
            x, y, w, h = rect[i]
            x1 = int(max(0, round(x))); y1 = int(max(0, round(y)))
            x2 = int(min(w0, round(x + w))); y2 = int(min(h0, round(y + h)))
            if x2 <= x1 or y2 <= y1:
                continue
            if self.mask_mode == "crop":
                m, mbox = self._mask_crop(proto, coef_in[k][i], ph, pw, r, pad,
                                          (x1, y1, x2, y2), (w0, h0))
            else:
                m = self._mask_full(proto, coef_in[k][i], ph, pw, r, pad, (w0, h0))
                mbox = (0, 0, w0, h0)
            dets.append({"bbox": (x1, y1, x2, y2), "mbox": mbox, "mask": m,
                         "score": float(score[k][i]), "cls": int(cid[k][i]),
                         "name": NAMES[int(cid[k][i])]})
        return dets

    def _mask_full(self, proto, coef, ph, pw, r, pad, orig_wh):
        """标准做法：protos 全图线性组合 -> 还原 letterbox（准确但慢，用于核对精度）。"""
        w0, h0 = orig_wh
        m = _sigmoid(coef @ proto.reshape(MASK_DIM, -1)).reshape(ph, pw)
        m = cv2.resize(m, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((h0, w0), np.float32)
        left, top = int(round(pad[0] - 0.1)), int(round(pad[1] - 0.1))
        canvas[top:top + nh, left:left + nw] = m
        return (canvas >= 0.5).astype(np.uint8)

    def _mask_crop(self, proto, coef, ph, pw, r, pad, box, orig_wh):
        """只在检测框覆盖的 protos 子块上做线性组合，直接缩放回原图框内。

        坐标链：原图 x_og = (x_net - pad_x) / r，而 x_net = x_proto / sx，sx = pw / in_w。
        """
        w0, h0 = orig_wh
        x1, y1, x2, y2 = box
        sx, sy = pw / float(self.in_w), ph / float(self.in_h)
        gx0 = int(max(0, math.floor((x1 * r + pad[0]) * sx)))
        gy0 = int(max(0, math.floor((y1 * r + pad[1]) * sy)))
        gx1 = int(min(pw, math.ceil((x2 * r + pad[0]) * sx)))
        gy1 = int(min(ph, math.ceil((y2 * r + pad[1]) * sy)))
        if gx1 <= gx0 or gy1 <= gy0:
            return np.zeros((max(1, y2 - y1), max(1, x2 - x1)), np.uint8), box
        sub = proto[:, gy0:gy1, gx0:gx1]
        m = _sigmoid(coef @ sub.reshape(MASK_DIM, -1)).reshape(gy1 - gy0, gx1 - gx0)
        ox0 = max(0, int(math.floor((gx0 / sx - pad[0]) / r)))
        oy0 = max(0, int(math.floor((gy0 / sy - pad[1]) / r)))
        ox1 = min(w0, int(math.ceil((gx1 / sx - pad[0]) / r)))
        oy1 = min(h0, int(math.ceil((gy1 / sy - pad[1]) / r)))
        ox1, oy1 = max(ox1, ox0 + 1), max(oy1, oy0 + 1)
        m = cv2.resize(m, (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_LINEAR)
        return (m >= 0.5).astype(np.uint8), (ox0, oy0, ox1, oy1)

    # ---------------- 推理 ----------------
    def infer(self, bgr):
        t0 = time.time()
        blob, r, pad, (w0, h0) = self.letterbox(bgr)
        t1 = time.time()
        outs = self.sess.run(None, {self.input_name: blob})
        t2 = time.time()
        if self.format == "raw":
            output0, protos = self._decode_raw(outs)
        else:
            output0, protos = outs[0], outs[1]
        dets = self.postprocess(output0, protos, r, pad, (w0, h0))
        t3 = time.time()
        self.ms = {"pre": 1e3 * (t1 - t0), "infer": 1e3 * (t2 - t1),
                   "post": 1e3 * (t3 - t2), "total": 1e3 * (t3 - t0)}
        return dets

    # ---------------- 绘制 ----------------
    @staticmethod
    def draw(bgr, dets, alpha=0.45, with_mask=True):
        """mask 半透明叠加 + 轮廓 + 检测框与标签；返回新图，不改入参。"""
        img = bgr.copy()
        if with_mask:
            for d in dets:
                mx1, my1, mx2, my2 = d["mbox"]
                mx1, my1 = max(0, mx1), max(0, my1)
                mx2, my2 = min(img.shape[1], mx2), min(img.shape[0], my2)
                if mx2 <= mx1 or my2 <= my1:
                    continue
                m = d["mask"]
                if m.shape[:2] != (my2 - my1, mx2 - mx1):
                    m = cv2.resize(m, (mx2 - mx1, my2 - my1), interpolation=cv2.INTER_NEAREST)
                reg = img[my1:my2, mx1:mx2]
                color = PALETTE[d["cls"]]
                # 掩膜混合走 OpenCV C++ 路径，代替 numpy 布尔索引：同一块区域布尔索引+乘加
                # 要 ~35ms/目标，这里 <5ms（实测整幅 draw 172ms -> 45ms）。
                # Python 绑定的 addWeighted 只到 (src1,a,src2,b,gamma[,dst[,dtype]])，没有 mask，
                # 所以拆成“整块加权 + copyTo 按 mask 写回”两步。
                col = np.empty_like(reg)
                col[:] = color
                cv2.copyTo(cv2.addWeighted(reg, 1.0 - alpha, col, alpha, 0.0), m, reg)
                cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                if cnts:
                    shift = np.array([[[mx1, my1]]], np.int32)
                    cv2.drawContours(img[my1:my2, mx1:mx2], [c - shift for c in cnts],
                                     -1, (255, 255, 255), 1)
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            color = tuple(int(v) for v in PALETTE[d["cls"]])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = "%s %.2f" % (d["name"], d["score"])
            org = (x1, max(14, y1 - 5))
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def main():
    import argparse
    ap = argparse.ArgumentParser(description="YOLO11-seg 单图/随机帧自测")
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", help="jpg/png；不给则用随机噪声图")
    ap.add_argument("--prefer", default="auto", choices=["auto", "spacemit", "cpu"])
    ap.add_argument("--mask-mode", default="crop", choices=["crop", "full"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", default="out_seg.jpg")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    m = YoloSeg(args.model, conf=args.conf, prefer=args.prefer, mask_mode=args.mask_mode)
    print("providers=%s load=%.1fs input=%s/%s dtype=%s" %
          (m.providers, m.load_seconds, m.input_name, (m.in_h, m.in_w), m.input_dtype))
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit("读不到图片 %s" % args.image)
    else:
        frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    ts = []
    for i in range(args.runs):
        dets = m.infer(frame)
        ts.append(m.ms["total"])
        print("run%d total=%.1fms (pre %.1f / infer %.1f / post %.1f) n=%d" %
              (i, m.ms["total"], m.ms["pre"], m.ms["infer"], m.ms["post"], len(dets)))
    print("avg %.1f ms -> %.2f fps" % (sum(ts) / len(ts), 1000.0 * len(ts) / sum(ts)))
    cv2.imwrite(args.save, YoloSeg.draw(frame, dets))
    print("saved", args.save, "det:", [(d["name"], round(d["score"], 2)) for d in dets])


if __name__ == "__main__":
    main()
