#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO11-seg 的 ONNX 推理封装：预处理 / 网络 / 后处理(NMS + mask 合成) / 叠加绘制。

模型约定（ultralytics 官方 yolo11n-seg.onnx，dynamic=False，不含 NMS 子图）：

    images    : [1,3,640,640] float32         letterbox + BGR->RGB + /255
    output0   : [1,116,8400]  float32         4(cx,cy,w,h) + 80(cls) + 32(mask coef)
    output1   : [1,32,160,160] float32        mask protos

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


def make_session(model_path, prefer="auto", intra_threads=8):
    """创建 InferenceSession -> (session, providers_used, load_seconds)。

    prefer: auto(有 EP 就用) | spacemit(强制 EP，失败抛错) | cpu
    """
    if prefer != "cpu":
        try:
            import spacemit_ort  # noqa: F401  给 onnxruntime 注入 SpaceMIT EP
        except Exception as exc:
            if prefer == "spacemit":
                raise
            print("[warn] spacemit_ort 不可用（按 CPU 跑）: %s" % exc)
    import onnxruntime as ort

    avail = ort.get_available_providers()
    if prefer == "spacemit" or (prefer == "auto" and "SpaceMITExecutionProvider" in avail):
        providers = ["SpaceMITExecutionProvider", "CPUExecutionProvider"]
        options = [{"SPACEMIT_EP_INTRA_THREAD_NUM": str(intra_threads)}, {}]
    else:
        providers, options = ["CPUExecutionProvider"], [{}]

    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.log_severity_level = 3
    t0 = time.time()
    sess = ort.InferenceSession(model_path, sess_options=so,
                                providers=providers, provider_options=options)
    return sess, sess.get_providers(), time.time() - t0


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)      # 截断避免 inf/inf 产生 NaN（两个分支都会被求值）
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class YoloSeg:
    def __init__(self, model_path, conf=0.25, iou=0.60, max_det=20,
                 prefer="auto", mask_mode="crop"):
        self.path = os.path.abspath(model_path)
        self.sess, self.providers, load_s = make_session(model_path, prefer)
        ins, outs = self.sess.get_inputs(), self.sess.get_outputs()
        if len(ins) != 1 or len(outs) != 2:
            raise RuntimeError("需要 ultralytics seg 导出(1 输入 2 输出)，实际 in=%s out=%s"
                               % ([i.name for i in ins], [o.name for o in outs]))
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
        dets = self.postprocess(outs[0], outs[1], r, pad, (w0, h0))
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
                sel = m.astype(bool)
                reg = img[my1:my2, mx1:mx2]
                out = reg.copy()
                color = PALETTE[d["cls"]]
                out[sel] = (reg[sel] * (1 - alpha) + color * alpha).astype(np.uint8)
                img[my1:my2, mx1:mx2] = out
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
