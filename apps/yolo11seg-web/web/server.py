#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K1 / MUSE Pi Pro 上 "OV5647 摄像头 -> YOLO11-seg 分割 -> 浏览器实时查看" 的 Web 后端。

三个线程：
    Cam    : 起 k1cam 子进程读 /dev/video50 的 NV12 码流，转成 BGR，只保留最新一帧
    Infer  : 取最新帧做推理，保存检测结果与耗时（比推理慢的帧直接丢弃，不排队）
    HTTP   : Flask；/video 用 MJPEG 多路复用把"最新帧 + 最新结果"叠加后推给浏览器

关键点：帧队列长度恒为 1（只保留最新帧）。推理比采集慢时会自动跳帧，
浏览器看到的画面仍然是采集帧率（结果叠加在每一帧上），而 HUD 里的 infer fps
才是真实的分割帧率——不假装实时。

用法（板端）：
    LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1 python3 server.py \
        --model ../models/yolo11n-seg_opset17.onnx --cam ../cam/k1cam --port 8000
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROTATE = 0            # 采集帧统一旋转角度（0/90/180/270），由 --rotate 设置
YUV_FULL_RANGE = False  # --full-range：把 OpenCV 的 limited-range(BT.601) 转换补偿成全量程
# cv2 的 YUV2BGR 一律按 limited-range 算：R = 1.164*(Y-16) + 1.596*(V-128)。
# 传感器/ISP 出的是 full-range 时，Y=19 会被压成 3.8（黑场直接剪到 0，YOLO 输入跟着废）。
# 对结果再套一次 f(x) = x/1.164 + 16 即可等价还原成 full-range：
#   f(R) = Y + 1.371*(V-128)，而 full-range 真值是 Y + 1.402*(V-128)，偏差 2%（肉眼不可见）。
RANGE_LUT = np.clip(np.round(np.arange(256, dtype=np.float64) / 1.164 + 16), 0, 255).astype(np.uint8)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from yolo_seg import PALETTE, YoloSeg  # noqa: E402

RT_LIB = "/usr/lib/libonnxruntime.so.1.18.1"


def ensure_ld_preload():
    """K1 上同时存在两份 libonnxruntime（python capi 的 1.18.0 与 /usr/lib 的 1.18.1）。

    不预加载 /usr/lib 那份时，SpaceMIT EP 会打 "Attempt to use DefaultLogger but none
    has been registered" 然后静默回退 CPU。LD_PRELOAD 只能在 exec 时生效，所以这里
    在发现缺失时原地 re-exec 一次（用环境变量做防递归）。
    """
    if sys.platform != "linux" or os.uname().machine != "riscv64":
        return
    if os.environ.get("MUSE_NO_LD_PRELOAD") == "1" or "LD_PRELOAD" in os.environ:
        return
    if not os.path.exists(RT_LIB):
        return
    os.environ["LD_PRELOAD"] = RT_LIB
    print("[info] 重新以 LD_PRELOAD=%s 启动本进程" % RT_LIB)
    os.execv(sys.executable, [sys.executable] + sys.argv)


# --------------------------------------------------------------------------- 配置
class Config:
    """可在线改的参数（/config 接口）。conf/iou 直接写进 model 实例，下一帧生效。"""

    def __init__(self, **kw):
        self._lock = threading.Lock()
        self.conf = kw.get("conf", 0.25)
        self.iou = kw.get("iou", 0.60)
        self.masks = kw.get("masks", True)
        self.alpha = kw.get("alpha", 0.45)
        self.hud = kw.get("hud", True)
        self.quality = kw.get("quality", 80)
        self.fps_out = kw.get("fps_out", 15)
        self.paused = False

    def snapshot(self):
        with self._lock:
            return {k: getattr(self, k) for k in
                    ("conf", "iou", "masks", "alpha", "hud", "quality", "fps_out", "paused")}

    def update(self, **kw):
        out = {}
        with self._lock:
            for k, v in kw.items():
                if not hasattr(self, k) or k in ("conf", "iou") and v is None:
                    continue
                if k in ("conf", "iou", "alpha", "quality", "fps_out"):
                    v = float(v)
                elif k in ("masks", "hud", "paused"):
                    v = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "on", "yes")
                setattr(self, k, v)
                out[k] = v
        return out



def rotate_frame(img):
    """模组装反时用它纠偏：所有采集源在发布前统一旋转，推理与绘制看到的是同一份像素。"""
    if ROTATE == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if ROTATE == 180:
        return cv2.flip(img, -1)
    if ROTATE == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img
# --------------------------------------------------------------------------- 采集
class Camera:
    """k1cam 子进程 -> 最新一帧 BGR。进程挂了自动重启。"""

    # 数字黑帧持续多久就在日志里提醒一次（本进程绝不主动 pkill cam-test，见 _lum_stats）
    BLACK_WARN_S = 5.0

    def __init__(self, k1cam, width, height, dev="/dev/video50", log_path="/tmp/k1cam.log"):
        self.bin, self.w, self.h, self.dev = k1cam, width, height, dev
        self.nbytes = width * height * 3 // 2
        self.log_path = log_path
        self.wait_s = int(os.environ.get("K1CAM_WAIT", "20"))
        self._last_hint = 0.0
        self._cond = threading.Condition()
        self.frame, self.seq, self.ts = None, 0, 0.0
        self.stat = {"frames": 0, "restarts": 0, "errors": 0, "opened": False,
                     "lum": 0.0, "lum_mean": 0.0, "black_s": 0.0, "stall_s": 0.0, "black_warns": 0,
                     "yp": None, "help_s": 0.0}
        self._black_since = None
        self._last_warn = 0.0
        self._proc = None                      # 当前 k1cam 子进程（退出时必收尾，见 _cleanup）
        self._stop = False
        self._t0 = time.time()
        self._th = threading.Thread(target=self._run, name="cam", daemon=True)
        self._th.start()
        # 关键：python 被 kill 掉不管 k1cam 的话，k1cam 会卡在 D 状态一直占着 /dev/video50，
        # 之后任何进程 open 都只会 EAGAIN，只能重启板子。所以 atexit + 信号都要收尾。
        atexit.register(self._cleanup)
        try:
            signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))   # SystemExit -> atexit 跑 _cleanup
            signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
        except Exception:
            pass

    def _cleanup(self):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(3)
        except Exception:
            proc.kill()

    def status(self):
        with self._cond:
            frames, ts = self.stat["frames"], self.ts
        el = max(1e-6, (ts - self._t0) if frames else (time.time() - self._t0))
        return dict(self.stat, fps=round(frames / el, 2),
                    stall_s=round(max(0.0, time.time() - ts) if ts else -1.0, 2),
                    help_s=round(time.time() - self._t0, 1) if frames == 0 else 0.0)

    def _lum_stats(self, gray):
        """只统计亮度分布，绝不在这里动 cam-test。

        黑帧签名：Y mean≈1 / std≈0.5 且 UV=128（中性）—— 是"有第二发 cam-test 在抢
        pipeline"，不是"没光"（没光仍有噪声纹理），也不是"常数 0"（buffer 没被写，
        那是 dma heap_flags 用错）。
        实测：在采集端已经 open + STREAMON 时 pkill cam-test，帧数会立刻冻结且 k1cam
        进入 kill -9 无效的 D 状态，只能重启板子。所以这里只报警、给出手动处置命令，
        自愈动作交给 _run() 的"子进程退出就重启自己这一发 reader"（不碰 cam-test）。
        """
        lum, mean = float(gray.std()), float(gray.mean())
        black = lum < 2.0 and mean < 4.0
        now = time.time()
        # Y 分位数：判"全量程还是 limited-range"只看这个最省事 ——
        # limited-range 的合法域是 16..235，p0 会顶在 >=16；full-range 则 p0=0、p100=255。
        p = [int(x) for x in np.percentile(gray, (0, 1, 50, 99, 100))]
        with self._cond:
            self.stat["lum"] = round(lum, 1)
            self.stat["lum_mean"] = round(mean, 1)
            self.stat["yp"] = p
            if black:
                if self._black_since is None:
                    self._black_since = now
                self.stat["black_s"] = round(now - self._black_since, 2)
                need = (self._black_since is not None
                        and now - self._black_since > self.BLACK_WARN_S
                        and now - self._last_warn > 30.0)
            else:
                self._black_since = None
                self.stat["black_s"] = 0.0
                need = False
            if need:
                self._last_warn = now
                self.stat["black_warns"] += 1
        if need:
            print("[cam] 连续 %.1fs 数字黑帧(std=%.2f, mean=%.2f)：疑似有第二发 cam-test 在抢 pipeline。"
                  "查 `ps -eo pid,ppid,etime,comm | grep cam-test` 把多余的清到只剩 1 发；"
                  "不要 pkill 正在推流的那发（会让本采集进程卡成 D 状态，只能重启板子）"
                  % (now - self._black_since, lum, mean), file=sys.stderr)

    def _spawn(self):
        log = open(self.log_path, "wb")
        proc = subprocess.Popen(
            [self.bin, "-d", self.dev, "-W", str(self.w), "-H", str(self.h),
             "-f", "NV12", "-n", "0", "-w", str(self.wait_s)],
            stdout=subprocess.PIPE, stderr=log, bufsize=self.nbytes * 2)
        self._proc = proc
        return proc

    def _camtest_n(self):
        """cam-test 实例数。>1 必黑帧；==0 且我们还没拿到帧 = 内核自拉那发握手失败了。"""
        try:
            r = subprocess.run(["pgrep", "-c", "-x", "cam-test"], capture_output=True, timeout=3)
            return int((r.stdout or b"0").decode().strip() or 0)
        except Exception:
            return -1

    def _stuck_hint(self):
        """一帧都没有时打一条能直接照抄的处置命令（本进程不碰 cam-test，理由见 _lum_stats）。"""
        if self._last_hint and time.time() - self._last_hint < 30.0:
            return
        self._last_hint = time.time()
        n = self._camtest_n()
        if n == 0:
            print("[cam] 0 帧且没有 cam-test 在跑：内核 open() 自拉的那发握手失败了。"
                  "处置：先 `pgrep -x k1cam` 确认没有残留采集端，再跑 `bash cam-start.sh`"
                  "（它会在设备空闲时清残留并补拉一发 cam-test）", file=sys.stderr)
        elif n > 1:
            print("[cam] 0 帧且有 %d 发 cam-test：多实例会互相抢 pipeline。"
                  "清到只剩 1 发：ps -eo pid,ppid,etime,comm | grep cam-test" % n, file=sys.stderr)
        else:
            print("[cam] 0 帧但有 1 发 cam-test：多半是采集端被卡在 D 状态占着设备，"
                  "看 `ps -eo pid,stat,comm | grep k1cam`；D 状态 kill -9 无效，"
                  "要么把 pipeline 恢复出帧再杀，要么重启板子", file=sys.stderr)

    def _publish(self, bgr):
        if YUV_FULL_RANGE:
            bgr = cv2.LUT(bgr, RANGE_LUT)
        with self._cond:
            self.frame, self.ts = rotate_frame(bgr), time.time()
            self.seq += 1
            self.stat["frames"] += 1
            self._cond.notify_all()

    def _run(self):
        while not self._stop:
            proc = None
            try:
                proc = self._spawn()
                self.stat["opened"] = True
                buf = bytearray()
                while True:
                    r = proc.stdout.read(self.nbytes - len(buf))
                    if not r:
                        break
                    buf += r
                    if len(buf) < self.nbytes:
                        continue
                    plane = np.frombuffer(bytes(buf), np.uint8).reshape(self.h * 3 // 2, self.w)
                    buf = bytearray()
                    self._publish(cv2.cvtColor(plane, cv2.COLOR_YUV2BGR_NV12))
                    # 亮度统计在小尺寸灰度上做，单帧开销 <1ms（plane 是 NV12：前 h 行是 Y）
                    self._lum_stats(cv2.resize(plane[:self.h:2, :self.w:2], (160, 90),
                                               interpolation=cv2.INTER_AREA))
            except Exception as exc:  # noqa: BLE001
                self.stat["errors"] += 1
                print("[cam] 异常: %r" % (exc,), file=sys.stderr)
            finally:
                self.stat["opened"] = False
                with self._cond:
                    got_any = self.stat["frames"] > 0
                if not got_any:
                    self._stuck_hint()
                if proc:
                    proc.terminate()
                    try:
                        proc.wait(2)
                    except Exception:  # noqa: BLE001
                        proc.kill()
            if self._stop:
                break
            self.stat["restarts"] += 1
            print("[cam] k1cam 退出，1s 后重启", file=sys.stderr)
            time.sleep(1.0)

    def latest(self, timeout=2.0, after=-1):
        """等到比 after 新的帧，返回 (bgr, seq)；超时返回 (None, -1)。"""
        with self._cond:
            if self.seq <= after and not self._cond.wait(timeout):
                return None, after
            return self.frame, self.seq

    def peek(self):
        with self._cond:
            return self.frame, self.seq, self.ts


class StillSource:
    """没有摄像头/不能起 k1cam 时的调试源：反复输出同一张图。"""

    def __init__(self, path, fps=15):
        img = cv2.imread(path)
        if img is None:
            raise SystemExit("读不到 %s" % path)
        self.base, self.frame, self.seq, self.ts = rotate_frame(img), None, 0, 0.0
        self._t0, self._fps = time.time(), fps
        self._cond = threading.Condition()
        self._stop = False
        threading.Thread(target=self._run, name="cam", daemon=True).start()

    def _run(self):
        dt = 1.0 / self._fps
        while not self._stop:
            t = time.time()
            with self._cond:
                self.frame, self.seq, self.ts = self.base.copy(), self.seq + 1, t
                self._cond.notify_all()
            time.sleep(max(0.0, dt - (time.time() - t)))

    def latest(self, timeout=2.0, after=-1):
        with self._cond:
            if self.seq <= after and not self._cond.wait(timeout):
                return None, after
            return self.frame, self.seq

    def peek(self):
        with self._cond:
            return self.frame, self.seq, self.ts

    def status(self):
        with self._cond:
            n = self.seq
        return {"frames": n, "restarts": 0, "errors": 0, "opened": True,
                "fps": round(n / max(1e-6, time.time() - self._t0), 2)}


# --------------------------------------------------------------------------- 推理线程
class Engine:
    def __init__(self, model, source, cfg):
        self.model, self.src, self.cfg = model, source, cfg
        self.t0 = time.time()
        self._lock = threading.Lock()
        self.dets, self.dets_seq = [], -1
        self.ms = {"pre": 0.0, "infer": 0.0, "post": 0.0, "total": 0.0}
        self.n = 0
        self._stop = False
        self._th = threading.Thread(target=self._run, name="infer", daemon=True)
        self._th.start()

    def _run(self):
        last = -1
        while not self._stop:
            if self.cfg.snapshot()["paused"]:
                time.sleep(0.2)
                continue
            t_wait = time.time()
            frame, seq = self.src.latest(timeout=1.0, after=last)
            wait_ms = (time.time() - t_wait) * 1000.0
            last = seq if seq > last else last
            if frame is None:
                continue
            try:
                self.model.conf = self.cfg.conf
                self.model.iou = self.cfg.iou
                dets = self.model.infer(frame)
            except Exception as exc:  # noqa: BLE001
                print("[infer] 失败: %r" % (exc,), file=sys.stderr)
                time.sleep(0.2)
                continue
            with self._lock:
                self.dets, self.dets_seq, self.ms, self.n = dets, seq, dict(self.model.ms), self.n + 1
                # wait = 等"比上一帧新"的采集帧花了多久。total+wait ≈ 1/fps 说明瓶颈在采集/调度侧，
                # 不是推理本身（量化模型上就出现过 total 419ms 但实际只有 0.21 次/秒的情况）。
                self.ms["wait"] = wait_ms

    def result(self):
        with self._lock:
            return list(self.dets), self.dets_seq, dict(self.ms), self.n

    def stop(self):
        self._stop = True


# --------------------------------------------------------------------------- 叠加/推流
def hud_overlay(img, info):
    if not info.get("hud", True):
        return img
    h, w = img.shape[:2]
    bar = np.zeros((34, w, 3), np.uint8)
    bar[:, :] = (20, 20, 20)
    img[0:34, 0:w] = cv2.addWeighted(img[0:34, 0:w], 0.35, bar, 0.65, 0)
    txt = ("YOLO11-seg  infer %.0f ms/frame  %.2f fps | view %d fps | objs %d | %s | %s"
           % (info["infer_ms"], info["infer_fps"], info["view_fps"], info["n"],
              info["providers"], time.strftime("%H:%M:%S")))
    cv2.putText(img, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)
    return img


def annotate(frame, dets, cfg):
    s = cfg.snapshot()
    img = YoloSeg.draw(frame, dets, alpha=s["alpha"], with_mask=s["masks"]) if dets else frame.copy()
    return img, s


# --------------------------------------------------------------------------- 服务
class App:
    def __init__(self, args):
        ensure_ld_preload()
        self.cfg = Config(conf=args.conf, iou=args.iou, quality=args.quality)
        self.model = YoloSeg(args.model, conf=args.conf, iou=args.iou,
                             prefer=args.prefer, mask_mode=args.mask_mode)
        if args.video:
            self.src = StillSourceVideo(args.video)
        elif args.still:
            self.src = StillSource(args.still)
        else:
            self.cam_log = args.cam_log
            self.src = Camera(args.cam, args.width, args.height, args.dev, args.cam_log)
        self.eng = Engine(self.model, self.src, self.cfg)
        self.t0 = time.time()
        self.views = {"n": 0, "t": time.time()}
        self._vlock = threading.Lock()
        print("[info] providers=%s input=%s load=%.1fs" %
              (self.model.providers, (self.model.in_h, self.model.in_w), self.model.load_seconds))

    # ---- 统计 ----
    def stats(self):
        dets, seq, ms, runs = self.eng.result()
        now = time.time()
        with self._vlock:
            view_fps = self.views["n"] / max(1e-6, now - self.views["t"])
        cam_st = self.src.status()
        return {
            "uptime": round(now - self.t0, 1),
            "cam": {"fps": cam_st["fps"], "size": [self.src.w if hasattr(self.src, "w") else 0,
                                                   self.src.h if hasattr(self.src, "h") else 0],
                    "frames": cam_st["frames"], "restarts": cam_st["restarts"],
                    "errors": cam_st["errors"],
                    # lum=Y 平面方差, lum_mean=Y 均值；lum<2 且 lum_mean<4 => cam-test 没在推流的黑帧
                    "lum": cam_st.get("lum"), "lum_mean": cam_st.get("lum_mean"),
                    "black_s": cam_st.get("black_s"), "stall_s": cam_st.get("stall_s"),
                    "black_warns": cam_st.get("black_warns"), "yp": cam_st.get("yp"),
                    "rotate": ROTATE, "log": self.cam_log if hasattr(self, "cam_log") else None},
            "view_fps": round(view_fps, 2),
            "infer": {"fps": round(runs / max(1e-6, now - self.eng.t0), 2),
                      "instant_fps": round(1000.0 / max(1e-6, ms["total"]), 2), "runs": runs,
                      "ms": {k: round(v, 1) for k, v in ms.items()}},
            "model": {"providers": self.model.providers, "input": [self.model.in_w, self.model.in_h],
                      "path": os.path.basename(self.model.path)},
            "dets": [{"name": d["name"], "cls": int(d["cls"]), "score": round(float(d["score"]), 3),
                      "bbox": [int(x) for x in d["bbox"]],
                      "color": [int(c) for c in PALETTE[int(d["cls"]) % len(PALETTE)]]} for d in dets],
            "config": self.cfg.snapshot(),
        }

    # ---- MJPEG ----
    def mjpeg(self):
        b = b"--frame\r\n"
        while True:
            frame, _, _ = self.src.peek()
            dets, _, ms, _ = self.eng.result()
            if frame is None:
                time.sleep(0.05)
                continue
            img, s = annotate(frame, dets, self.cfg)
            info = {"infer_ms": ms["total"], "infer_fps": 1000.0 / max(1e-6, ms["total"]),
                    "view_fps": 0, "n": len(dets), "hud": s["hud"],
                    "providers": "+".join(p.replace("ExecutionProvider", "") for p in self.model.providers)}
            with self._vlock:
                self.views["n"] += 1
            info["view_fps"] = self.views["n"] / max(1e-6, time.time() - self.views["t"])
            img = hud_overlay(img, info)
            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(s["quality"])])
            if not ok:
                continue
            yield b + b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg.tobytes()) \
                + jpg.tobytes() + b"\r\n"
            time.sleep(max(0.0, 1.0 / s["fps_out"] - 0.005))


class StillSourceVideo(StillSource):
    def __init__(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit("打不开视频 %s" % path)
        self.cap, self.frame, self.seq, self.t0 = cap, None, 0, time.time()
        self._cond = threading.Condition()
        self._stop = False
        threading.Thread(target=self._runv, daemon=True).start()

    def _runv(self):
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self._cond:
                self.frame, self.seq = rotate_frame(f), self.seq + 1
                self._cond.notify_all()

    def latest(self, timeout=2.0):
        with self._cond:
            if self.frame is None:
                self._cond.wait(timeout)
            return self.frame, self.seq

    def peek(self):
        with self._cond:
            return self.frame, self.seq, time.time()

    def status(self):
        return {"frames": self.seq, "restarts": 0, "errors": 0, "opened": True,
                "fps": round(self.seq / max(1e-6, time.time() - self.t0), 2)}


def build_flask(app):
    from flask import Flask, Response, jsonify, request, send_from_directory

    srv = Flask(__name__, static_folder=HERE)

    @srv.route("/")
    def index():
        return send_from_directory(HERE, "index.html")

    @srv.route("/video")
    def video():
        return Response(app.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @srv.route("/snapshot.jpg")
    def snapshot():
        frame, _, _ = app.src.peek()
        dets, _, _, _ = app.eng.result()
        if frame is None:
            return "no frame yet", 503
        img, _ = annotate(frame, dets, app.cfg)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return Response(jpg.tobytes(), mimetype="image/jpeg")

    @srv.route("/stats")
    @srv.route("/api/stats")
    def stats():
        return jsonify(app.stats())

    @srv.route("/config", methods=["GET", "POST"])
    def config():
        if request.method == "GET":
            return jsonify(app.cfg.snapshot())
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        return jsonify({"ok": True, "updated": app.cfg.update(**data), "now": app.cfg.snapshot()})

    @srv.route("/health")
    def health():
        return "ok"

    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "..", "models", "yolo11n-seg_opset17.onnx"))
    ap.add_argument("--cam", default=os.path.join(HERE, "..", "cam", "k1cam"))
    ap.add_argument("--dev", default="/dev/video50")
    ap.add_argument("--cam-log", default="/tmp/k1cam-server.log")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--prefer", default="auto", choices=["auto", "spacemit", "cpu"])
    ap.add_argument("--mask-mode", default="crop", choices=["crop", "full"])
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--still", help="无摄像头时用一张图当输入（调试）")
    ap.add_argument("--video", help="无摄像头时用 mp4 当输入（调试）")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="采集帧旋转角度；OV5647 模组装反时用 180")
    ap.add_argument("--full-range", action="store_true",
                    help="NV12 是 full-range(JPEG 那种) 时用它补偿 cv2 默认的 limited-range 转换，"
                         "否则暗部会被整体压到 0（既难看也废掉模型输入）")
    args = ap.parse_args()
    global ROTATE, YUV_FULL_RANGE
    ROTATE = args.rotate
    YUV_FULL_RANGE = args.full_range
    if not os.path.exists(args.model):
        raise SystemExit("模型不存在: %s" % args.model)
    app = App(args)
    srv = build_flask(app)
    try:
        srv.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    finally:
        app.eng.stop()


if __name__ == "__main__":
    main()
