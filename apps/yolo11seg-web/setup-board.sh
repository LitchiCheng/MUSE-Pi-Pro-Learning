#!/usr/bin/env bash
# MUSE Pi Pro (SpacemiT K1 / Bianbu Star) 板端一键环境准备。在开发板上执行：
#
#   ./setup-board.sh                依赖 + k1cam + 相机配置 + 模型 + 单帧自检
#   ./setup-board.sh --service      额外装两个 systemd 服务（cam-test + web）并开机自启
#   ./setup-board.sh --skip-camera  只处理软件依赖和模型
#   ./setup-board.sh --no-model     不下载模型（自己放好后重跑）
#
# 幂等：可反复执行，每步先检测再动作。sudo 需要密码时用 MUSE_SUDO_PASS 环境变量传入。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ---- sudo helper：优先复用凭证缓存；有 MUSE_SUDO_PASS 就走 askpass（保持 stdin 干净，
#      这样 SUDO tee / SUDO python3 - 这类从 stdin 读数据的写法不会被密码管道吃掉）----
_ASKPASS_DIR=""
cleanup_askpass() { [ -n "$_ASKPASS_DIR" ] && rm -rf "$_ASKPASS_DIR" 2>/dev/null; }
trap cleanup_askpass EXIT
SUDO() {
  if [ "$(id -u)" -eq 0 ]; then "$@"; return $?; fi
  if sudo -n "$@" 2>/dev/null; then return 0; fi
  if [ -n "${MUSE_SUDO_PASS:-}" ]; then
    export MUSE_SUDO_PASS
    if [ -z "$_ASKPASS_DIR" ]; then
      _ASKPASS_DIR=$(mktemp -d)
      printf '#!/bin/sh\nprintf "%%s\\n" "$MUSE_SUDO_PASS"\n' > "$_ASKPASS_DIR/ap"
      chmod 700 "$_ASKPASS_DIR/ap"
    fi
    SUDO_ASKPASS="$_ASKPASS_DIR/ap" sudo -A "$@"
  else
    sudo "$@"     # 交互场景：让 sudo 自己在终端要密码
  fi
}
CAM_DEV=/dev/video50
CSI_JSON=/usr/share/camera_json/csi3_camera_detect.json     # 摄像头接 CSI1/CSI2 时改这里
SVIVI_JSON=/root/svivi_cam1.json
MODEL_DIR="$HERE/models"
MODEL_URL="${MODEL_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx}"
OPSET="${OPSET:-17}"
DO_SERVICE=0; SKIP_CAMERA=0; NO_MODEL=0
for a in "$@"; do case "$a" in
  --service) DO_SERVICE=1;; --skip-camera) SKIP_CAMERA=1;; --no-model) NO_MODEL=1;;
esac; done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m[!!]\033[0m %s\n' "$*"; }
bad()  { printf '    \033[31m[no]\033[0m %s\n' "$*"; }

step "0. 平台确认"
uname -a
[ "$(uname -m)" = "riscv64" ] || warn "当前不是 riscv64：本脚本只针对 K1 板端，在 x86 上跑无意义"
df -h "$HERE" | tail -1

step "1. 系统依赖（opencv / onnxruntime + SpaceMIT EP / flask / onnx / gcc）"
# 全走 apt：Bianbu 是 PEP668  externally-managed 环境，pip 直装会被拦
PKGS="python3-opencv python3-numpy python3-flask python3-onnx python3-pip gcc libopencv-dev onnxruntime python3-spacemit-ort"
MISS=""
for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISS="$MISS $p"; done
if [ -z "$MISS" ]; then ok "全部已安装"; else
  warn "待安装:$MISS"
  SUDO apt-get update -qq >/dev/null 2>&1
  SUDO apt-get install -y --no-install-recommends $PKGS 2>&1 | tail -3 \
    || warn "apt 安装失败（网络/源问题？），后面几步可能不通过"
fi
python3 - <<'PY'
import importlib
for m in ("cv2", "numpy", "flask", "onnx", "onnxruntime", "spacemit_ort"):
    try:
        mod = importlib.import_module(m)
        print("    [ok] %-13s %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:
        print("    [no] %-13s %r" % (m, e))
PY

step "2. 编译 k1cam（V4L2 虚拟相机 -> NV12 帧流）"
if gcc -O2 -Wall -o "$HERE/cam/k1cam" "$HERE/cam/k1cam.c" 2>"$HERE/cam/build.log"; then
  ok "生成 $HERE/cam/k1cam（不链接 libv4l，只用内核 V4L2 + dma_heap 头文件）"
else
  bad "编译失败："; tail -8 "$HERE/cam/build.log"
fi

step "3. 相机配置（OV5647 @ CSI -> /dev/video50）"
if [ "$SKIP_CAMERA" = 1 ]; then warn "按参数跳过"; else
  [ -e "$CAM_DEV" ] || bad "$CAM_DEV 不存在：查排线方向 / camera 驱动"
  if SUDO grep -q '"use_v4l"' "$SVIVI_JSON" 2>/dev/null; then
    ok "$SVIVI_JSON 已含 use_v4l"
  else
    command -v cam-test >/dev/null || bad "板上没有 cam-test，无法自动探测 sensor，请按官方文档手动配置"
    if [ -f "$CSI_JSON" ] && command -v cam-test >/dev/null; then
      warn "用 $CSI_JSON 探测 sensor（约 20s，会占住 CSI 口）"
      SUDO pkill -9 -f cam-test 2>/dev/null; sleep 1
      SUDO timeout 60 cam-test "$CSI_JSON" >/tmp/camtest.log 2>&1
      grep -oE "detect [a-z0-9_]+ sensors in csi[0-9]: success" /tmp/camtest.log | tail -1 | sed 's/^/      /'
      AUTO=$(ls -t /tmp/csi*_camera_auto.json 2>/dev/null | head -1)
      if [ -n "${AUTO:-}" ]; then
        SUDO cp "$AUTO" "$SVIVI_JSON"
        # 在顶层第一个 { 之后插 "use_v4l": 1,  注意必须带逗号，否则 svivi 解析失败
        SUDO python3 - "$SVIVI_JSON" <<'PY'
import json, sys
p = sys.argv[1]
txt = open(p, encoding="utf-8", errors="replace").read()
d = json.loads(txt)
if isinstance(d, list): d = d[0]
if "use_v4l" not in d:
    d = {"use_v4l": 1, **d}
open(p, "w", encoding="utf-8").write(json.dumps(d, indent=2, ensure_ascii=False))
print("      use_v4l 写入完成")
PY
        SUDO python3 -m json.tool "$SVIVI_JSON" >/dev/null 2>&1 && ok "JSON 合法" || bad "JSON 解析失败，手工检查 $SVIVI_JSON"
      else
        bad "cam-test 没生成 /tmp/csi*_camera_auto.json，见 /tmp/camtest.log"
      fi
    else
      warn "$CSI_JSON 不存在；摄像头若接 CSI1/CSI2，请把脚本里 csi3 换成对应编号重跑"
    fi
  fi
  if [ -f "$SVIVI_JSON" ]; then
    warn "起 cam-test 并校验出图内容"
    MUSE_SUDO_PASS="${MUSE_SUDO_PASS:-}" "$HERE/cam-start.sh" || \
      warn "出图校验没通过，按 docs/k1-mipi-camera.md 的排查清单处理"
  fi
fi

step "4. 模型（官方 float ONNX -> opset$OPSET）"
MODEL="$MODEL_DIR/yolo11n-seg_opset$OPSET.onnx"
if [ "$NO_MODEL" = 1 ]; then
  warn "按参数跳过下载"
else
  mkdir -p "$MODEL_DIR"
  RAW="$MODEL_DIR/$(basename "$MODEL_URL")"
  [ -f "$RAW" ] || { warn "下载 $MODEL_URL"; curl -L --max-time 900 -o "$RAW" "$MODEL_URL" || bad "下载失败（换镜像/代理后重跑）"; }
  if [ -f "$MODEL_DIR/yolo11n-seg_opset$OPSET.onnx" ]; then
    ok "已存在 $MODEL"
  elif [ -f "$RAW" ]; then
    # ultralytics 导出的 yolo11n-seg.onnx 是 opset 22，而板上 onnxruntime 1.18.1 的官方
    # schema 只到 opset 21，不降级连会话都建不起来（报错见 docs/k1-aipu-onnxruntime.md §4）。
    # 这里只改 opset 版本号：YOLO 图里用到的算子(Conv/Relu/Add/Mul/Concat/Resize/
    # Transpose/Reshape/Split/Softmax/Sigmoid...) 在 13~22 语义一致，实测 14/15/17/18 结果相同。
    # 注意：降 opset 与"能不能上智算核"无关，不要指望它提速。
    python3 - "$RAW" "$OPSET" <<'PY' || true
import os, sys, onnx
src, ver = sys.argv[1], int(sys.argv[2])
m = onnx.load(src)
for o in m.opset_import:
    if o.domain in ("", "ai.onnx"): o.version = ver
dst = src.replace(".onnx", "_opset%d.onnx" % ver)
onnx.save(m, dst)
print("      ->", dst, os.path.getsize(dst), "bytes")
PY
    [ -s "$MODEL" ] || bad "opset 降级失败"
  fi
fi
MODEL=""
for c in "$MODEL_DIR"/yolo11n-seg_opset*.onnx "$MODEL_DIR"/yolo11n-seg*.onnx; do [ -f "$c" ] && MODEL="$c" && break; done
[ -n "$MODEL" ] && ok "使用模型 $MODEL ($(stat -c %s "$MODEL") bytes)" || warn "没有可用模型，第 5 步会跳过"

step "5. 自检：单帧推理（用合成图，只验流程不验精度）"
if [ -n "$MODEL" ]; then
  python3 - "$HERE/cam/selftest.jpg" <<'PY'
import numpy as np, cv2, sys
rng = np.random.default_rng(7)
img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (80, 120), (420, 400), (20, 200, 20), -1)   # 一个"物体"
cv2.imwrite(sys.argv[1], img)
PY
  LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1 python3 "$HERE/web/yolo_seg.py" \
      --model "$MODEL" --image "$HERE/cam/selftest.jpg" --runs 3 2>&1 | tail -6 \
    || bad "单帧推理失败，先看上面的报错"
  rm -f "$HERE/cam/selftest.jpg"
else
  warn "跳过"
fi

step "6. systemd 服务（可选）"
if [ "$DO_SERVICE" = 1 ] && [ -z "$MODEL" ]; then
  bad "没有可用模型，跳过服务安装（先不带 --no-model 重跑，或手工把模型放进 models/）"
  DO_SERVICE=0
fi
if [ "$DO_SERVICE" = 1 ]; then
  SUDO tee /etc/systemd/system/muse-camtest.service >/dev/null <<EOF
[Unit]
Description=MUSE Pi Pro camera pipeline (cam-test / svivi virtual camera)
After=local-fs.target

[Service]
ExecStart=/usr/bin/cam-test $SVIVI_JSON
Restart=always
RestartSec=3
EOF
  SUDO tee /etc/systemd/system/muse-yolo11seg-web.service >/dev/null <<EOF
[Unit]
Description=MUSE Pi Pro YOLO11-seg realtime segmentation web
After=muse-camtest.service
Requires=muse-camtest.service

[Service]
Environment=LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1
WorkingDirectory=$HERE/web
ExecStart=/usr/bin/python3 $HERE/web/server.py --model $MODEL --cam $HERE/cam/k1cam --port 8000
Restart=on-failure
RestartSec=5
EOF
  SUDO systemctl daemon-reload && SUDO systemctl enable --now muse-camtest.service && SUDO systemctl enable --now muse-yolo11seg-web.service
  sleep 8
  SUDO systemctl --no-pager --full status muse-yolo11seg-web.service 2>&1 | head -8 | sed 's/^/    /'
  warn "浏览器打开 http://<板子IP>:8000"
else
  warn "跳过（要开机自启加 --service）；手动启动：$HERE/run-web.sh"
fi

printf '\n完成。启动网页： %s/run-web.sh   然后浏览器访问 http://<板子IP>:8000\n' "$HERE"
