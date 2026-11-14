#!/usr/bin/env bash
# 启动 web（默认走摄像头）。常用覆盖：
#   PORT=8080 ./run-web.sh
#   MODEL=../models/yolo11n-seg_opset17.onnx PREFER=cpu ./run-web.sh
#   STILL=$HOME/work/pics/bus.jpg ./run-web.sh     无光/没摄像头时用静态图演示分割链路
#   ROTATE=180 ./run-web.sh                          模组装反（画面上下颠倒）时纠偏
#   MODEL=models/yolov8n-seg.q.onnx PREFER=spacemit ./run-web.sh
#       官方 int8 走智算核，单帧 ~408ms（float 的 3 倍快）。它崩不崩几乎只取决于
#       "ORT 线程池有没有和 EP 线程数配平"，见 yolo_seg._tie_intra 的注释；
#       常驻演示请配 ./supervise-web.sh 兜底，且同一时刻只跑一个实例
set -uo pipefail
shopt -s nullglob
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ "$(uname -m)" = "riscv64" ] && [ -e /usr/lib/libonnxruntime.so.1.18.1 ] \
  && export LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1     # 不预加载这份，SpaceMIT EP 会静默回退 CPU

# 模型选择：优先 opset17 版（板端 onnxruntime 1.18.1 能吃），再退到其它 seg 模型。
# 注意不能简单 ls yolo11n-seg*.onnx | head -1 —— 那样会先选中 ultralytics 原始导出的
# yolo11n-seg.onnx(opset22)，onnxruntime 1.18.1 的 schema 只到 opset21，直接报
# "ValidateOpsetForDomain ... Opset 22 is under development"。
pick_model(){
  local pat
  # 默认永远选 float yolo11n-seg：它能连跑几小时。
  # 官方 int8 *.q.onnx 的 13 输出(DFL)解码已支持、单帧快 3 倍，但 SpaceMIT EP 1.2.2
  # 在并发采集下几十秒就 SIGSEGV（见 docs/k1-aipu-onnxruntime.md §7），所以不进默认列表，
  # 要试就显式给 MODEL=models/yolov8n-seg.q.onnx PREFER=spacemit（最好配 supervise-web.sh）。
  for pat in 'yolo11n-seg_opset17.onnx' 'yolo11n-seg_opset*.onnx' '*_qdq*.onnx' '*_qlin*.onnx'; do
    for f in "$HERE"/models/$pat; do
      [ -f "$f" ] && { printf '%s\n' "$f"; return 0; }
    done
  done
  return 1
}
MODEL=${MODEL:-$(pick_model)}
[ -n "$MODEL" ] || { echo "[no] models/ 下没有可用的 yolo11n-seg*.onnx，先跑 setup-board.sh"; exit 1; }

export EP_THREADS=${EP_THREADS:-4}   # SPACEMIT_EP_INTRA_THREAD_NUM；只有 int8 模型吃得到
# ORT 线程池不在这里定：yolo_seg._tie_intra() 会按模型是不是 int8 自动收到和 EP_THREADS 一致
# （float 模型开满 8 核）。要手工覆盖就 INTRA_THREADS=2 ./run-web.sh。
ARGS=(--model "$MODEL" --cam "$HERE/cam/k1cam" --port "${PORT:-8000}")
ARGS+=(--rotate "${ROTATE:-0}")
[ -n "${WIDTH:-}" ]  && ARGS+=(--width "$WIDTH" --height "${HEIGHT:-$WIDTH}")
[ -n "${CONF:-}" ]   && ARGS+=(--conf "$CONF")
[ -n "${PREFER:-}" ] && ARGS+=(--prefer "$PREFER")
[ -n "${STILL:-}" ]  && ARGS+=(--still "$STILL")
[ -n "${VIDEO:-}" ]  && ARGS+=(--video "$VIDEO")

# server.py 自己会 re-exec 补 LD_PRELOAD，这里直接前台运行；要后台就加 nohup
exec python3 "$HERE/web/server.py" "${ARGS[@]}"
