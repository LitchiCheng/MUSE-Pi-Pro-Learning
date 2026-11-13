#!/usr/bin/env bash
# 启动 web（默认走摄像头）。常用覆盖：
#   PORT=8080 ./run-web.sh
#   MODEL=../models/yolo11n-seg_opset17.onnx PREFER=cpu ./run-web.sh
#   STILL=$HOME/work/pics/bus.jpg ./run-web.sh     无光/没摄像头时用静态图演示分割链路
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ "$(uname -m)" = "riscv64" ] && [ -e /usr/lib/libonnxruntime.so.1.18.1 ] \
  && export LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1     # 不预加载这份，SpaceMIT EP 会静默回退 CPU

MODEL=${MODEL:-$(ls "$HERE"/models/yolo11n-seg_opset*.onnx "$HERE"/models/yolo11n-seg*.onnx 2>/dev/null | head -1)}
[ -n "$MODEL" ] || { echo "[no] models/ 下没有 yolo11n-seg*.onnx，先跑 setup-board.sh"; exit 1; }

ARGS=(--model "$MODEL" --cam "$HERE/cam/k1cam" --port "${PORT:-8000}")
[ -n "${WIDTH:-}" ]  && ARGS+=(--width "$WIDTH" --height "${HEIGHT:-$WIDTH}")
[ -n "${CONF:-}" ]   && ARGS+=(--conf "$CONF")
[ -n "${PREFER:-}" ] && ARGS+=(--prefer "$PREFER")
[ -n "${STILL:-}" ]  && ARGS+=(--still "$STILL")
[ -n "${VIDEO:-}" ]  && ARGS+=(--video "$VIDEO")

# server.py 自己会 re-exec 补 LD_PRELOAD，这里直接前台运行；要后台就加 nohup
exec python3 "$HERE/web/server.py" "${ARGS[@]}"
