#!/usr/bin/env bash
# 在**板子上**跑：给 models/ 里的每个模型出一张"算子落在哪个 EP"的账单。
#
#   ./run-epcheck.sh                       # 默认查 float yolo11 + 官方 int8 v8 两份
#   ./run-epcheck.sh ../models/a.onnx ...  # 指定模型
#
# 判据与踩坑见 ../docs 的 k1-aipu-onnxruntime.md §1.2；这里只是把它变成一条命令。
# 不碰 /dev/video50，所以可以和正在跑的 web 并存 —— 但**耗时数字会被抢核污染**
# （实测同一份 int8：单实例 infer 283ms，和 float web 并存时 797~1055ms），
# 要看耗时请先停 web，本脚本只回答"落没落在智算核上"。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_LIB=${ORT_LIB:-/usr/lib/libonnxruntime.so.1.18.1}
WORK=${WORK:-/tmp/epcheck}
M=("$@")
if [ ${#M[@]} = 0 ]; then
  M=("$HERE/../models/yolo11n-seg_opset17.onnx" "$HERE/../models/yolov8n-seg.q.onnx")
fi
mkdir -p "$WORK"
LOGS=()
for m in "${M[@]}"; do
  [ -f "$m" ] || { echo "[skip] 没有 $m"; continue; }
  b=$(basename "$m")
  echo "==> $b"
  # LD_PRELOAD 不是可选项：不预加载 ORT 时 EP 注册得上但 int8 会静默回退 CPU（§1.1）
  LD_PRELOAD="$ORT_LIB" python3 "$HERE/ep.py" "$m" > "$WORK/$b.log" 2>&1
  echo "    $(grep -o 'PROVIDERS .*' "$WORK/$b.log" | tail -1)"
  LOGS+=("$WORK/$b.log")
done
echo
python3 "$HERE/place.py" "${LOGS[@]}"
