#!/usr/bin/env bash
# 在 **PC 侧**（任意架构，不需要 torch）把 float yolo11-seg 量化成能上 K1 智算 kernel 的 int8 模型。
#
#   ./run-quant.sh                         # 用 models/yolo11n-seg_opset17.onnx，出 models/y11seg_qdq_actint8.onnx
#   SRC=../models/x.onnx OUT=/tmp/x8.onnx ./run-quant.sh
#   ./run-quant.sh --act uint8             # 复现"不上核"那一行（对照实验用）
#
# 判据（为什么必须 --act int8）见仓库 docs/k1-aipu-onnxruntime.md §3.4；
# 三个必要参数一个都不能省：
#   --act int8      EP 的 int8 kernel 只注册了 tensor(int8) 激活，ORT 默认的 uint8 会被判不支持
#   --ops Conv      只量卷积；逐元素算子留给 EP 的 float kernel，精度最稳
#   --exclude /model.23/Concat_6   output0 一个张量跨 3 个数量级，量化它 = 检不出任何东西
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC=${SRC:-$HERE/../models/yolo11n-seg_opset17.onnx}
OUT=${OUT:-$HERE/../models/y11seg_qdq_actint8.onnx}
CORPUS=${CORPUS:-$HERE/../../corpus/*.jpg}
ACT=int8
[ "${1:-}" = "--act" ] && ACT=${2:-int8}

[ -f "$SRC" ] || { echo "[no] 找不到源模型 $SRC（先在板子上跑 setup-board.sh 或从 models/ 拷一份）"; exit 1; }
# 校准图必须是真的会遇到的画面：QuantizeLinear 的 scale 完全由这批图决定，
# 拿随机噪声当校准集 = 精度掉到检不出东西。没有就明确报错，别静默降级。
n_img=$(python3 -c 'import glob,os,sys;print(len([p for g in sys.argv[1].split(",") for p in glob.glob(g) if os.path.isfile(p)]))' "$CORPUS" 2>/dev/null || echo 0)
if [ "${n_img:-0}" -lt 8 ]; then
  echo "[no] 校准图太少（$n_img 张，来自 $CORPUS）"
  echo "     给一批真实场景图（20~50 张就够，覆盖你关心的画面），然后："
  echo "     CORPUS=/path/to/photos/*.jpg ./run-quant.sh"
  echo "     校准图决定每层的 scale；不给真实画面，quant_qdq.py 会退化成随机噪声凑数（误差没量化过，但 §3.3 的'值域不对就检不出东西'是实测过的坑）。"
  exit 1
fi
python3 -c "import onnx, onnxruntime" 2>/dev/null \
  || { echo "[no] 缺依赖：pip install onnx onnxruntime numpy opencv-python-headless"; exit 1; }

mkdir -p "$(dirname "$OUT")"
echo "==> 量化 $SRC  ->  $OUT  (act=$ACT)"
python3 "$HERE/quant_qdq.py" "$SRC" "$OUT" \
        --images "$CORPUS" --n 36 --ops Conv --exclude /model.23/Concat_6 \
        --format qdq --act "$ACT" || exit 1
rm -f "$OUT.pre.onnx"

echo
echo "==> PC 侧结构自检（Q/DQ 的 zero_point 类型必须是 INT8）"
python3 "$HERE/onnxlite.py" "$OUT" | sed -n '1,8p'
case "$ACT" in
  int8)  echo "    期望：Q/DQ 输入类型签名里 zp 是 INT8；上板 epcheck 的 Conv 类应 99/101 落 EP";;
  uint8) echo "    这一发是对照组：epcheck 会显示 Conv 类全在 CPU（和 float 一样不提速）";;
esac

echo
echo "==> 下一步（拷到板子上出账单 + 对精度）"
cat <<'EOF'
  rsync -a --exclude __pycache__ --exclude logs/ ./ <user>@<board>:work/yolo11seg-web/
  ssh <user>@<board> 'cd work/yolo11seg-web && ./epcheck/run-epcheck.sh models/y11seg_qdq_actint8.onnx'
  ssh <user>@<board> 'cd work/yolo11seg-web && LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1 \
      python3 web/yolo_seg.py --model models/y11seg_qdq_actint8.onnx --image ~/work/pics/bus.jpg'
EOF
