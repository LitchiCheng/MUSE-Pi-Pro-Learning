#!/usr/bin/env bash
# 常驻跑 web 的看门狗：进程退出（含 SIGSEGV）就 2s 后重启，并记录存活时长。
#
#   ./supervise-web.sh                 # 默认 float 模型（稳定，~0.5 fps）
#   MODEL=models/yolov8n-seg.q.onnx PREFER=spacemit ./supervise-web.sh   # int8 走智算核
#
# 为什么需要它：SpaceMIT EP 1.2.2 跑 int8 量化模型会在 sess.run 里随机 SIGSEGV，
# 而且没有任何 Python 异常可捕获（进程直接消失）。线程数配平后实测能连跑 32 min 不崩，
# 但样本太少不敢当结论，详见 docs/k1-aipu-onnxruntime.md §7。
# 这里只在**web 自身**外面套一层循环，不碰 cam-test（自启 cam-test 已判定有害，
# 内核会在 open(/dev/video50) 时自己把流拉起来）。
#
# k1cam 是 run-web.sh 的子进程、stdout 是本管道的读端：python 挂掉后管道关闭，
# k1cam 会拿到 EPIPE 自己退出，所以不会留下 D 状态进程霸占 /dev/video50。
# （实测：q 模型崩 3 次后 ps 里没有残留 k1cam，重启后 8s 内恢复出图。）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=${LOG:-$HERE/logs/web-supervise.log}
STALL_KILL=${STALL_KILL:-120}     # 秒：推理 0 增长则主动杀掉重开（比僵死着强）；0=关
mkdir -p "$(dirname "$LOG")"

n=0
while :; do
  n=$((n + 1))
  t0=$(date +%s)
  echo "$(date '+%F %T') [watchdog] 第 $n 次启动 web" >> "$LOG"
  ( cd "$HERE" && bash run-web.sh >> "$LOG" 2>&1 ) &   # 子 shell 独立 pgid，好收拾
  pid=$!
  last_runs=-1; same=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
    [ "$STALL_KILL" = "0" ] && continue
    # 只看门"进程活着但不再出结果"的情况：/stats 的 infer.runs 连续 N 轮不变 => 卡死
    r=$(curl -s --max-time 5 "http://127.0.0.1:${PORT:-8000}/stats" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["infer"]["runs"])' 2>/dev/null)
    [ -z "$r" ] && continue                     # /stats 没答上不算卡（可能在启动中）
    if [ "$r" = "$last_runs" ]; then
      same=$((same + 1))
      if [ "$same" -ge $((STALL_KILL / 10)) ]; then
        echo "$(date '+%F %T') [watchdog] ${STALL_KILL}s 内 runs 恒为 $r，判定卡死，杀掉重开" >> "$LOG"
        kill -TERM "$pid" 2>/dev/null; sleep 3; kill -KILL "$pid" 2>/dev/null
        break
      fi
    else
      same=0; last_runs=$r
    fi
  done
  alive=$(( $(date +%s) - t0 ))
  echo "$(date '+%F %T') [watchdog] web 退出（存活 ${alive}s，累计第 $n 次）" >> "$LOG"
  pgrep -x k1cam >/dev/null 2>&1 && { echo "$(date '+%F %T') [watchdog] 残留 k1cam，清掉" >> "$LOG"; pkill -x k1cam; }
  sleep 2
done
