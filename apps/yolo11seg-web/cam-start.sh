#!/usr/bin/env bash
# 点亮 K1 上的 MIPI 虚拟相机：校验 /dev/video50 能出"真帧"，并保证 cam-test 只有一个实例。
#
#   ./cam-start.sh                 清理 + 采帧校验（默认 24s 窗口）
#   ./cam-start.sh --win 40        加长校验窗口（等价于裸数字 `./cam-start.sh 40`）
#   ./cam-start.sh --json <path>   指定 svivi json（默认 /root/svivi_cam1.json）
#   W=1920 H=1080 ./cam-start.sh   换分辨率校验
#
# 机制（全部板上实测，证据见 docs/k1-mipi-camera.md）：
#   * open("/dev/video50") 时内核用 call_usermodehelper 自动拉起一发 cam-test
#     （ps 里 PPid=2），它需要 1~2s 才向内核注册格式，这期间客户端的 S_FMT 直接返回
#     EAGAIN。所以要做的是"重试 ioctl"（k1cam -w，默认 20s），不是提前起常驻服务。
#   * 格式注册成功后帧流是持续的（实测 72s 内 cam-test 数恒为 1、无重启、无黑帧）。
#     所以 supervisor 循环 / 开机自启 cam-test 的 systemd 服务都是多余的。
#   * 多于一发 cam-test 会互相抢 pipeline → 恒黑帧（Y mean≈1, std≈0.5, UV=128）。
#     本脚本因此只做一件事：在"确认没有采集端在推流"时把多余实例清干净。
#   * 铁律：不要在采集端已经 open + STREAMON 的时候 pkill cam-test。实测那样帧数会
#     立刻冻结，k1cam 进入 kill -9 也杀不掉的 D 状态，只能重启板子。
#   * 但 open() 的自拉不是 100% 成功：cam-test 自己也会 "netlink_recv ... EAGAIN" 退出，
#     于是采集端只能一直 EAGAIN 到超时（表现 = 一帧都没有）。所以本脚本带一个"只在
#     窗口内确实 0 帧且 cam-test 数为 0"时才补一发的一次性补救（最多 3 次），
#     绝不是常驻 supervisor —— 常驻循环一旦和内核自拉的那发并存就是恒黑帧。
set -uo pipefail

JSON="${JSON:-/root/svivi_cam1.json}"
DEV="${DEV:-/dev/video50}"
W="${W:-1280}"; H="${H:-720}"
WIN="${CHECK_WIN:-24}"
K1CAM="$(cd "$(dirname "${BASH_SOURCE[0]}")/cam" && pwd)/k1cam"
for a in "$@"; do
  case "$a" in
    --win) _NEXT=win ;;
    --json) _NEXT=json ;;
    *) if [ "${_NEXT:-}" = win ]; then WIN="$a"; _NEXT=
       elif [ "${_NEXT:-}" = json ]; then JSON="$a"; _NEXT=
       elif [[ "$a" == --* ]]; then bad "未知参数 $a"; exit 2
       elif [[ "$a" =~ ^[0-9]+$ ]]; then WIN="$a"          # 裸数字 = 校验窗口秒数
       else JSON="$a"; fi ;;                              # 其它位置参数 = svivi json
  esac
done

# ---- sudo helper：优先复用凭证缓存；有 MUSE_SUDO_PASS 就走 askpass（保持 stdin 干净，
#      这样 SUDO tee 这类从 stdin 读数据的写法不会被密码管道吃掉）----
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
say() { printf '%s\n' "$*"; }
bad() { printf '\033[31m[no]\033[0m %s\n' "$*"; }
ok()  { printf '\033[32m[ok]\033[0m %s\n' "$*"; }

# pgrep -c 在"没匹配"时也打印 0 且退码 1，所以不要写 `|| echo 0`（会打印两行 0，
# [ 0\n0 -gt 1 ] 直接报 integer expression expected）。
# 必须用 -x：-f 会把"执行本行的 shell 自己"（cmdline 含 cam-test）也算进去。
n_run() { local c; c=$(pgrep -c -x cam-test 2>/dev/null); printf '%s' "${c:-0}"; }

if [ ! -e "$DEV" ]; then bad "$DEV 不存在：先确认排线方向与 camera 驱动（见 docs/k1-mipi-camera.md）"; exit 1; fi
if [ ! -x "$K1CAM" ]; then bad "找不到 $K1CAM，先跑 setup-board.sh 第 2 步"; exit 1; fi

# 采集端独占：别的 k1cam（上次没退干净的 web/探针）在跑的话，本脚本的 reader 会读到
# 0 帧，看起来像"相机坏了"；而且此刻绝对不能动 cam-test。
OTHER=$(pgrep -x k1cam | tr '\n' ' ')
if [ -n "${OTHER// /}" ]; then
  bad "已有 k1cam 在独占 $DEV（pid: $OTHER）—— 先停掉它（例如 web 服务：pkill -f server[.]py）再校验；"
  bad "这时候千万别 pkill cam-test：帧会冻结且 k1cam 会卡成 D 状态，只能重启板子"
  exit 6
fi

# 到这里设备一定空闲，清理残留 cam-test（含上一版脚本留下的 supervisor 循环）是安全的。
SUPS=$( { pgrep -f cam-test 2>/dev/null; pgrep -f k1cam_sup 2>/dev/null; } | sort -n -u | \
        while read -r p; do c=$(cat "/proc/$p/comm" 2>/dev/null) || continue
                         [ "$c" = cam-test ] || printf '%s\n' "$p"; done | tr '\n' ' ')
if [ "$(n_run)" -gt 0 ] || [ -n "${SUPS// /}" ]; then
  say "设备空闲，清理残留：cam-test $(n_run) 个 / supervisor ${SUPS:-无}（随后由 open() 触发内核自拉一发）"
  for p in $SUPS; do SUDO kill -9 "$p" 2>/dev/null; done
  SUDO pkill -9 -x cam-test 2>/dev/null
  sleep 2
fi
# 遗留的自启服务会在重启后又变成多实例，一并提醒（本脚本不再安装任何服务）
if systemctl list-unit-files 2>/dev/null | grep -q '^muse-camtest[.]service'; then
  bad "检测到历史遗留的 muse-camtest.service —— 它会和内核自拉的 cam-test 并存造成恒黑帧，"
  bad "建议：sudo systemctl disable --now muse-camtest.service && sudo rm /etc/systemd/system/muse-camtest.service"
fi

# 出图校验：不看"有没有字节"，看像素。三种"假帧"要分开（判据来源见 docs）：
#   常数 0            = buffer 根本没被写（dma heap_flags 用错，k1cam 固定 0xf0）
#   Y≈1/std≈0.5,UV=128 = 数字黑帧：有第二个 cam-test 在抢 pipeline
#   UV 方差巨大        = 在读未初始化内存
# reader 常驻窗口 WIN 秒；open() 会触发内核拉 cam-test，k1cam 内部对 EAGAIN 重试，
# 所以正常情况下首张真帧应在 1~3s 内出现。
say "采帧校验 ${WIN}s（reader 常驻，靠 open() 触发内核自拉 cam-test，不做任何 pkill）..."
export K1CAM_BIN="$K1CAM" K1DEV="$DEV" K1W="$W" K1H="$H" K1WIN="$WIN" K1FLAG=/tmp/.k1cam_check_frames
: > "$K1FLAG"
# 一次性补救：reader 已经 open + STREAMON（这才是 cam-test 握手能成功的前提），
# 但 4s 过去了还是一帧没有、且当前没有 cam-test 在跑 -> 补拉一发。
# 条件里"0 帧"是防着在正常推流时误拉第二发（那会变成恒黑帧）。
( for i in 1 2 3; do
    sleep 4
    f=$(head -1 "$K1FLAG" 2>/dev/null); case "${f:-0}" in *[!0-9]*|"") f=0;; esac
    [ "$f" -gt 0 ] && break     # 已经出帧了就绝对不要再拉第二发
    [ "$(n_run)" = 0 ] || break
    say "[fix] 前 ${i} 次：0 帧且没有 cam-test 在跑（内核自拉那发握手失败），补拉一发"
    SUDO sh -c "setsid nohup /usr/bin/cam-test $JSON >>/tmp/svivi_cam1.log 2>&1 </dev/null &"
  done ) &
WATCH=$!
timeout "$((WIN + 25))" python3 - <<'PYCHK'
import os, select, subprocess, time
import numpy as np
W, H = int(os.environ["K1W"]), int(os.environ["K1H"])
n = W * H * 3 // 2
proc = subprocess.Popen([os.environ["K1CAM_BIN"], "-d", os.environ["K1DEV"],
                         "-W", str(W), "-H", str(H), "-f", "NV12", "-n", "0", "-w", "20"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
fd = proc.stdout.fileno()
os.set_blocking(fd, False)          # 关键：驱动停帧时阻塞 read() 会永久挂住，连汇总都打不出来
t0, win = time.time(), float(os.environ.get("K1WIN", "24"))
buf = bytearray()
frames = nreal = nblack = nzero = 0
mean = std = uvstd = 0.0
uv = np.zeros(1, np.int32)
first_real = last_black = None
try:
    eof = False
    while time.time() - t0 < win and not eof:
        for chunk in select.select([fd], [], [], 0.5)[0]:
            try:
                r = os.read(chunk, n - len(buf))
            except BlockingIOError:
                continue             # EAGAIN：暂时没有数据，回去看时钟
            if r == b"":             # EOF：k1cam 自己退出了（设备被占 / 分配失败 / 重试超时）
                err = proc.stderr.read().decode("utf-8", "replace").strip()
                print("[no] k1cam 退出: %s" % (err.splitlines()[-1] if err else "(无 stderr)"))
                eof = True
                break
            buf += r
            if len(buf) < n:
                continue
            f = np.frombuffer(bytes(buf), np.uint8); buf = bytearray()
            y, uv = f[: W * H].astype(np.int32), f[W * H:]
            frames += 1
            mean, std = float(y.mean()), float(y.std())
            if frames % 15 == 1:
                try:
                    open(os.environ.get("K1FLAG", "/tmp/.k1cam_check_frames"), "w").write("%d\n" % frames)
                except Exception:
                    pass
            if mean < 0.05 and std < 0.05:
                nzero += 1                              # 常数 0：buffer 根本没被写
            elif mean < 4 and std < 2 and abs(float(uv[0::2].mean()) - 128) < 8:
                nblack += 1; last_black = time.time() - t0   # 数字黑帧：pipeline 被抢
            else:
                nreal += 1
                if first_real is None:
                    first_real = time.time() - t0
                    print("  首张真帧 t=%.1fs  Y mean=%.1f std=%.1f" % (first_real, mean, std))
            if frames % 45 == 0:
                print("  ... %d 帧 真%d 黑%d 零%d (当前 Y mean=%.1f std=%.1f)"
                      % (frames, nreal, nblack, nzero, mean, std))
finally:
    proc.terminate()
    try:
        proc.wait(3)
    except Exception:
        proc.kill()
print("frames=%d | 真帧=%d 黑帧=%d 零帧=%d | 首张真图 t=%s | 最后一次黑帧 t=%s"
      % (frames, nreal, nblack, nzero,
         "-" if first_real is None else "%.1fs" % first_real,
         "-" if last_black is None else "%.1fs" % last_black))
if frames == 0:
    print("[no] 一帧都没读到：确认没有第二个采集端在跑，看 dmesg 与 /tmp/svivi_cam1.log"); raise SystemExit(2)
if nreal == 0:
    if nzero == frames:
        print("[no] 全是常数 0 帧：dma buffer 没被写。检查 cam-test 是否在跑、"
              "k1cam 是否用 heap_flags=0xf0 申请 buffer"); raise SystemExit(3)
    print("[!!] 窗口内全是数字黑帧(Y≈1/std≈0.5)：多半是又有第二发 cam-test 起来了。"
          "用 ps -eo pid,ppid,etime,comm | grep cam-test 数一下，清到只剩 1 发；"
          "不要 pkill 正在推流的那发（会把采集端钉成 D 状态）"); raise SystemExit(4)
if float(uv.std()) > 40 and nreal < frames * 0.2:
    print("[!!] 色度抖动大、真帧少：像在读未初始化内存，检查 dma_heap 分配方式"); raise SystemExit(4)
if nreal < frames * 0.5:
    print("[!!] 只有 %.0f%% 的帧是真图：pipeline 不稳，最常见是有第二个 cam-test/supervisor 在抢"
          % (100.0 * nreal / frames)); raise SystemExit(4)
print("[ok] 出图正常：%.0f%% 为真帧（%.1f fps，无需任何 supervisor）"
      % (100.0 * nreal / frames, frames / max(1e-6, time.time() - t0)))
PYCHK
RC=$?
wait $WATCH 2>/dev/null
rm -f "$K1FLAG"
exit $RC
