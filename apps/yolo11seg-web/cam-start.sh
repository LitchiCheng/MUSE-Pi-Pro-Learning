#!/usr/bin/env bash
# 点亮 K1 上的 MIPI 虚拟相机：保证"有且只有一个" cam-test 在跑，并验证真的出图。
#
#   ./cam-start.sh                 启动/检查 + 出图校验
#   ./cam-start.sh --install        额外安装 systemd 服务 muse-camtest.service（开机自启）
#   ./cam-start.sh --json <path>    指定 svivi json（默认 /root/svivi_cam1.json）
#
# 为什么需要它（都是实测踩出来的）：
#   * 板子重启后没有任何服务会启动 cam-test，/dev/video50 虽然存在但读不到真帧；
#   * cam-test 多开会抢 tuning server 端口(12218)，后起的实例会 "wait kernel stream
#     on ... Resource temporarily unavailable" 然后自己 power off，结果就是
#     采到的帧全是 0 或全是噪声，看起来像"驱动坏了"；
#   * 官方 camera.service 只做 chmod，并不负责拉 cam-test。
set -uo pipefail

JSON="${JSON:-/root/svivi_cam1.json}"
DEV="${DEV:-/dev/video50}"
W="${W:-1280}"; H="${H:-720}"
K1CAM="$(cd "$(dirname "${BASH_SOURCE[0]}")/cam" && pwd)/k1cam"
INSTALL=0
for a in "$@"; do case "$a" in --install) INSTALL=1;; --json) :;; *) [[ "$a" == /* || "$a" == .* ]] || { JSON="$a"; };; esac; done

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
say() { printf '%s\n' "$*"; }
bad() { printf '\033[31m[no]\033[0m %s\n' "$*"; }
ok()  { printf '\033[32m[ok]\033[0m %s\n' "$*"; }

if [ ! -e "$DEV" ]; then bad "$DEV 不存在：先确认排线方向与 camera 驱动（见 docs/k1-mipi-camera.md）"; exit 1; fi
if [ ! -x "$K1CAM" ]; then bad "找不到 $K1CAM，先跑 setup-board.sh 第 2 步"; exit 1; fi

N=$(pgrep -fc 'cam-test' 2>/dev/null || echo 0)
if [ "${N:-0}" -gt 1 ]; then
  bad "检测到 $N 个 cam-test 实例（会互相抢 pipeline），全部杀掉后重启一个"
  SUDO pkill -9 -f cam-test; sleep 2
elif [ "${N:-0}" = "1" ]; then
  ok "已有 1 个 cam-test 在跑"
fi
if [ "$(pgrep -fc 'cam-test' 2>/dev/null || echo 0)" -lt 1 ]; then
  SUDO sh -c "setsid nohup /usr/bin/cam-test $JSON > /tmp/svivi_cam1.log 2>&1 < /dev/null &"
  say "已启动 cam-test（日志 /tmp/svivi_cam1.log），等 8s 让 sensor 出流"
  sleep 8
fi

# 出图校验：不看"有没有字节"，看像素是否合理。
# 真帧的特征：UV 平面均值 ~128（灰/黑的色度中性值），Y 平面相邻像素差 < 10。
# 全 0 = 驱动没写进我们的 buffer；UV 方差巨大 = 读到的是没初始化的内存（噪声）。
say "采 8 帧 ${W}x${H} 做内容校验 ..."
TMP=$(mktemp /tmp/k1cam_check.XXXXXX.nv12)
timeout 25 "$K1CAM" -d "$DEV" -W "$W" -H "$H" -f NV12 -n 8 > "$TMP" 2>/tmp/k1cam_check.log
python3 - "$TMP" "$W" "$H" <<'PY'
import sys, numpy as np
path, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
n = W * H * 3 // 2
d = np.fromfile(path, dtype=np.uint8)
fr = d.size // n
if fr < 1:
    print("[no] 没拿到任何整帧（%d 字节）；看 /tmp/k1cam_check.log" % d.size); sys.exit(2)
f = d[:n]; y = f[:W*H].reshape(H, W).astype(np.int32); uv = f[W*H:]
u, v = uv[0::2].astype(np.int32), uv[1::2].astype(np.int32)
print("frames=%d  Y mean=%.1f std=%.1f | U mean=%.1f  V mean=%.1f | 相邻像素差=%.2f"
      % (fr, y.mean(), y.std(), u.mean(), v.mean(), np.abs(np.diff(y, axis=1)).mean()))
if y.std() < 0.05 and uv.std() < 0.05:
    print("[no] 整帧是常数 0：驱动没写进 buffer。常见原因是 cam-test 多开或没跑，"
          "或 dma buffer 不是 heap_flags=0xf0 申请的"); sys.exit(3)
if abs(float(u.mean()) - 128) > 24 or float(uv.std()) > 40:
    print("[!!] 色度偏离中性 128 且抖动大：像在读未初始化内存（噪声帧），"
          "检查 dma_heap 分配方式 / 换一个分辨率"); sys.exit(4)
if y.mean() < 4:
    print("[ok] 链路通了，但画面极暗（Y 均值 %.1f）：环境没光或曝光太短，白天/开灯后重测" % y.mean())
else:
    print("[ok] 出图正常")
PY
RC=$?
SUDO rm -f "$TMP" 2>/dev/null
if [ "$INSTALL" = 1 ]; then
  SUDO tee /etc/systemd/system/muse-camtest.service >/dev/null <<EOF
[Unit]
Description=MUSE Pi Pro camera pipeline (cam-test / svivi virtual camera)
After=local-fs.target

[Service]
ExecStart=/usr/bin/cam-test $JSON
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
  SUDO systemctl daemon-reload && SUDO systemctl enable --now muse-camtest.service
  say "已安装 muse-camtest.service（开机自启 cam-test；注意本脚本手动起的实例会被它抢占，装了之后就别再手动起）"
fi
exit $RC
