/* k1cam: SpacemiT K1 虚拟相机(/dev/video50, spacemit vivi, m2m + dmabuf)帧流工具
 *
 * 用法: k1cam [-d /dev/video50] [-W 1920] [-H 1080] [-f NV12] [-n 帧数(0=无限)]
 *            [-b 缓冲数] [-v] [--heap /dev/dma_heap/system] [--hf 0xf0]
 * 输出: 每帧原始 NV12 字节写 stdout(Y 平面 + UV 交错平面，连续排布)，日志走 stderr
 * 前提: /usr/bin/cam-test <svivi_cam1.json> 已在跑（它负责 sensor/ISP pipeline），
 *       且同一时刻只有它一个实例；见同目录 cam-start.sh
 * 参考: spacemit 官方 v4l2_test_spacemit 的 stateful_m2m + V4L2_MEMORY_DMABUF 路径
 *
 * 三处踩过的坑都在这个文件里，改动时注意：
 *   1) capture buffer 必须是 dma-buf，且要用 heap_flags=0xf0 申请（flags=0 拿到的
 *      buffer 驱动不会写入，画面全 0）；0xf0 返回的实际大小可能大于/小于请求值，
 *      必须按真实大小 mmap —— 见 dmabuf_alloc()
 *   2) 每个 plane 在 buffer 内的起点是它自己的 data_offset，不能累加偏移
 *   3) 重新 QBUF 会 EAGAIN（fd 非阻塞 + m2m 队列忙），要重试，否则几个 buffer
 *      用完流就停（表现为"只出 1~2 帧"）
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/time.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <linux/dma-heap.h>

#define NB 4
#define NP 2
static volatile int g_run = 1;
static void on_sig(int s){ (void)s; g_run = 0; }
static int xioctl(int fd, unsigned long req, void *arg){
  int r; do { r = ioctl(fd, req, arg); } while (r < 0 && errno == EINTR);
  return r;
}

static const char *g_heap_paths[2] = { "/dev/dma_heap/system", "/dev/dma_heap/linux,cma" };
/* heap_flags 表；两个槽位故意同为 0xf0 —— 实测 flags=0 的 buffer 驱动不写入(画面全 0)，
 * 所以不拿它做兜底。用 --hf 可以整体改成别的值做实验。 */
static unsigned g_heap_flags_list[2] = { 0xf0, 0xf0 };
static int g_nb = 4;

static int heap_open(const char *path, int *fd){
  *fd = open(path, O_RDWR | O_CLOEXEC);
  return *fd >= 0;
}

/* dma-buf 分配(K1 内核 6.6.63 的实测坑):
 *  1) heap_flags=0xf0(官方示例值)时, /dev/dma_heap/system 返回的 buffer 实际大小
 *     可能 *小于* 请求值(实测请求 1MB -> 得到 520192B), 按请求长度 mmap 直接 EINVAL;
 *  2) 同一参数连续调用有时会 EINVAL(与 page-pool / 前一次采集进程残留的引用有关),
 *     需要换 heap / 换 flags 重试。
 * 这里按 [system -> linux,cma] x [heap_flags 0xf0] 组合重试(最多 16 轮)，并且始终以
 * lseek 拿到的真实大小去 mmap。heap_flags=0 试过但不可用：buffer 能 mmap、能 DQBUF，
 * 驱动却不往里面写，画面全 0。 */
static int dmabuf_alloc(int fds[2], size_t need, int *fd_out, size_t *real_out){
  /* 只用 heap_flags=0xf0(官方要求): 实测 flags=0 拿到的 buffer 驱动不会写入, 画面全 0 */
  const unsigned *hf = g_heap_flags_list;
  for (int attempt = 0; attempt < 16; attempt++){
    for (int h = 0; h < 2; h++){
      if (fds[h] < 0) continue;
      for (int f = 0; f < 2; f++){
        struct dma_heap_allocation_data hd;
        memset(&hd, 0, sizeof hd);
        hd.len = need; hd.fd_flags = O_RDWR | O_CLOEXEC; hd.heap_flags = hf[f];
        if (xioctl(fds[h], DMA_HEAP_IOCTL_ALLOC, &hd)){
          if (attempt == 0 && f == 0 && h == 0) fprintf(stderr, "alloc retry: %s\n", strerror(errno));
          continue;
        }
        off_t real = lseek(hd.fd, 0, SEEK_END);
        if (real >= (off_t)need){ *fd_out = hd.fd; *real_out = (size_t)real; return 0; }
        close(hd.fd);   /* buffer 比要求的还小, 换一次/换一个 heap */
      }
    }
    usleep(30000);
  }
  fprintf(stderr, "dmabuf: 16 次重试仍拿不到 >= %zu 字节的 buffer\n", need);
  return -1;
}

/* EAGAIN/EBUSY 时重试到超时；其它 errno 立刻失败。等到点返回非 0。 */
static int ioctl_wait(int fd, unsigned long req, void *arg, const char *what, int timeout_s, FILE *log){
  struct timeval t0, t; gettimeofday(&t0, NULL);
  int tries = 0;
  for (;;){
    if (!xioctl(fd, req, arg)) return 0;
    if (errno != EAGAIN && errno != EBUSY && errno != ETIMEDOUT){ perror(what); return -1; }
    gettimeofday(&t, NULL);
    if ((t.tv_sec - t0.tv_sec) >= timeout_s){
      fprintf(log, "%s: 等 %ds 仍然 EAGAIN/EBUSY（cam-test 没起来？）\n", what, timeout_s);
      return -1;
    }
    if (++tries <= 3) fprintf(log, "  %s EAGAIN, 等 cam-test 注册格式... (%ds)\n", what, (int)(t.tv_sec - t0.tv_sec));
    usleep(200000);
  }
}

int main(int argc, char **argv){
  const char *dev = "/dev/video50";
  int W = 1280, H = 720, nframes = 0, verbose = 0, wait_s = 20;
  __u32 fourcc = V4L2_PIX_FMT_NV12;
  for (int i = 1; i < argc; i++){
    if (!strcmp(argv[i],"-d") && i+1 < argc) dev = argv[++i];
    else if (!strcmp(argv[i],"-W") && i+1 < argc) W = atoi(argv[++i]);
    else if (!strcmp(argv[i],"-H") && i+1 < argc) H = atoi(argv[++i]);
    else if (!strcmp(argv[i],"-n") && i+1 < argc) nframes = atoi(argv[++i]);
    else if (!strcmp(argv[i],"-w") && i+1 < argc) wait_s = atoi(argv[++i]);
    else if (!strcmp(argv[i],"-f") && i+1 < argc){
      const char *f = argv[++i];
      if (!strcasecmp(f,"NV12")) fourcc = V4L2_PIX_FMT_NV12;
      else if (!strcasecmp(f,"BGR3")) fourcc = V4L2_PIX_FMT_BGR24;
      else { fprintf(stderr,"bad format %s\n", f); return 1; }
    } else if (!strcmp(argv[i],"-v")) verbose = 1;
    else if (!strcmp(argv[i],"-b") && i+1 < argc) g_nb = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--heap") && i+1 < argc){ g_heap_paths[0] = argv[++i]; g_heap_paths[1] = ""; }
    else if (!strcmp(argv[i],"--hf") && i+1 < argc){ unsigned f = (unsigned)strtoul(argv[++i],0,0);
      g_heap_flags_list[0] = f; g_heap_flags_list[1] = f; }
    else { fprintf(stderr,"usage: %s [-d dev] [-W w] [-H h] [-f NV12|BGR3] [-n n] [-w wait_s] [-v]\n", argv[0]); return 1; }
  }
  signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
  FILE *log = stderr;

  int vfd = open(dev, O_RDWR);
  if (vfd < 0){ perror("open video"); return 1; }
  /* 官方示例同样把 video fd 设为非阻塞: 否则 DQBUF 会在没有帧时无限期睡眠,
     上层(server.py)就没法靠超时判断"相机挂了"并重启采集进程 */
  { int fl = fcntl(vfd, F_GETFL, 0); if (fl >= 0) fcntl(vfd, F_SETFL, fl | O_NONBLOCK); }

  /* ---- 1. capture 格式 ----
   关键: K1 的 vcam 驱动在 open() 时用 call_usermodehelper 自己拉起一发 cam-test
   (ps 里能看到 PPid=2 的 "/bin/bash -c /usr/bin/cam-test <json>")。cam-test 从被拉起
   到向内核注册格式要 1~2s, 这期间 S_FMT/REQBUFS 会直接返回 EAGAIN
   ("Resource temporarily unavailable")。所以这些 ioctl 必须带超时重试, 而不是退出 ——
   否则客户端要么"打不开相机", 要么被迫在外部常驻一个 supervisor 循环重拉 cam-test,
   而多拉出来的 cam-test 会和内核自拉的那发抢 pipeline, 结果就是一致地读到黑帧。 */
  struct v4l2_format f; memset(&f, 0, sizeof f);
  f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  f.fmt.pix_mp.width = W; f.fmt.pix_mp.height = H;
  f.fmt.pix_mp.pixelformat = fourcc; f.fmt.pix_mp.field = V4L2_FIELD_NONE;
  f.fmt.pix_mp.num_planes = 2;
  if (ioctl_wait(vfd, VIDIOC_S_FMT, &f, "S_FMT capture", wait_s, log)) return 1;
  int npl = f.fmt.pix_mp.num_planes; if (npl > NP) npl = NP;
  size_t plane_len[NP] = {0, 0};
  for (int p = 0; p < npl; p++) plane_len[p] = f.fmt.pix_mp.plane_fmt[p].sizeimage;
  size_t frame_len = plane_len[0] + plane_len[1];
  fprintf(log, "fmt %ux%u fourcc=%.4s planes=%d len=%zu+%zu\n",
          f.fmt.pix_mp.width, f.fmt.pix_mp.height, (char*)&fourcc, npl, plane_len[0], plane_len[1]);

  /* ---- 3. CAPTURE 队列: DMABUF 内存型.
   官方流程: REQBUFS -> (逐buffer) QUERYBUF 拿 plane 长度 -> 从 /dev/dma_heap/system
   分配一块 frame_len 连续内存(heap_flags=0xf0) -> 每个 plane 的 m.fd 都填该 fd,
   其余字段保持 0 -> QBUF ---- */
  struct v4l2_requestbuffers rq;
  memset(&rq, 0, sizeof rq);
  rq.count = g_nb; rq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; rq.memory = V4L2_MEMORY_DMABUF;
  if (ioctl_wait(vfd, VIDIOC_REQBUFS, &rq, "REQBUFS capture", wait_s, log)) return 1;
  size_t rq_count = rq.count < NB ? rq.count : NB;   /* 驱动可能给得比请求少 */
  int heaps[2] = { -1, -1 };
  for (int i = 0; i < 2; i++) heap_open(g_heap_paths[i], &heaps[i]);
  if (heaps[0] < 0 && heaps[1] < 0){ perror("open dma_heap"); return 1; }
  void *cmap[NB]; int maps_fd[NB][NP]; size_t map_len[NB];
  for (int i = 0; i < (int)rq_count; i++){
    struct v4l2_buffer b; struct v4l2_plane pl[NP]; memset(&b, 0, sizeof b); memset(pl, 0, sizeof pl);
    b.index = i; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; b.memory = V4L2_MEMORY_DMABUF;
    b.length = npl; b.m.planes = pl;
    if (ioctl_wait(vfd, VIDIOC_QUERYBUF, &b, "QUERYBUF capture", wait_s, log)) return 1;
    size_t sum = 0; for (int p = 0; p < npl; p++) sum += pl[p].length;
    if (sum == 0) sum = frame_len;
    int fd; size_t real = 0;
    if (dmabuf_alloc(heaps, sum, &fd, &real)) return 1;
    void *m = mmap(NULL, real, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED){ perror("mmap heap"); return 1; }
    cmap[i] = m; maps_fd[i][0]=maps_fd[i][1]=fd; map_len[i] = real;
    if (i == 0) fprintf(log, "dmabuf ok: req=%zu real=%zu heap=%s\n", sum, real, g_heap_paths[0]);
  }
  for (int i = 0; i < (int)rq_count; i++){
    struct v4l2_buffer b; struct v4l2_plane pl[NP]; memset(&b, 0, sizeof b); memset(pl, 0, sizeof pl);
    b.index = i; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; b.memory = V4L2_MEMORY_DMABUF;
    b.length = npl; b.m.planes = pl;
    /* 只填 m.fd, 其余为 0 —— 与官方 queue_all_buffers 一致; 用同一块共享内存占位两个 plane 槽 */
    for (int p = 0; p < npl; p++) pl[p].m.fd = maps_fd[i][p];
    if (ioctl_wait(vfd, VIDIOC_QBUF, &b, "QBUF capture", wait_s, log)) return 1;
  }
  /* 注意: 不主动 munmap/close dma-buf fd, 让 exit 统一回收; 见 cleanup 段 */
  for (int i = 0; i < 2; i++) if (heaps[i] >= 0) close(heaps[i]);

  /* ---- 4. 两边 STREAMON (先 capture 后 output, 同官方) ---- */
  enum v4l2_buf_type t;
  t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (ioctl_wait(vfd, VIDIOC_STREAMON, &t, "STREAMON capture", wait_s, log)) return 1;
  fprintf(log, "streaming...\n");

  int done = 0, err = 0;
  while (g_run){
    struct pollfd pf = { .fd = vfd, .events = POLLIN };
    int pr = poll(&pf, 1, 3000);
    if (pr == 0){ fprintf(log, "poll timeout: no frame in 3s\n"); if (++err > 3) break; continue; }
    if (pr < 0){ if (errno == EINTR) break; perror("poll"); if (++err > 3) break; continue; }
    struct v4l2_buffer b; struct v4l2_plane pl[NP];
    memset(&b, 0, sizeof b); memset(pl, 0, sizeof pl);
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; b.memory = V4L2_MEMORY_DMABUF;
    b.length = npl; b.m.planes = pl;
    if (xioctl(vfd, VIDIOC_DQBUF, &b)){
      /* EAGAIN: 驱动给了事件但队列里还没就绪 buffer —— 回到 poll 重试即可, 不能退出 */
      if (errno == EAGAIN || errno == EWOULDBLOCK){ if (++err > 200) break; usleep(2000); continue; }
      perror("DQBUF"); break;
    }
    if (verbose && done < 3)
      fprintf(log, "frame seq=%u idx=%u planes:", b.sequence, b.index);
    if (verbose && done < 3){
      for (int p2=0;p2<npl;p2++) fprintf(log, " [len=%u used=%u off=%u]", pl[p2].length, pl[p2].bytesused, pl[p2].data_offset);
      fprintf(log, "\n");
    }
    char *base = (char*)cmap[b.index];
    /* 每个 plane 在共享 buffer 里的起点是它自己的 data_offset(K1 上 NV12:
       plane0 off=0 len=W*H, plane1 off=W*H len=W*H/2), 不能再叠加累计偏移,
       否则第二平面会从 buffer 外面读 -> 画面上半噪声/下半绿点 */
    for (int p = 0; p < npl; p++){
      size_t n = pl[p].bytesused ? pl[p].bytesused : plane_len[p];
      size_t o = pl[p].data_offset;
      if (o + n > map_len[b.index]) n = map_len[b.index] > o ? map_len[b.index] - o : 0;
      if (n == 0) continue;
      if (fwrite(base + o, 1, n, stdout) != n){
        fprintf(log, "stdout closed by reader, exit\n"); g_run = 0; break;
      }
    }
    if (!g_run) break;
    fflush(stdout);
    b.memory = V4L2_MEMORY_DMABUF; b.length = npl; b.m.planes = pl;

    /* 回收 buffer 必须重新 QBUF。fd 处于非阻塞态 + m2m 内部队列忙时这里会 EAGAIN,
       重试而不是退出, 否则 4 个 buffer 用完之后流就停了(表现: 只出 1~2 帧) */
    { int qe = 0;
      while (xioctl(vfd, VIDIOC_QBUF, &b) && (errno == EAGAIN || errno == EWOULDBLOCK)){
        if (++qe > 500){ perror("QBUF again: still EAGAIN"); g_run = 0; break; }
        usleep(2000);
      }
      if (!g_run) break;
      if (qe && verbose && done < 8) fprintf(log, "  (QBUF retried %d times)\n", qe);
    }
    err = 0;
    if (++done == nframes) break;
    if ((done % 60) == 0) fprintf(log, "%d frames\n", done);
  }
  /* 退出前把流停掉、把 buffer 从驱动侧摘下来：K1 的 vcam 驱动在进程被 kill -9
     或不做 STREAMOFF/REQBUFS(0) 时可能把 CMA 页一直挂在 pipeline 上，
     反复重启采集进程后 heap_flags=0xf0 的分配就会开始报 EINVAL（出图变黑/无帧）。 */
  t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; xioctl(vfd, VIDIOC_STREAMOFF, &t);
  memset(&rq, 0, sizeof rq);
  rq.count = 0; rq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; rq.memory = V4L2_MEMORY_DMABUF;
  xioctl(vfd, VIDIOC_REQBUFS, &rq);
  for (int i = 0; i < (int)rq_count; i++){
    if (cmap[i] && cmap[i] != MAP_FAILED) munmap(cmap[i], map_len[i]);
    for (int p = 0; p < npl; p++) if (maps_fd[i][p] > 0) close(maps_fd[i][p]);
  }
  close(vfd);
  fprintf(log, "exit after %d frames\n", done);
  return 0;
}
