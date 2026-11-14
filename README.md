# MUSE-Pi-Pro-Learning

MUSE Pi Pro（SpacemiT K1，RISC-V）上手记录：内核构建脚本 + 一个完整的应用例子 + 踩坑文档。

## env-prepare.sh

在 **x86_64 Ubuntu** 上准备 K1 内核构建环境（需 root）：装 build-essential/repo/debhelper 等依赖，
用 tuna 镜像的 git-repo 从 gitee 的 `bianbu-linux/manifests` 拉 `k1-bl-v2.2.y` 清单并 sync，
再下载 SpacemiT 交叉工具链 v1.0.5 解到 `toolchain/`。两步都有幂等判断（目录已存在就跳过）。

## build-kernel.sh

用上面的工具链交叉编译 `linux-6.6`（`k1_defconfig` + `LOCALVERSION=""`），产物：
`make modules_install` 到 `output/`、`bindeb-pkg` 出的 `*deb` 也移到 `output/`，最后 `scpToBoard`
推到板子。脚本顶部 `username` / `board_ip` 是**上一位作者的机器地址，跑之前必须改成自己的**；
首次执行只会 clone 内核源码，编译要从第二次开始（`build()` 里的分支就是这么写的）。

## apps/yolo11seg-web

板端实时示例：OV5647(MIPI) → `/dev/video50` → `k1cam` 采 NV12 → ONNXRuntime(SpaceMIT EP)
跑 YOLO11n-seg → Flask MJPEG 网页。包含板端一键准备脚本 `setup-board.sh`、
相机点亮校验 `cam-start.sh`、启动脚本 `run-web.sh`、量化脚本 `quant/`。
详见 [apps/yolo11seg-web/README.md](apps/yolo11seg-web/README.md)。

## docs

| 文档 | 讲什么 |
|---|---|
| [k1-env-setup.md](docs/k1-env-setup.md) | 开发环境搭建：apt/pip(PEP668)、SSH、zsh 默认 shell、依赖清单 |
| [k1-mipi-camera.md](docs/k1-mipi-camera.md) | MIPI 相机点亮：cam-test 单实例、dma-buf `heap_flags=0xf0`、plane 偏移、症状对照表 |
| [k1-aipu-onnxruntime.md](docs/k1-aipu-onnxruntime.md) | ONNXRuntime + 智算核：float 不进 AIPU、int8 QDQ 配方、opset 上限、EP 验证方法、**EP 长跑 SIGSEGV 的实测矩阵与 native 栈** |
