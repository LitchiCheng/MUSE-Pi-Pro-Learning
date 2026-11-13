# MUSE-Pi-Pro-Learning

MUSE Pi Pro（SpacemiT K1，RISC-V）上手记录：内核构建脚本 + 一个完整的应用例子 + 踩坑文档。

## env-prepare.sh

## build-kernel.sh

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
| [k1-aipu-onnxruntime.md](docs/k1-aipu-onnxruntime.md) | ONNXRuntime + 智算核：float 不进 AIPU、int8 QDQ 配方、opset 上限、EP 验证方法 |
## env-prepare.sh

## build-kernel.sh