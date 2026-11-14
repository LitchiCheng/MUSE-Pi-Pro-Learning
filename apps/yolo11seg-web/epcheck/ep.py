#!/usr/bin/env python3
"""建一次 session 并把 ORT 的 VERBOSE 日志全留在 stdout —— 用来判断算子到底落在哪个 EP。

用法（板端，务必带 LD_PRELOAD，见 run-epcheck.sh 的说明）：
    LD_PRELOAD=/usr/lib/libonnxruntime.so.1.18.1 python3 ep.py <模型.onnx> > <模型>.log 2>&1

为什么需要它：`sess.get_providers()` 里出现 SpaceMITExecutionProvider 只说明 EP 注册成功了，
不代表算子真在智算核上跑；而 `SPACEMIT_EP_DUMP_SUBGRAPHS` 在这版 runtime 上恒为
spine_subgraphs=0（会误判成"全没加速"）。唯一能对得上耗时的判据是
ORT 自己的 VerifyEachNodeIsAssignedToAnEp 节点清单，交给 place.py 统计。
"""
import os
import sys

import onnxruntime as ort   # 顺序：必须先 ORT 再 spacemit_ort，反序会让 int8 静默回退 CPU
import spacemit_ort  # noqa: F401

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(2)
model = sys.argv[1]

so = ort.SessionOptions()
so.log_severity_level = 0      # VERBOSE：节点清单只在这个级别输出
so.log_verbosity_level = 3
so.intra_op_num_threads = int(os.environ.get("INTRA", "4"))
sess = ort.InferenceSession(model, sess_options=so,
                            providers=["SpaceMITExecutionProvider", "CPUExecutionProvider"])
sys.stderr.write("PROVIDERS %s\n" % (sess.get_providers(),))
