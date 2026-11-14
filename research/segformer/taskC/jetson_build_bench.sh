#!/usr/bin/env bash
# 在 Jetson 上一键：构建 TensorRT FP16 引擎 + 测速（纯模型 GPU 时间）。
# 用法: bash jetson_build_bench.sh [onnx 所在目录]
# 先把 segf_b0_480x640.onnx / segf_b0_240x320.onnx / pidnet_m_512x640.onnx 拷到该目录
# （在有 torch+transformers 的机器上 `python export_onnx.py .` 生成；PIDNet 的由 verify_bench.py 生成）
set -e
D=${1:-.}
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
sudo nvpmodel -m 0 || true      # 最高功耗模式（会提示重启时选 no 即可）
sudo jetson_clocks || true      # 锁最高频率
for m in segf_b0_480x640 segf_b0_240x320 pidnet_m_512x640; do
  [ -f "$D/$m.onnx" ] || { echo "skip $m (no onnx)"; continue; }
  echo "=== $m ==="
  $TRTEXEC --onnx=$D/$m.onnx --fp16 --saveEngine=$D/${m}_fp16.engine \
    --memPoolSize=workspace:2048 --warmUp=2000 --iterations=300 \
    --useCudaGraph --useSpinWait --noDataTransfers 2>&1 | tee $D/${m}_trtexec.log | \
    grep -E "GPU Compute Time|Throughput"
done
# 再次只测速（不重新构建）: trtexec --loadEngine=segf_b0_480x640_fp16.engine --useCudaGraph --useSpinWait --noDataTransfers
# 看 "GPU Compute Time: ... median = X ms" 那一行即单张推理时间
