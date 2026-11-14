# SegFormer-B0 Cityscapes 部署准备与测速（2026-11-11）

服务器：RTX 4090D（共享），torch 2.8+cu128，onnxruntime-gpu 1.22，TensorRT 10.13。100 张山猫相机图（640×480）。
测法：预热 30 次，跑 300 次取中位数，每次都等 GPU 算完再计时。
- 仅模型：输入已在 GPU 上，只算网络本身。
- 端到端：uint8 图片 → GPU 归一化/缩放 → 模型 → 放大回 480×640、softmax、置信度 0.7 → 类别图拷回 CPU。
- 一致性：与 PyTorch fp32 对比路面掩码（逐像素）；“前方路面比例”= 底部 120 行、中间 320 列中判为路面的比例。

| 模型 | 输入(H×W) | 后端 | 仅模型 ms | 端到端 ms | 路面掩码一致率 | 前方路面比例 (PyTorch fp32) | logits 最大误差 |
|---|---|---|---|---|---|---|---|
| SegF-B0 | 480x640 | PyTorch fp32 | 7.98 | 8.60 | 100.00% | 0.999 (0.999) | 0.000 |
| SegF-B0 | 480x640 | PyTorch fp16 (autocast) | 10.77 | 9.72 | 100.00% | 0.999 (0.999) | 0.051 |
| SegF-B0 | 480x640 | ORT CUDA fp32 | 2.64 | 3.34 | 100.00% | 0.999 (0.999) | 0.021 |
| SegF-B0 | 480x640 | ORT CUDA fp32 (dynamic onnx) | 3.01 | 3.67 | 100.00% | 0.999 (0.999) | 0.022 |
| SegF-B0 | 480x640 | **TensorRT fp16** | **0.84** | **1.43** | 99.99% | 0.999 (0.999) | 0.153 |
| SegF-B0 | 240x320 | PyTorch fp32 | 7.49 | 6.67 | 100.00% | 0.981 (0.981) | 0.000 |
| SegF-B0 | 240x320 | PyTorch fp16 (autocast) | 8.67 | 9.51 | 99.99% | 0.981 (0.981) | 0.057 |
| SegF-B0 | 240x320 | ORT CUDA fp32 | 2.32 | 2.94 | 100.00% | 0.981 (0.981) | 0.026 |
| SegF-B0 | 240x320 | ORT CUDA fp32 (dynamic onnx) | 2.51 | 3.14 | 99.99% | 0.981 (0.981) | 0.025 |
| SegF-B0 | 240x320 | **TensorRT fp16** | **0.61** | **1.22** | 99.97% | 0.981 (0.981) | 0.165 |
| PIDNet-M | 512x640 | PyTorch fp32 | 6.81 | 7.70 | 100.00% | 0.630 (0.630) | 0.000 |
| PIDNet-M | 512x640 | PyTorch fp16 (autocast) | 8.79 | 9.63 | 99.97% | 0.630 (0.630) | 0.021 |
| PIDNet-M | 512x640 | **TensorRT fp16** | **0.82** | **1.44** | 99.92% | 0.630 (0.630) | 0.053 |

要点：
- PyTorch 的 7~11ms 基本是 Python 逐层调度开销（fp16 反而更慢），不代表模型真实速度。
- TRT fp16 下 B0(640×480) 与 PIDNet-M(640×512) 几乎一样快（~0.8ms），结果与 PyTorch 一致（掩码 ≥99.97%，前方路面 0.999 不变）。
- 320×240 只快一点（0.61 vs 0.84ms），但前方路面从 0.999 掉到 0.981，不值得降分辨率。

## Jetson Orin 估算（未实测，误差大）
没找到 NVIDIA 或可靠来源公布的 SegFormer-B0 / PIDNet 在 Orin 上的数字（NGC 的 SegFormer 卡片 Orin 行为空）。只能按硬件规格比例粗估：
- 4090D vs AGX Orin 64GB：显存带宽 ~1008 vs 204.8 GB/s（≈5×），FP16 张量算力约 3.5~5×；小模型还会被每层启动开销拖累 → 取 4~8 倍。
- Orin NX 16GB：带宽 102.4 GB/s（≈10×），GPU 算力约为 AGX 的 1/3 → 取 10~18 倍。

| 模型（TRT fp16，仅模型） | 4090D 实测 | AGX Orin 估算 | Orin NX 16GB 估算 |
|---|---|---|---|
| SegF-B0 640×480 | 0.84 ms | ~3.5–7 ms | ~8–15 ms |
| PIDNet-M 640×512 | 0.82 ms | ~3.5–7 ms | ~8–15 ms |

两者在 Jetson 上大概率同一量级，相机 30Hz（33ms）应都够用。端到端还要加前后处理（若放 CPU 做会明显更慢，建议放 GPU）。以 `jetson_build_bench.sh` 实测为准。

## 文件
- `export_onnx.py`：从 HF 权重导出 3 个 ONNX（输入 `input` [1,3,H,W] 归一化 RGB；输出 `logits` [1,19,H/4,W/4]）。
- `verify_bench.py`：服务器上的一致性检查 + 测速（产出 `results.json`）。
- `jetson_build_bench.sh`：Jetson 上 trtexec 构建 FP16 引擎并测速。
- ONNX：`segf_b0_480x640.onnx`、`segf_b0_240x320.onnx`、`segf_b0_dyn.onnx`、`pidnet_m_512x640.onnx`（本地和服务器 `/root/autodl-tmp/seg/taskC/` 都有）。
  注意：TRT 引擎（`.plan/.engine`）跟设备和 TRT 版本绑定，必须在 Jetson 上重新构建。
