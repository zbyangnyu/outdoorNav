# 复现步骤（服务器 `/root/autodl-tmp/seg/taskE/`）

```bash
export PATH=/root/miniconda3/bin:$PATH HF_HOME=/root/autodl-tmp/hf
cd /root/autodl-tmp/seg/taskE
python -u prep_data.py                 # 生成 data/{img,gt}/ 和 data/split.json（RUGD 图片按需远程下载，约 20 分钟）
python -u fetch_itlp.py                # 追加 ITLP-Campus-Outdoor（机器人视角校园图，约 15 分钟）
KD=0.5 EMA=1 SAVE_ALL=1 OUT=run_v5 ITERS=20000 python -u train.py   # 最终版训练（约 80 分钟），每 1000 步存 run_v5/itN
python swa.py run_v5/swa run_v5/it10000 run_v5/it12000 run_v5/it14000 run_v5/it16000 run_v5/it18000 run_v5/it20000
OUT=eval python eval_all.py run_v5/swa                 # 门槛 0.7，与原版 B0 CS 并排
THR=0.5 OUT=eval_thr05 python eval_all.py run_v5/swa   # 门槛 0.5
python export_onnx.py run_v5/swa b0_taskE_v5swa_640x480.onnx   # 导出 ONNX 并与 PyTorch 对比
python select_ckpt.py run_v5 run_v6    # 每个检查点的分来源验证分数（选权重参考）
python quick_lynx.py CKPT              # 只看实机 100 张路面分数（诊断用，不参与选权重）
python plot_curves.py run_v5 run_v6 run_kd run_v2   # 训练曲线 -> curves.png
```

依赖：transformers、torch、opencv、remotezip、onnxruntime-gpu（服务器已装）；ADE20K 用服务器现成的
`/root/autodl-tmp/ade/ADEChallengeData2016`（只读 training），RUGD 标注用 taskD 下载好的 `RUGD_annotations/`。
taskD 的 `perturb_eval.py`、`hard_eval.py`、taskB 的 `evaluate.py` 被 `eval_all.py` 直接 import 复用。

脚本说明：
- `train.py`：最终版（`KD` 蒸馏权重、`EMA`、`SAVE_ALL` 由环境变量控制；v6 = `KD=0`）。服务器上的 `train_kd.py`/`train_ema.py` 是它的早期副本。`train_v1.py` = 第一版（无树影/落叶增强，会把实机柏油路认成草土）。
