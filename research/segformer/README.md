# SegFormer 实验交接（2026-11-14）

这里保存从本机隐藏工作目录迁出的训练、评估、导出脚本与实验记录，共 69 个原始文件。
原目录被 `.git/info/exclude` 排除，之前没有进入 Git；迁出文件内容不变，校验值见 `source_manifest.json`。

## 当前方案

先读 `taskE/results.md`、`taskE/REPRO.md`、`taskE/train.py`。
- 最新实验交付选择：SegFormer-B0 Cityscapes → 4 类，0 其他、1 硬质路面、2 草地泥土、3 台阶。
- 全模型微调：骨干学习率 3e-5，分割头 6e-4，AdamW；640×480，batch 16，20,000 步。
- ADE20K + RUGD + ITLP，共训练 6,868 张、验证 472 张；山猫 100 张不参与训练。
- 台阶图片抽样 ×3，台阶损失权重 ×3；含低视角裁剪、树影、落叶及光照天气增强。
- 冻结原版 B0 Cityscapes 作教师，蒸馏路面概率，权重 0.5；不在已标注草土、台阶像素上蒸馏。
- EMA 后对第 10k、12k、14k、16k、18k、20k 的检查点做 SWA，最终 v5 SWA 建议置信门槛 0.5。
- 报告：门槛 0.5 时实机路面 IoU 0.964，ADE 台阶误认成路 3.2%，台阶召回约 0.38。
  台阶召回不是“下楼梯专门测试”的成绩。门槛和权重平均策略看过测试结果后确定，存在偏乐观风险。

## 已保存 / 未保存

已保存：全部原目录 .py/.sh/.md/.json/.jsonl/.csv/.log 文件，覆盖 compare、早期微调、taskA–E、stairs、padding、PDF 报告生成源码。
未保存到 Git：图片、视频、数据集、PDF、ONNX、模型权重、Python 环境。日志和指标是历史记录，本次未重新训练或复测。
本地早期 `finetune/ft_b0/model.safetensors` 不是最新 v5 SWA，不能作为 v5 交付权重。

## 权重处理（交接决定）

在另一台电脑重新训练，不连接 AutoDL，也不取回旧训练权重。
以 taskE 的数据处理、训练和评估代码为基础，下载公开的 SegFormer-B0 Cityscapes 预训练权重及所需数据，重新生成微调权重。
历史报告、日志中的服务器地址与权重位置仅为实验来源，不是新环境的依赖。
`tools/r.sh` 是历史连接脚本，新环境不需要使用，也不需要原机器的 SSH 私钥。

## 另一台电脑运行须知

原脚本保留服务器绝对路径 `/root/autodl-tmp/seg`、`/root/autodl-tmp/hf`、`/root/autodl-tmp/ade`。
先检查新电脑 GPU、显存及 PyTorch 支持情况；原代码使用 CUDA/bfloat16，不能直接在 Mac 上原样训练。
将硬编码的历史服务器路径改为新电脑的配置路径，再准备模型缓存与数据；不依赖原 AutoDL 目录。
保留子目录结构，taskE 评估会导入根目录 compare.py、taskB 与 taskD 模块。
依赖以脚本及 REPRO.md 为准：torch、transformers、numpy、opencv-python、remotezip、onnx、onnxruntime（GPU 环境可用 onnxruntime-gpu），画图另需 matplotlib；没有经过锁定的版本环境。
原版 PIDNet 比较还依赖 `compare.py` 引用的 vendor/semantic_local_grid 和其权重，本归档不包含该外部源码；不是训练 taskE 的必需项。
报告提及导航分支 `feat/segformer-b0-backend`，但当前导航 checkout 不含 segformer_backend.py；此次也未找到本地该文件。需先核查远端分支，缺失则实现适配器，不能认定已部署。
独立导航仓库不由根仓库跟踪，目录、origin、分支、提交见 `workspace_repositories.json`。本次归档不修改这些仓库。

## 下楼梯研究

下楼梯边缘监督的代码及公开数据流程已完成阶段验证，见 [阶段报告](taskE/STAIR_REPORT.md)
和 [复现步骤](taskE/STAIR_REPRO.md)。山猫人工标注与正式对比仍待补充。

目标：低视角接近下楼梯时，更早、稳定识别入口，同时保持平地识别。
1. 先采集并标注山猫下楼梯图片；按物理楼梯/采集序列划分训练和测试，避免相邻帧泄漏。
2. 建立同数据、同训练量的直接微调对照。
3. 第一项方法实验：在现有 4 类模型上增加台阶边缘辅助监督，与直接微调比较。
4. 再视失败案例决定增加运动对齐后的连续画面融合，或有效深度辅助；不能把简单多数投票当成解决系统性漏认。
5. 除分割指标外，测楼梯误认成平路、平地/坡道误报、首次稳定识别时距边缘距离、推理速度及平地能力退步。
这些都是研究建议，没有承诺收益，也尚不能认定为独创方法。

参考：
- https://experts.umn.edu/en/publications/descending-stair-detection-approach-and-traversal-with-an-autonom/
- https://www.mdpi.com/2313-433X/11/8/258

## 山猫约束

遵守根目录 AGENTS.md。不使用山猫 ODOM；SSH 为 `ssh -p 6003 nvidia@svr.gddg08.top`。
本机曾因公钥认证失败无法访问山猫；另一台电脑需要自己的有效 SSH 配置。
测试等待尽量短；部署按根目录同步说明直接替换，不备份旧导航栈。
