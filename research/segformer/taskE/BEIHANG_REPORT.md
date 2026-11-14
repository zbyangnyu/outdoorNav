# 北航公园台阶 A/B 离线对比

## 数据与来源

北航 RGB-D Stair Dataset v1，DOI 10.17632/6kffmjt7g2.1。作者 Chen Wang、Zhongcai Pei、Shuang Qiu、Yachun Wang、Zhiyong Tang。素材按 CC BY 4.0 提供，样例图由发布的 RGB、分割和边线标注拼版并着色。 [数据页面](https://data.mendeley.com/datasets/6kffmjt7g2/1) · [许可](https://creativecommons.org/licenses/by/4.0/)

- ZIP 大小 1,018,349,513 字节，SHA-256 为 `e4922916010993b16ba8cc4c06a9237a1e959644780e2296490eb48595aeccc9`。数据中 2,832 对 RGB、分割和台阶边线文件可以逐名配对。原始 154 张公开 test RGB 没有同格式的分割和边线真值。
- 用标注筛出 1,201 张下行候选（有凸边、无凹边与立面）和 1,631 张上行候选（有凹边和立面），含同名左右版本。随后检查公园 `park*` 子集的 355 个缩略图，按文件名合并左右版本后，下行候选 164 张，上行候选 191 张。部分近拍图几乎只有地面，方向依据原标注判断。
- 公园子集有明确的室外顶部下望画面。没有逐帧机位高度记录，确认符合山猫低机位高度的数量未知。图中还可看见若干拍摄者身影。实验衡量公园台阶拍摄域中的上下行差距。
- 按人工观察的编号邻近段、铺装、建筑和楼梯边墙保守分成 10 组。同一组整体进入一个集合，分别为 4 组训练、2 组验证、4 组测试。用图像感知哈希抽查了 12 对最相似的跨集合样本，未辨出明显同帧或同一楼梯。原数据没有物理楼梯和采集序列 ID，这项人工分组隔离有不确定性。

| 划分 | 北航下行 | 北航上行 | ADE 普通道路 | ADE 草土和混合 |
|---|---:|---:|---:|---:|
| 训练 | 63 | 63 | 240 | 200 |
| 验证 | 52 | 52 | 40 | 0 |
| 测试 | 44 | 44 | 40 | 0 |

发布的分割标注中，128 是立面，255 是踏面。训练时两者均映射为阶梯类。北航图中的其余像素在四分类损失中忽略，因为没有路面/草土/其他的真值，仍按已知非台阶区域用于边缘监督与台阶 IoU。普通道路指标来自独立的 ADE 人工像素标注。分组与样本见 [固定划分](beihang_split.csv)，原始计数见 [数据审查](beihang_audit.json)。

## 方法与复现

A 保留四分类微调和道路蒸馏，B 用同一个分割网络以及额外的训练期边缘头。B 在北航样本上同时使用台阶外轮廓和发布的逐级边线，边线图与 RGB 和分割标签做相同裁剪及翻转。在两组都使用的 ADE 样本上从分割标签提取轮廓。两组均只使用 RGB，SegFormer-B0 从相同 Cityscapes 版本初始化。

固定批量 8、输入 512×384、种子 20261115、每组 1,000 步并保存末步 EMA。验证集负责选置信门槛。最终 80% 目标使用 B 的测试集「下行台阶 IoU ÷ 上行台阶 IoU」，同时公布绝对 IoU、召回、误作路面和普通道路 IoU，避免把上行做差后比例变好。

B 首轮的验证下行 / 上行 IoU 为 75.43%，于是增加一次验证试验，把边缘损失权重从 0.2 提到 0.6。两组 B 使用同一阈值规则。只有 0.6 提高验证下行 IoU，且相比 A 的下行召回和普通道路 IoU 下降均不超过 2 个百分点时，才选 0.6。这个决定在查看测试结果前作出。

在仓库根目录运行，下方依赖、ADE ZIP 和预训练权重沿用 [原复现步骤](STAIR_REPRO.md)。生成物放在仓库外。

```powershell
$p = '../.venv-segformer/Scripts/python.exe'
$d = 'research/segformer/taskE'
$r = '../segformer-local'
New-Item -ItemType Directory -Force "$r/public/beihang" | Out-Null
curl.exe -fL 'https://data.mendeley.com/public-api/zip/6kffmjt7g2/download/1' -o "$r/public/beihang/beihang-v1.zip"
& $p "$d/prepare_beihang.py" --beihang "$r/public/beihang/beihang-v1.zip" --ade "$r/public/ADEChallengeData2016.zip" --output "$r/beihang-data"
$env:HF_HUB_OFFLINE = '1'
$common = @('--config',"$d/stair_beihang.json",'--data',"$r/beihang-data",'--split',"$r/beihang-data/split.json",'--beihang')
& $p -u "$d/stair_experiment.py" train @common --variant A --pretrained "$r/pretrained" --output "$r/beihang-A"
& $p -u "$d/stair_experiment.py" train @common --variant B --pretrained "$r/pretrained" --output "$r/beihang-B"
& $p -u "$d/stair_experiment.py" train @common --variant B --edge-weight 0.6 --pretrained "$r/pretrained" --output "$r/beihang-B-edge06"
& $p -u "$d/stair_experiment.py" compare @common --a "$r/beihang-A" --b "$r/beihang-B-edge06" --output "$r/beihang-comparison"
```

生成的 `beihang-data/annotation_examples.jpg` 左侧为原图，中间浅蓝表示踏面、橙色表示立面，右侧绿色为凸边、橙色为凹边。固定本地数据的 `split.json` SHA-256 为 `fad1942b336ca948687618bab54512a45d9df4862cd9e0ba7922d8896e87b912`。

## 验证集选型

| 模型 | 下行 IoU | 上行 IoU | 下行误作路面 | 下行召回 | 道路 IoU |
|---|---:|---:|---:|---:|---:|
| A | 60.15% | 79.76% | 1.34% | 95.78% | 89.14% |
| B，边缘权重 0.2 | 60.24% | 79.85% | 1.23% | 96.04% | 89.12% |
| B，边缘权重 0.6 | 60.42% | 79.93% | 1.08% | 96.32% | 89.11% |

最终选择 0.6，A 和 B 的置信门槛均为 0.6。三次训练分别用时约 232、239、239 秒，均为 1,000 步。 [验证记录](beihang_validation.json)

## 独立测试

**B 把下行误作路面的比例相对降低了 21.03%，但下行 IoU 只达到上行的 72.16%，80% 目标未达成。**

| 指标 | A 直接微调 | B 加边缘监督 |
|---|---:|---:|
| 下行台阶误作路面 | 3.82% | 3.01% |
| 下行台阶召回 | 89.88% | 91.70% |
| 下行台阶 IoU | 52.62% | 53.38% |
| 上行台阶召回 | 97.82% | 98.34% |
| 上行台阶 IoU | 73.93% | 73.97% |
| 下行 IoU / 上行 IoU | 71.17% | 72.16% |
| 下行召回 / 上行召回 | 91.89% | 93.25% |
| ADE 普通道路 IoU | 89.60% | 89.40% |
| ADE 无楼梯误报像素比例 | 0.318% | 0.279% |
| ADE 无楼梯画面出现误报 | 6/40 | 4/40 |
| RTX 5090 单张前向，中位数 / P95 | 11.70 / 12.57 ms | 11.59 / 12.31 ms |

正式公开测试共 128 张，44 张下行含 4,180,263 个楼梯像素，44 张上行含 5,610,527 个楼梯像素，另有 40 张普通道路。普通道路 IoU 下降 0.21 个百分点，下行召回上升 1.82 个百分点。相比 B 的上行 IoU，达到 80% 需要下行约 59.18%，实测相差 5.80 个百分点。 [原始指标](beihang_metrics.json)

| 测试场景组 | 每方向帧数 | 下行 IoU A → B | 上行 IoU A → B | 下行误作路面 A → B |
|---|---:|---:|---:|---:|
| 草岸砖路 | 4 | 44.64% → 49.98% | 78.42% → 78.62% | 24.21% → 19.64% |
| 建筑平台 | 16 | 47.59% → 47.71% | 70.19% → 69.89% | 0.93% → 0.70% |
| 雕饰石阶 | 21 | 57.60% → 58.79% | 75.32% → 75.59% | 3.81% → 2.98% |
| 粗石台阶 | 3 | 51.05% → 50.04% | 81.14% → 81.27% | 1.24% → 0.95% |

`beihang-comparison/comparisons.jpg` 给出原图、真值、A 和 B。草岸砖路的 `park28` 从顶部看时，A、B 仍分别把 90.3% 和 79.1% 的台阶像素认成路。拍自同一场景下方的 `park24` 有清晰立面，两组都能覆盖大部分阶梯。建筑平台的 `park61` 中，两组把台阶认作路面的比例都为零，但也把相邻平台划入楼梯，拉低了 IoU。下一轮需要补独立低机位场景，尤其标清第一阶与顶部平台的交界。

本轮只有一次验证驱动的调整，最终测试只比较 A 和选定的 B。公开数据上看到误认下降，80% 的 IoU 目标未达成。每个测试方向只覆盖 4 个推定场景组，北航图也没有可验证的低机位记录，结果不能确定山猫视角的表现。没有连接、部署或测试山猫。

本机选定权重位于仓库旁边的 `segformer-local/beihang-A/selected/model.safetensors` 和 `segformer-local/beihang-B-edge06/selected/model.safetensors`。
