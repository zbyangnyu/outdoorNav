# 下楼梯边缘监督离线实验复现

## 环境与公开数据检查

以下命令在 Windows PowerShell 中从 outdoorNav 仓库根目录运行。已验证 Python 3.13.11、RTX 5090、NVIDIA 驱动 610.88 和 PyTorch 2.8.0+cu128。数据、缓存、输出和权重放在仓库旁边的 `segformer-local`。

```powershell
python -m venv ../.venv-segformer
$p = '../.venv-segformer/Scripts/python.exe'
$d = 'research/segformer/taskE'
$r = '../segformer-local'
& $p -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& $p -m pip install -r "$d/stair_requirements.txt"
& $p -m pip check
New-Item -ItemType Directory -Force "$r/public","$r/pretrained" | Out-Null
curl.exe -fL https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -o "$r/public/ADEChallengeData2016.zip"
$model = 'https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/resolve/21b3847fae21ddee674abd31129307b6a1235bd9'
curl.exe -fL "$model/config.json" -o "$r/pretrained/config.json"
curl.exe -fL "$model/pytorch_model.bin" -o "$r/pretrained/pytorch_model.bin"
& $p "$d/prepare_stair_smoke.py" "$r/public/ADEChallengeData2016.zip" "$r/smoke-data"
$env:HF_HUB_OFFLINE = '1'
$common = @('--config',"$d/stair_smoke.json",'--split',"$d/stair_smoke_split.json",'--data',"$r/smoke-data",'--smoke')
& $p -u "$d/stair_experiment.py" train @common --variant A --pretrained "$r/pretrained" --output "$r/public-A"
& $p -u "$d/stair_experiment.py" train @common --variant B --pretrained "$r/pretrained" --output "$r/public-B"
& $p -u "$d/stair_experiment.py" compare @common --a "$r/public-A" --b "$r/public-B" --output "$r/public-comparison"
```

输出目录必须是新的。重跑时可以加一个共同后缀，并在 compare 命令引用对应目录。每组保存 `selected/model.safetensors`、`selected.json` 和 `log.jsonl`。对比目录保存 `metrics.json` 和 `comparisons.jpg`。

- ADE20K ZIP SHA-256 为 `7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29`。
- B0 Cityscapes 权重 SHA-256 为 `027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c`。
- 固定公开划分是 12 张训练、6 张验证、6 张留作流程对比。楼梯图按原始标注选择，视角及地点未核定。公开检查使用 4 步训练，正式配置见下节。

## 正式数据与训练

为需要纳入实验的山猫帧制作与原图同尺寸的灰度 ID PNG。像素 ID 为 `0` 其他、`1` 普通硬质路面、`2` 草地或泥土、`3` 楼梯可见踏面与立面、`255` 无法判定。标清第一级台阶与顶部平台的分界，平台仍按实际路材标为路面或其他。遮挡可按可见物体标注，难以确定的像素填 `255`。验证和测试用人工标注。

填写 `$r/lynx-annotations.csv`，表头如下。

```csv
part,n,image,mask,scene,sequence,stair_id
```

- `part` 为 `train`、`val` 或 `test`，`n` 为全局唯一帧 ID，`image`、`mask` 是相对于 `$r` 的路径。
- `scene` 用 `downstairs` 标记从顶部看向下行楼梯的画面，用 `stairs` 标记其他楼梯视角，用 `road` 标记没有楼梯的普通道路，用 `no_stairs` 标记其他无楼梯画面，用 `mixed` 标记其余画面。
- 同一采集序列使用同一 `sequence`；同一物理楼梯在不同序列中仍使用同一 `stair_id`。在划分前按两者归组，同一组只进一个集合。画面包含多座楼梯时，人工确认涉及这些楼梯的所有序列均进入同一集合。每个集合都需要山猫下楼梯画面和普通道路画面，测试宜覆盖多座独立楼梯和多段序列。先冻结划分，再训练。
- 需要加入历史 Task E 的公开训练样本时，用 `SEG_ADE`、`SEG_TASKD_DATA` 和 `SEG_DATA` 指定 ADE20K 解压目录、含 `RUGD_annotations` 的目录和准备好的数据输出目录。用 `prep_data.py` 生成 ADE20K/RUGD 数据，再运行 `fetch_itlp.py` 加入 ITLP。可选 `SEG_RUGD_FRAMES` 指定 RUGD 帧 ZIP 地址，`HF_ENDPOINT` 指定公开 Hugging Face 下载端点。把输出目录放在 `$r` 下，并向下方的 `make_stair_split.py` 追加 `--public-split "$r/taskE-data/split.json"`。仅追加历史训练部分。ITLP 自带伪标签仅用于训练，边缘监督与正式验证、测试均使用人工标注。

```powershell
& $p "$d/make_stair_split.py" --annotations "$r/lynx-annotations.csv" --data $r --output "$r/field-split.json"
$common = @('--config',"$d/stair_field.json",'--split',"$r/field-split.json",'--data',$r)
& $p -u "$d/stair_experiment.py" train @common --variant A --pretrained "$r/pretrained" --output "$r/field-A"
& $p -u "$d/stair_experiment.py" train @common --variant B --pretrained "$r/pretrained" --output "$r/field-B"
& $p -u "$d/stair_experiment.py" compare @common --a "$r/field-A" --b "$r/field-B" --output "$r/field-comparison"
```

A、B 共用 Cityscapes 起点、采样顺序、增强、种子 20261115、Task E 四分类损失和路面蒸馏。两组均训练 20,000 步，均取第 20,000 步的 EMA 权重。每 1,000 步记录验证分数。用最终权重在验证集选择置信阈值，按楼梯 IoU 和道路 IoU 的均值减去楼梯误作路面比例与无楼梯像素误报比例排序，平分按配置中阈值的顺序取第一个。正式比较在 A、B 确定后运行一次。少量 CUDA 运算可能非逐位确定，种子、配置和数据划分固定。

B 在共享分割特征上加一个单通道边缘头，损失权重为 0.2。在完成几何增强后找出台阶类和已知非台阶类的相邻像素，向外扩 3 像素，跳过邻近忽略区的像素，并对边缘和非边缘做等权平均。用普通分割标注可提取楼梯区域轮廓，逐级踏面之间的边线需要单独标注。保存及推理使用四分类分割网络。

主指标在标为 `downstairs` 的帧中统计，分母为人工标注为楼梯的像素。普通道路 IoU 在 `road` 帧中统计。无楼梯误报同时报告误报像素比例与出现任意有效像素误报的帧比例。统计统一在模型输入分辨率进行，`255` 排除。速度为同机单张、预热 10 次后 40 次同步前向的中位数和 P95，计时范围为模型前向。报告时同时看分母、独立楼梯数及逐楼梯数据。A 接近零或独立测试楼梯过少时，20% 相对改善目标难以可靠判断。
