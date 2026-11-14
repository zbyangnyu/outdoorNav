"""Build the segmentation comparison report (Chinese, A4) with reportlab."""
import os, sys
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
                                KeepTogether, ListFlowable, ListItem)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figs')
OUT = os.path.join(HERE, '山猫路面与楼梯分割对比报告.pdf')
pdfmetrics.registerFont(TTFont('CN', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))

INK, MUTED, LINE = colors.HexColor('#1d2420'), colors.HexColor('#5d6762'), colors.HexColor('#d5dbd7')
GOOD, BAD, HEAD = colors.HexColor('#e3f3e6'), colors.HexColor('#fbe3d6'), colors.HexColor('#eef1ee')
base = dict(fontName='CN', textColor=INK, wordWrap='CJK', alignment=TA_LEFT)
S = {
    'title': ParagraphStyle('t', fontSize=22, leading=30, spaceAfter=6, **base),
    'sub': ParagraphStyle('s', fontSize=10.5, leading=16, textColor=MUTED, **{k: v for k, v in base.items() if k != 'textColor'}),
    'h1': ParagraphStyle('h1', fontSize=15, leading=22, spaceBefore=10, spaceAfter=6, **base),
    'h2': ParagraphStyle('h2', fontSize=12, leading=18, spaceBefore=8, spaceAfter=4, **base),
    'p': ParagraphStyle('p', fontSize=10, leading=16.5, spaceAfter=5, **base),
    'cap': ParagraphStyle('c', fontSize=9, leading=14, spaceBefore=3, spaceAfter=10,
                          textColor=MUTED, **{k: v for k, v in base.items() if k != 'textColor'}),
    'cell': ParagraphStyle('cell', fontSize=8.8, leading=12.5, **base),
    'cellh': ParagraphStyle('cellh', fontSize=8.8, leading=12.5, textColor=MUTED,
                            **{k: v for k, v in base.items() if k != 'textColor'}),
}
W = A4[0] - 30 * mm


def P(t, s='p'): return Paragraph(t, S[s])


def bullets(items):
    return ListFlowable([ListItem(P(t), leftIndent=10, value='•') for t in items], bulletType='bullet',
                        start='•', leftIndent=12, bulletFontName='CN', bulletFontSize=9)


def table(rows, widths, hl=None):
    """rows[0] is the header. hl: {row_index: colour}."""
    data = [[P(str(c), 'cellh' if i == 0 else 'cell') for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    st = [('LINEBELOW', (0, 0), (-1, -1), .4, LINE), ('BACKGROUND', (0, 0), (-1, 0), HEAD),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 4),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 4), ('LEFTPADDING', (0, 0), (-1, -1), 5)]
    for r, c in (hl or {}).items():
        st.append(('BACKGROUND', (0, r), (-1, r), c))
    t.setStyle(TableStyle(st))
    return t


def fig(name, caption, max_h=118 * mm, heading=None):
    path = os.path.join(FIG, name)
    w, h = PILImage.open(path).size
    s = min(W / w, max_h / h)
    head = [P(heading, 'h2')] if heading else []
    return KeepTogether(head + [Image(path, width=w * s, height=h * s), P(caption, 'cap')])


def footer(c, doc):
    c.saveState()
    c.setFont('CN', 8); c.setFillColor(MUTED)
    c.drawString(15 * mm, 9 * mm, '山猫室外导航 · 视觉路面识别对比')
    c.drawRightString(A4[0] - 15 * mm, 9 * mm, f'第 {doc.page} 页')
    c.restoreState()


story = []
# ---------------------------------------------------------------- cover + summary
story += [P('山猫路面与楼梯分割：方法对比报告', 'title'),
          P('2026-11-11 · 实机图片来自山猫 Gemini 330 彩色相机 · 计算在 RTX 4090 D 上离线完成', 'sub'),
          Spacer(1, 10), P('一句话结论', 'h1'),
          P('推荐用<b>微调后的 SegFormer-B0（4 类）</b>：一个约 370 万参数的小模型，同时认出路面和楼梯；'
            '在没参与训练的山猫图上路面全部认出，楼梯认出数量比现成模型多一倍以上。'
            '它的短板是“站在楼梯顶往下看”仍然认不全，下楼需要深度相机或雷达来把关。'),
          P('主要发现', 'h1'),
          bullets([
              '现部署的 <b>PIDNet-M</b> 在图片底部用“复制最后一行”补边，会让离机器人最近的路面把握下降。按部署的 0.7 门槛，100 张图里 30 张把正前方的路漏掉一半以上；改成补黑色降到 8 张。'
              '但从整张图看，两种补边有 93.6% 的像素结果相同，差别集中在最底部约 60 行（见第 4 节）。',
              '即使补边改好，PIDNet 在强光逆光的画面里仍会把整片马路认成植被（见例子 L3）。',
              '<b>SegFormer-B0 城市街景版</b>认路最稳，100 张 0 张漏认，但它没有“楼梯”类别。',
              '没有楼梯类别的模型（PIDNet、城市街景版）会把楼梯直接认成“马路”（见例子 S1、S4），对导航来说这比“认不出”更危险。',
              '<b>SegFormer-B0 ADE20K 版</b>有楼梯类别，但认路较差（12 张漏认），楼梯也常常只认出一部分。',
              '<b>微调后的 B0</b>：留出的 40 张山猫图 0 张漏认；ADE20K 标注图上楼梯认出 18→50 张（共 99 张），楼梯范围重合度 0.19→0.39；代价是没有楼梯的图约 2% 会误报楼梯。',
          ]),
          PageBreak()]

# ---------------------------------------------------------------- methods
story += [P('1  对比了哪些方法', 'h1'),
          P('五种主要方法如下。另外还顺带测过 SegFormer-B1、B2 和“PIDNet 放大两倍”，结果列在第 3 节的表里。'),
          table([['方法', '训练数据', '类别', '大小（百万参数）', '图片怎么送进模型'],
                 ['PIDNet-M 现部署', 'Cityscapes 城市街景', '19 类', '28.5', '640×480 原图底部复制最后一行 32 次，补成 640×512（模型要求长宽是 64 的倍数）'],
                 ['PIDNet-M 补黑边', '同上', '19 类', '28.5', '底部补 32 行黑色，其余相同'],
                 ['SegFormer-B0 城市街景', 'Cityscapes 城市街景', '19 类', '3.7', '原图直接输入（要求 32 的倍数，480 已满足，不用补边）'],
                 ['SegFormer-B0 ADE20K', 'ADE20K 通用场景', '150 类（含楼梯、草地、泥土）', '3.8', '原图直接输入'],
                 ['SegFormer-B0 微调', 'ADE20K + 前 60 张山猫图', '4 类：硬质路面、草地泥土、楼梯、其他', '3.7', '原图直接输入']],
                [.19, .17, .17, .11, .36], {1: BAD, 5: GOOD}),
          Spacer(1, 6),
          P('微调是怎么做的', 'h2'),
          bullets([
              '起点：SegFormer-B0 ADE20K 版（本来就认识楼梯）。把 150 类合并成 4 类：马路、人行道、小路、跑道、土路、地面 → 硬质路面；草地、泥土、田地、沙地、土地 → 草地泥土；楼梯、台阶 → 楼梯；其余 → 其他。',
              '训练数据：ADE20K 训练集中 1238 张有楼梯的图、4067 张有路面的图、随机 3000 张其他图；外加前 60 张山猫图。',
              '山猫图没有人工标注，用 SegFormer-B1 城市街景版自动生成标注（它在 100 张里认路 0 漏认），把握不足 0.7 的像素不参与训练。',
              '每批 16 张，随机缩放、裁剪 512×512、左右翻转、调亮度；楼梯类的损失权重加倍。训练 8000 步，4090 上约 16 分钟。',
              '测试时只用从未参与训练的图：山猫后 40 张、ADE20K 验证集、网上楼梯照片。',
          ]),
          PageBreak()]

# ---------------------------------------------------------------- evaluation method
story += [P('2  评估是怎么做的', 'h1'),
          P('2.1  测试图片', 'h2'),
          table([['图片', '数量', '有没有人工标注', '用来测什么'],
                 ['山猫实机图（Gemini 330，640×480，同一段校园柏油路约 11 分钟）', '100 张；微调时前 60 张训练、后 40 张只做测试', '没有', '认路是否稳定；会不会把路误报成楼梯'],
                 ['ADE20K 验证集中楼梯占画面 >1% 的图', '99 张', '有（逐像素）', '楼梯认得准不准'],
                 ['ADE20K 验证集中完全没有楼梯的图（随机抽取）', '300 张', '有', '楼梯误报'],
                 ['网上室外楼梯照片（Wikimedia Commons，CC 授权）', '29 张（剔除下载失败和没有楼梯的）', '没有', '真实室外楼梯能否认出，含往下看的视角']],
                [.36, .24, .14, .26]),
          P('2.2  指标', 'h2'),
          table([['指标', '怎么算', '含义'],
                 ['前方路面认出率', '每张山猫图底部 120 行、中间 320 列（机器人正前方地面）里，被判为马路或人行道、且把握 ≥0.7 的像素比例，再对所有图取平均', '录制时机器人一直走在路上，这块区域应该全是路，越接近 100% 越好'],
                 ['漏认张数', '前方路面认出率 <50% 的图片张数', '越少越好'],
                 ['楼梯重合度（IoU）', '在 ADE20K 标注图上：预测楼梯与人工标注楼梯的交集 ÷ 并集', '0 到 1，越大说明楼梯的范围画得越准'],
                 ['楼梯认出', '一张图里被判为楼梯的面积 >1% 就算认出', '越多越好'],
                 ['楼梯误报', '没有楼梯的图里被判为楼梯的面积 >1%', '越少越好'],
                 ['把握门槛 0.7', '模型对某个像素最有把握的类别概率 <0.7 时，记为“拿不准”', '与导航分支部署代码一致']],
                [.2, .5, .3]),
          P('2.3  这套评估的局限', 'h2'),
          bullets([
              '山猫图没有人工标注，只能发现“该是路却没认出”，发现不了“把别的东西认成路”。',
              '“正前方”区域紧贴图片底部，正好是补边影响最大的位置；0.7 门槛又刚好卡在 PIDNet 把握下降的位置。两者叠加放大了补边造成的差别（第 4 节有不设门槛、按图片位置拆开的结果）。',
              '100 张山猫图来自同一条路的同一段录像，场景单一；前 60 张和后 40 张也很相似，对微调模型偏有利。',
              'ADE20K 验证集与 ADE20K 训练数据风格相同，对 ADE20K 版和微调模型偏有利；网上照片能部分弥补这一点。',
              '第 5 节的例子是挑选出来说明问题的，不代表平均水平。',
              '速度只在 4090 上粗测，大小不同的模型都在 20 毫秒左右，主要是程序开销，结果不可信；需要在山猫的 Jetson 上实测。',
          ]),
          PageBreak()]

# ---------------------------------------------------------------- results
story += [P('3  结果', 'h1'),
          P('3.1  认路（山猫实机图，全部 100 张，把握门槛 0.7）', 'h2'),
          table([['方法', '前方路面认出率', '漏认张数（/100）'],
                 ['PIDNet-M 现部署', '63.0%', '30'], ['PIDNet-M 补黑边', '91.1%', '8'],
                 ['PIDNet-M 放大两倍（不需要补边）', '93.2%', '8'],
                 ['SegFormer-B0 城市街景', '99.9%', '0'], ['SegFormer-B1 城市街景', '100.0%', '0'],
                 ['SegFormer-B0 ADE20K', '86.9%', '12'], ['SegFormer-B1 ADE20K', '80.4%', '18'],
                 ['SegFormer-B2 ADE20K', '95.3%', '3']],
                [.5, .25, .25], {1: BAD, 4: GOOD}),
          Spacer(1, 6),
          P('微调模型用前 60 张训练过，所以只在后 40 张上和其他方法比：'),
          table([['方法（后 40 张）', '前方路面认出率', '漏认张数（/40）', '误报楼梯（/40）'],
                 ['SegFormer-B0 城市街景', '100.0%', '0', '没有楼梯类别'],
                 ['SegFormer-B0 ADE20K', '85.0%', '5', '0'],
                 ['SegFormer-B0 微调', '100.0%', '0', '0']],
                [.4, .2, .2, .2], {3: GOOD}),
          P('3.2  认楼梯', 'h2'),
          table([['方法', '楼梯重合度 IoU', '标注图认出（/99）', '无楼梯图误报（/300）', '网上照片认出（/29）'],
                 ['SegFormer-B0 ADE20K', '0.19', '18', '0', '19'], ['SegFormer-B1 ADE20K', '0.27', '31', '1', '19'],
                 ['SegFormer-B2 ADE20K', '0.29', '38', '0', '23'], ['SegFormer-B0 微调', '0.39', '50', '6', '22']],
                [.3, .16, .18, .18, .18], {4: GOOD}),
          P('PIDNet 和城市街景版没有“楼梯”类别，无法参与这项比较。看图发现，它们会把楼梯认成“马路”（例子 S1、S4）。', 'cap'),
          P('3.3  看图总结出的规律', 'h2'),
          bullets([
              '<b>从下往上看楼梯</b>：ADE20K 版和微调版都能认出，微调版画出的范围更完整（S1、S2、A1、A2）。',
              '<b>从上往下看楼梯</b>：所有方法都认不全。台阶的竖面被挡住，只看到一层层平面，看起来像一块斜坡；网上和 ADE20K 里这种视角的照片也很少（S3、S4）。',
              '<b>其他类别</b>：几种方法对天空、建筑、树木的判断大体一致。微调版只有 4 类，其余都归为“其他”，对导航够用，但不能用来识别行人、车辆等。',
          ]),
          PageBreak()]

# ---------------------------------------------------------------- padding
story += [P('4  为什么底部补边会让 PIDNet 漏认', 'h1'),
          P('PIDNet 要求图片长宽是 64 的倍数。部署代码把 640×480 底部补 32 行凑成 640×512，补的方法是把最后一行像素复制 32 遍。'
            '复制出来的是一排竖条纹（下图），更像栏杆或草丛，不像路面。PIDNet 判断一个像素时会参考周围一大片，'
            '这块假纹理紧贴机器人正前方，让附近路面的把握从约 0.88 降到约 0.71，大量像素落到 0.7 门槛以下变成“拿不准”，一部分被认成植被。'),
          fig('padding_examples_zoom.jpg', '放大图：红线以上是真实路面，红线以下是复制出来的 32 行。', 40 * mm),
          P('差别有多大，取决于怎么统计', 'h2'),
          table([['把握门槛', '复制补边：前方认出', '复制补边：漏认', '补黑色：前方认出', '补黑色：漏认'],
                 ['不设门槛', '83.2%', '12', '95.1%', '6'], ['0.5', '82.5%', '12', '94.6%', '6'],
                 ['0.6', '75.4%', '19', '93.2%', '8'], ['0.7（部署值）', '63.0%', '30', '91.1%', '8'],
                 ['0.8', '42.9%', '52', '86.8%', '10']],
                [.2, .2, .2, .2, .2], {4: BAD}),
          Spacer(1, 6),
          table([['图片位置（行号，共 480 行）', '复制补边：认成路', '补黑色：认成路', '平均把握（复制 / 黑）'],
                 ['240–300', '70.1%', '71.9%', '0.87 / 0.86'], ['300–360', '81.3%', '81.0%', '0.88 / 0.87'],
                 ['360–420', '85.1%', '89.6%', '0.81 / 0.87'], ['420–450', '76.0%', '93.2%', '0.71 / 0.88'],
                 ['450–480（最底部）', '71.4%', '99.0%', '0.71 / 0.96']],
                [.3, .23, .23, .24]),
          P('第二张表不设门槛。整张图上两种补边 93.6% 的像素结果相同，路面区域重合度 0.885。结论：补边的影响是真实的，但只在最底部约 60 行；'
            '“30 张对 8 张”是部署门槛下机器人实际会遇到的情况，不能理解成 PIDNet 整体差这么多。', 'cap'),
          PageBreak(),
          fig('padding_examples.jpg', '差别最大的 4 张（专门挑出来的，所以看起来差别比平均情况大）。每行从左到右：复制补边的输入、结果，补黑色的输入、结果。'
              '白框是统计用的“正前方”区域，红虚线以下是补出来的 32 行；绿色马路，灰色拿不准，橙色植被。', 235 * mm,
              heading='补边对比例子'),
          PageBreak()]

# ---------------------------------------------------------------- examples
legend = ('颜色说明：所有方法尽量用同一种颜色表示同一类东西，例如马路紫色、人行道粉色、植被和树橄榄绿、草地浅绿、泥土棕色、'
          '楼梯橙色、天空蓝色、建筑深灰。灰色表示把握不到 0.7（拿不准）。微调模型只有 4 类，“其他”为深灰。'
          '每张小图下方列出面积最大的几个类别及其占画面的比例。')
story += [P('5  可视化例子', 'h1'), P(legend),
          P('5.1  山猫实机图（均来自后 40 张，微调时没用过）', 'h2'),
          fig('fig_lynx_066.jpg', 'L1（第 66 张）树荫斑驳的路。各方法都认出了大部分马路；两种 PIDNet 补边看起来几乎一样；ADE20K 版约四分之一像素拿不准；微调版把路面几乎全部标为硬质路面。'),
          fig('fig_lynx_089.jpg', 'L2（第 89 张）强光下的路。现部署 PIDNet 马路只占 12%，近处大片被认成植被或拿不准；补黑边后马路升到 31%，但仍有不少植被；三个 SegFormer 都认出了完整路面。'),
          fig('fig_lynx_096.jpg', 'L3（第 96 张）与 L2 相似的强光画面。PIDNet 无论哪种补边，都把大半马路认成植被，说明这里补边不是唯一原因，模型本身在这种光照下就不稳；SegFormer 城市街景版和微调版都认对了。'),
          fig('fig_web_042.jpg', 'S1 从下往上看的台阶。PIDNet 和城市街景版把整段台阶认成“马路”；ADE20K 版和微调版都正确标出楼梯。',
              heading='5.2  网上室外楼梯照片'),
          fig('fig_web_013.jpg', 'S2 很长的一段上坡台阶。ADE20K 版只认出零散几块楼梯；微调版完整标出整段台阶，底部广场标为硬质路面。'),
          fig('fig_web_038.jpg', 'S3 站在楼梯顶往下看。所有方法都没认出这是楼梯：PIDNet 和城市街景版认成马路或拿不准，ADE20K 版认成地面，微调版只在右下角标出一小块。'),
          fig('fig_web_041.jpg', 'S4 往下走进水池的楼梯。PIDNet 和城市街景版把台阶认成马路；微调版只认出两小块楼梯，还把近处台阶误认成草地泥土。往下看仍是最大的短板。'),
          fig('fig_ade_00000588.jpg', 'A1 石头房子前的台阶。人工标注里楼梯占 12%、人行道 13%。城市街景版把台阶下部认成人行道，ADE20K 版只认出一小块楼梯，微调版标出的楼梯范围最接近标注。', 150 * mm,
              heading='5.3  ADE20K 验证集（有人工标注，可以对照“标准答案”）'),
          fig('fig_ade_00000017.jpg', 'A2 两墙之间的窄台阶。PIDNet 和城市街景版都没认出楼梯；ADE20K 版和微调版都基本标对了台阶范围。', 175 * mm),
          PageBreak()]

# ---------------------------------------------------------------- conclusion
story += [P('6  建议与下一步', 'h1'),
          bullets([
              '<b>近期</b>：若暂时保留 PIDNet，把 vendor/semantic_local_grid/inference.py 中的 BORDER_REPLICATE 改成补黑色，一行改动，不增加计算量。',
              '<b>模型</b>：用微调后的 SegFormer-B0（4 类）替换 PIDNet。导航分支已支持替换模型，需要写一个适配器，把“硬质路面”当作道路、把“楼梯”交给楼梯确认逻辑。',
              '<b>数据</b>：让山猫在楼梯顶往下拍一批图，再拍一些台阶、草坪边、土路，各标注几十张后重新微调。这是提升“往下看楼梯”的唯一可靠办法。',
              '<b>下楼安全</b>：下楼时以深度相机或雷达检测“前方地面突然下沉”为主，相机分割只作参考；保持“相机和雷达都确认才放行”的设计。',
              '<b>评估</b>：挑 20 到 30 张山猫图人工圈出路面和楼梯，在整张图上计算重合度，同时报告不设门槛和 0.7 门槛两种结果，避免第 2.3 节提到的放大效应。',
              '<b>速度</b>：在山猫 Jetson 上用 TensorRT 测 B0 的实际速度；不够快可把输入缩小到 320×240（测试中路面仍 0 漏认）。',
          ]),
          Spacer(1, 10),
          P('附：文件位置（均不进 git）', 'h2'),
          P('seg_compare/ 下：padding/（补边分析）、stairs/（楼梯测试与照片来源 sources.csv）、finetune/（微调模型 ft_b0、训练与测试脚本、结果）、pdf_report/（本报告与配图）。'),
          P('图片来源：山猫实机录像；ADE20K（MIT CSAIL）；网上楼梯照片来自 Wikimedia Commons，各自为 CC BY / CC BY-SA 授权，具体作者见 sources.csv。', 'cap')]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm,
                        bottomMargin=16 * mm, title='山猫路面与楼梯分割对比报告', author='')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
