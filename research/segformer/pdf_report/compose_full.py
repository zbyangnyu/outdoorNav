"""Colour full-class predictions, add per-panel legends, one figure per example image."""
import json, sys, zlib, numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC, DST = sys.argv[1], sys.argv[2]
FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
F = lambda s: ImageFont.truetype(FONT, s)
L = json.load(open(f'{SRC}/labels.json'))

ZH = {'road': '马路', 'sidewalk': '人行道', 'building': '建筑', 'wall': '墙', 'fence': '栅栏', 'pole': '杆子',
      'traffic light': '红绿灯', 'traffic sign': '交通标志', 'vegetation': '植被', 'terrain': '草地土地', 'sky': '天空',
      'person': '人', 'rider': '骑车人', 'car': '汽车', 'truck': '卡车', 'bus': '公交车', 'train': '火车',
      'motorcycle': '摩托车', 'bicycle': '自行车', 'floor': '地面', 'tree': '树', 'ceiling': '天花板', 'grass': '草地',
      'earth': '泥土', 'plant': '植物', 'house': '房子', 'field': '田地', 'railing': '栏杆', 'rock': '岩石',
      'stairs': '楼梯', 'stairway': '楼梯', 'step': '台阶', 'path': '小路', 'water': '水', 'sea': '海', 'river': '河',
      'mountain': '山', 'column': '柱子', 'signboard': '招牌', 'sand': '沙地', 'skyscraper': '高楼', 'streetlight': '路灯',
      'bench': '长椅', 'flower': '花', 'palm': '棕榈', 'hill': '山坡', 'dirt track': '土路', 'land': '土地',
      'runway': '跑道', 'door': '门', 'windowpane': '窗', 'base': '底座', 'pot': '花盆', 'box': '箱子', 'awning': '雨棚',
      'other': '其他', 'paved': '硬质路面', 'grass_soil': '草地泥土', 'sidewalk, pavement': '人行道', 'pier': '码头',
      'bridge': '桥', 'fountain': '喷泉', 'swimming pool': '泳池', 'canopy': '顶棚', 'ashcan': '垃圾桶', 'bannister': '扶手',
      'stage': '台子', 'tower': '塔', 'booth': '亭子', 'hovel': '小屋', 'wall ': '墙', 'cabinet': '柜子', 'person ': '人'}
UNSURE = (150, 150, 150)


def zh(n):
    key = n.split(';')[0].split(',')[0].strip()
    return ZH.get(n, ZH.get(key, key))


def color(n):
    k = n.split(';')[0].split(',')[0].strip()
    groups = [(('road', 'paved'), (128, 64, 128)), (('sidewalk', 'path', 'runway'), (244, 35, 232)),
              (('floor',), (190, 140, 170)), (('building', 'house', 'skyscraper', 'tower'), (70, 70, 70)),
              (('wall',), (102, 102, 156)), (('fence', 'railing', 'bannister'), (190, 153, 153)),
              (('pole', 'streetlight', 'column'), (153, 153, 153)), (('traffic light',), (250, 210, 120)),
              (('traffic sign', 'signboard'), (220, 220, 0)),
              (('vegetation', 'tree', 'plant', 'palm', 'flower'), (107, 142, 35)),
              (('terrain', 'grass', 'field', 'grass_soil'), (152, 251, 152)),
              (('earth', 'sand', 'land', 'dirt track', 'hill', 'mountain'), (170, 130, 80)), (('rock',), (120, 100, 85)),
              (('sky',), (70, 130, 180)), (('person', 'rider'), (220, 20, 60)), (('car',), (0, 0, 142)),
              (('truck', 'bus', 'train'), (0, 60, 100)), (('motorcycle', 'bicycle'), (119, 11, 32)),
              (('stairs', 'stairway', 'step'), (255, 120, 0)), (('water', 'sea', 'river', 'swimming pool'), (0, 170, 210)),
              (('other',), (55, 55, 55))]
    for names, col in groups:
        if k in names:
            return col
    h = zlib.crc32(k.encode())
    return (90 + h % 120, 90 + (h >> 8) % 120, 90 + (h >> 16) % 120)


METHODS = {'pidnet_rep': ('PIDNet-M 现部署（复制补边）', 'cs'), 'pidnet_blk': ('PIDNet-M 补黑边', 'cs'),
           'b0_cs': ('SegFormer-B0 城市街景', 'cs'), 'b0_ade': ('SegFormer-B0 ADE20K', 'ade'),
           'b0_ft': ('SegFormer-B0 微调（4 类）', 'ft')}
PW = 560   # panel width


def panel(rgb, lab, names, title, conf=None, gt=False):
    h, w = lab.shape
    cols = np.array([color(n) for n in names] + [(0, 0, 0)], np.float32)
    idx = np.where(lab < 0, len(names), lab)
    lay = cols[idx]
    keys = idx.copy()
    if conf is not None:
        uns = conf.astype(np.float32) < .7
        lay[uns] = UNSURE
        keys = np.where(uns, -2, keys)
    out = (rgb.astype(np.float32) * .4 + lay * .6).astype(np.uint8)
    im = Image.fromarray(out)
    s = PW / w
    im = im.resize((PW, round(h * s)), Image.LANCZOS)
    # legend: top classes by area (merge same Chinese name)
    vals, cnt = np.unique(keys, return_counts=True)
    agg = {}
    for v, n in zip(vals, cnt):
        if v == -2: name, col = '拿不准', UNSURE
        elif v == len(names): name, col = '未标注', (0, 0, 0)
        else: name, col = zh(names[v]), color(names[v])
        a = agg.setdefault(name, [0, col]); a[0] += n
    top = sorted(agg.items(), key=lambda t: -t[1][0])[:6]
    leg_h = 64
    title_h = 34
    can = Image.new('RGB', (PW, title_h + im.height + leg_h), (255, 255, 255))
    d = ImageDraw.Draw(can)
    d.rectangle([0, 0, PW, title_h], fill=(30, 36, 33))
    d.text((10, 6), title, font=F(20), fill=(255, 255, 255))
    can.paste(im, (0, title_h))
    tot = sum(a[0] for _, a in top) or 1
    tot = keys.size
    for j, (name, (n, col)) in enumerate(top):
        x, y = 8 + (j % 3) * (PW // 3), title_h + im.height + 8 + (j // 3) * 28
        d.rectangle([x, y + 3, x + 18, y + 21], fill=col, outline=(90, 90, 90))
        d.text((x + 24, y + 1), f'{name} {n / tot * 100:.0f}%', font=F(18), fill=(30, 30, 30))
    return can


def figure(key):
    rgb = np.array(Image.open(f'{SRC}/{key}_rgb.jpg').convert('RGB'))
    h, w = rgb.shape[:2]
    tiles = []
    orig = Image.fromarray(rgb).resize((PW, round(h * PW / w)), Image.LANCZOS)
    o = Image.new('RGB', (PW, 34 + orig.height + 64), (255, 255, 255))
    d = ImageDraw.Draw(o); d.rectangle([0, 0, PW, 34], fill=(30, 36, 33))
    d.text((10, 6), '原图', font=F(20), fill=(255, 255, 255)); o.paste(orig, (0, 34))
    tiles.append(o)
    if key.startswith('ade'):
        gt = np.load(f'{SRC}/{key}_gt.npy')
        tiles.append(panel(rgb, gt, L['ade'], '人工标注（标准答案）'))
    for mk, (title, space) in METHODS.items():
        try:
            z = np.load(f'{SRC}/{key}__{mk}.npz')
        except FileNotFoundError:
            continue
        if mk == 'pidnet_rep' and not key.startswith('lynx'):
            title = 'PIDNet-M（无需补边）'
        tiles.append(panel(rgb, z['cls'].astype(np.int32), L[space], title, z['conf']))
    cols = 3
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    fig = Image.new('RGB', (cols * PW + (cols - 1) * 12, rows * th + (rows - 1) * 12), (255, 255, 255))
    for i, t in enumerate(tiles):
        fig.paste(t, ((i % cols) * (PW + 12), (i // cols) * (th + 12)))
    fig.save(f'{DST}/fig_{key}.jpg', quality=88)
    print(key, fig.size)


for k in L['images']:
    figure(k)
