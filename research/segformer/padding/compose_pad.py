import json, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]
F = lambda s: ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', s)
summ = json.load(open(src + '/summary.json'))
COL = {0: (0, 200, 0), 1: (120, 230, 120), 8: (255, 140, 0)}   # road, sidewalk, vegetation
OTHER, UNSURE = (90, 140, 230), (150, 150, 150)

def color(pad, cls, conf):
    img = pad.astype(np.float32)
    lay = np.zeros_like(img); lay[:] = OTHER
    for k, v in COL.items(): lay[cls == k] = v
    lay[conf < .7] = UNSURE
    out = (img * .45 + lay * .55).astype(np.uint8)
    return out

def mark(im, title):
    d = ImageDraw.Draw(im)
    d.rectangle([160, 360, 480, 480], outline=(255, 255, 255), width=3)          # front region
    for x in range(0, 640, 16): d.line([x, 480, x + 8, 480], fill=(255, 40, 40), width=3)   # pad boundary
    d.rectangle([0, 0, 640, 40], fill=(0, 0, 0))
    d.text((10, 6), title, font=F(24), fill=(255, 255, 255))
    return im

W, H, CAP = 640, 512, 70
legend_h = 60
rows = []
for s in summ:
    i = s['frame']
    tiles = []
    for tag, name in [('rep', '复制边缘'), ('blk', '补黑色')]:
        z = np.load(f'{src}/f{i:03d}_{tag}.npz')
        tiles.append(mark(Image.fromarray(z['pad']), f'输入：底部 32 行{name}'))
        tiles.append(mark(Image.fromarray(color(z['pad'], z['cls'], z['conf'])), f'PIDNet 结果（{name}）'))
    row = Image.new('RGB', (W * 4, H + CAP), (255, 255, 255))
    for k, t in enumerate(tiles): row.paste(t, (k * W, 0))
    r, b = s['rep'], s['blk']
    br = r['front_breakdown']
    txt = (f"第 {i} 张：前方路面认出  复制边缘 {r['front_ok']*100:.0f}%  →  补黑色 {b['front_ok']*100:.0f}%。"
           f"复制边缘时，前方 {br.get('unsure(<0.7)',0)*100:.0f}% 拿不准、{br.get('vegetation',0)*100:.0f}% 被认成植被；"
           f"路面平均把握 {r['road_conf_mean_front']:.2f} → {b['road_conf_mean_front']:.2f}")
    ImageDraw.Draw(row).text((14, H + 18), txt, font=F(26), fill=(30, 30, 30))
    rows.append(row)
sheet = Image.new('RGB', (W * 4, legend_h + sum(r.height for r in rows)), (255, 255, 255))
d = ImageDraw.Draw(sheet); x = 14
for c, n in [((0, 200, 0), '马路'), ((120, 230, 120), '人行道'), ((255, 140, 0), '植被'), (UNSURE, '拿不准（把握<0.7）'),
             (OTHER, '其他类别'), ((255, 255, 255), '白框：机器人正前方'), ((255, 40, 40), '红虚线：以下是补出来的 32 行')]:
    d.rectangle([x, 18, x + 26, 44], fill=c, outline=(80, 80, 80)); d.text((x + 34, 16), n, font=F(24), fill=(30, 30, 30))
    x += 34 + int(d.textlength(n, font=F(24))) + 30
y = legend_h
for r in rows: sheet.paste(r, (0, y)); y += r.height
sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS)
sheet.save(out, quality=88)
# zoom on the padded strip of the first example
z = np.load(f"{src}/f{summ[0]['frame']:03d}_rep.npz")['pad']
crop = Image.fromarray(z[400:512, 0:640]).resize((1280, 224), Image.NEAREST)
dz = ImageDraw.Draw(crop); dz.line([0, 160, 1280, 160], fill=(255, 40, 40), width=3)
dz.text((10, 6), '放大看：红线以上是真实路面，红线以下是把最后一行复制 32 遍', font=F(26), fill=(255, 255, 0))
crop.save(out.replace('.jpg', '_zoom.jpg'), quality=90)
print(sheet.size)
