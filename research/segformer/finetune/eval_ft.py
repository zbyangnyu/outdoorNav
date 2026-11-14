"""Compare fine-tuned 4-class B0 with stock B0 (Cityscapes / ADE20K) on held-out data.

Lynx frames 60-99 (never trained on), ADE20K validation stairs + 300 stair-free images,
29 web stair photos. Same 0.7 confidence rule as deployment.
"""
import glob, json, os, sys, numpy as np, cv2, torch
sys.path.insert(0, '/root/autodl-tmp/seg')
import compare as c
from transformers import SegformerForSemanticSegmentation

ADE = '/root/autodl-tmp/ade/ADEChallengeData2016'
WEB = '/root/autodl-tmp/seg/stairs_web'
OUT = '/root/autodl-tmp/seg/out_ft'
GT_STAIR = [53, 59, 121]


class M:
    def __init__(self, path, paved, soft, stair):
        self.m = SegformerForSemanticSegmentation.from_pretrained(path).to(c.dev).eval()
        self.lut = np.zeros(self.m.config.num_labels, np.uint8)
        for k in paved: self.lut[k] = 1
        for k in soft: self.lut[k] = 2
        for k in stair: self.lut[k] = 3

    def cat(self, rgb, short=None):
        h, w = rgb.shape[:2]
        if short:
            s = short / min(h, w)
            x = cv2.resize(rgb, (max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)))
        else:
            x = rgb                                    # Lynx 640x480 is already a multiple of 32
        x = (x.astype(np.float32) / 255 - c.MEAN) / c.STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(c.dev)
        with torch.inference_mode():
            y = torch.nn.functional.interpolate(self.m(pixel_values=x).logits, size=(h, w), mode='bilinear',
                                                align_corners=False)
            conf, cls = y.softmax(1).max(1)
        out = self.lut[cls[0].cpu().numpy()]
        out[conf[0].cpu().numpy() < c.THR] = 255
        return out


def hf(name):
    return glob.glob(f'{c.HF}/models--nvidia--segformer-{name}/snapshots/*')[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    lynx = sorted(glob.glob(c.IMG_DIR + '/*.png'))[60:]
    lynx_imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in lynx]
    # ADE validation split, same selection as stairs_eval.py
    pos, neg = [], []
    for a in sorted(glob.glob(f'{ADE}/annotations/validation/*.png')):
        g = cv2.imread(a, cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
        f = np.isin(g, GT_STAIR).mean()
        if f > .01: pos.append((a, g))
        elif f == 0: neg.append(a)
    neg = list(np.random.default_rng(0).choice(neg, 300, replace=False))
    web = sorted(glob.glob(WEB + '/*.jpg'))
    def img_of(a): return cv2.cvtColor(cv2.imread(a.replace('annotations', 'images').replace('.png', '.jpg')), cv2.COLOR_BGR2RGB)

    specs = [('B0 城市街景', lambda: M(hf('b0-finetuned-cityscapes-1024-1024'), [0, 1], [9], [])),
             ('B0 ADE20K', lambda: M(hf('b0-finetuned-ade-512-512'), [6, 11, 52, 54, 91], [9, 13, 29, 46, 94], GT_STAIR)),
             ('B0 微调', lambda: M('/root/autodl-tmp/seg/ft_b0', [1], [2], [3]))]
    stats, tiles_l, tiles_w = {}, {}, {}
    for name, make in specs:
        m = make()
        cats = [m.cat(im) for im in lynx_imgs]
        front = np.array([(k[-120:, 160:480] == 1).mean() for k in cats])
        s = {'lynx_front_paved': float(front.mean()), 'lynx_bad': int((front < .5).sum()),
             'lynx_stair_false': int(sum((k == 3).mean() > .01 for k in cats))}
        tiles_l[name] = cats
        if name != 'B0 城市街景':
            inter = union = hit = 0
            for a, g in pos:
                p = m.cat(img_of(a), 512) == 3
                gs, valid = np.isin(g, GT_STAIR), g >= 0
                inter += (p & gs & valid).sum(); union += ((p | gs) & valid).sum(); hit += p.mean() > .01
            fa = sum((m.cat(img_of(a), 512) == 3).mean() > .01 for a in neg)
            wc = [m.cat(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), 480) for f in web]
            tiles_w[name] = wc
            s.update({'ade_stair_iou': float(inter / max(union, 1)), 'ade_stair_hit': int(hit), 'ade_pos': len(pos),
                      'ade_false_alarm': int(fa), 'ade_neg': len(neg),
                      'web_hit': int(sum((k == 3).mean() > .01 for k in wc)), 'web_n': len(web)})
        stats[name] = s
        print(name, s, flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(stats, open(OUT + '/stats.json', 'w'), indent=1, ensure_ascii=False)
    en = {'B0 城市街景': 'B0 Cityscapes', 'B0 ADE20K': 'B0 ADE20K', 'B0 微调': 'B0 fine-tuned'}
    for i, f in enumerate(lynx):
        row = [c.overlay(lynx_imgs[i], np.zeros(lynx_imgs[i].shape[:2], np.uint8), 'original')]
        row += [c.overlay(lynx_imgs[i], tiles_l[n][i], en[n]) for n in tiles_l]
        g = np.vstack([np.hstack(row[:2]), np.hstack(row[2:])])
        cv2.imwrite(f'{OUT}/lynx_{60 + i:03d}.jpg', cv2.resize(g, None, fx=.6, fy=.6), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for i, f in enumerate(web):
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        row = [c.overlay(img, np.zeros(img.shape[:2], np.uint8), 'original')]
        row += [c.overlay(img, tiles_w[n][i], en[n]) for n in tiles_w]
        g = np.hstack(row)
        cv2.imwrite(f'{OUT}/{os.path.basename(f)}', cv2.resize(g, None, fx=.5, fy=.5), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('EVAL_DONE', flush=True)


if __name__ == '__main__':
    main()
