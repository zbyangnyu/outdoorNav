"""Stair recognition test.

A. ADE20K validation images (pixel labels): stair IoU, stair image hit rate,
   false alarm rate on images without stairs.
B. Wikimedia outdoor stair photos: image hit rate + overlays.
Stair = ADE classes stairs(53) stairway(59) step(121), 0-based.
"""
import glob, json, os, numpy as np, cv2, torch, compare as c

ADE = '/root/autodl-tmp/ade/ADEChallengeData2016'
WEB = '/root/autodl-tmp/seg/stairs_web'
OUT = '/root/autodl-tmp/seg/out_stairs'
STAIR = [53, 59, 121]
ROAD = [6, 11, 52, 54, 91]   # road sidewalk path runway dirt-track (same as compare.ADE paved)


def run(model, rgb, short=512):
    """Resize so the short side ~= `short`, both sides multiples of 32 (no padding)."""
    h, w = rgb.shape[:2]
    s = short / min(h, w)
    W, H = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
    x = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255 - c.MEAN) / c.STD
    x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(c.dev)
    with torch.inference_mode():
        y = model.m(pixel_values=x).logits
        y = torch.nn.functional.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        conf, cls = y.softmax(1).max(1)
    cls, conf = cls[0].cpu().numpy(), conf[0].cpu().numpy()
    return cls, conf


def main():
    os.makedirs(OUT, exist_ok=True)
    ann = sorted(glob.glob(f'{ADE}/annotations/validation/*.png'))
    gts = {}
    for a in ann:
        g = cv2.imread(a, cv2.IMREAD_UNCHANGED).astype(np.int32) - 1   # 0-based, -1 = unlabeled
        frac = np.isin(g, STAIR).mean()
        if frac > .01:
            gts[a] = ('stairs', g)
        elif frac == 0 and len(gts) < 100000:
            gts.setdefault('_neg', []).append(a)
    neg = gts.pop('_neg')
    rng = np.random.default_rng(0)
    neg = list(rng.choice(neg, 300, replace=False))
    pos = list(gts)
    print('ADE val: stair images', len(pos), 'negatives sampled', len(neg), flush=True)
    web = sorted(glob.glob(WEB + '/*.jpg'))
    results = {}
    for name, mid in [('SegF-B0 ADE', 'b0-finetuned-ade-512-512'),
                      ('SegF-B1 ADE', 'b1-finetuned-ade-512-512'),
                      ('SegF-B2 ADE', 'b2-finetuned-ade-512-512')]:
        m = c.SegF(mid)
        inter = union = 0
        hit = 0
        for a in pos:
            img = cv2.cvtColor(cv2.imread(a.replace('annotations', 'images').replace('.png', '.jpg')), cv2.COLOR_BGR2RGB)
            cls, conf = run(m, img)
            p = np.isin(cls, STAIR) & (conf >= c.THR)
            g = np.isin(gts[a][1], STAIR)
            valid = gts[a][1] >= 0
            inter += (p & g & valid).sum(); union += ((p | g) & valid).sum()
            hit += p.mean() > .01
        fa = 0
        for a in neg:
            img = cv2.cvtColor(cv2.imread(a.replace('annotations', 'images').replace('.png', '.jpg')), cv2.COLOR_BGR2RGB)
            cls, conf = run(m, img)
            fa += (np.isin(cls, STAIR) & (conf >= c.THR)).mean() > .01
        whit, wfrac = 0, []
        for i, f in enumerate(web):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            cls, conf = run(m, img, 480)
            cat = np.zeros(cls.shape, np.uint8)
            cat[np.isin(cls, ROAD)] = 1
            cat[np.isin(cls, STAIR)] = 3
            cat[conf < c.THR] = 255
            fr = (cat == 3).mean(); wfrac.append(float(fr)); whit += fr > .01
            results.setdefault(i, []).append(c.overlay(img, cat, name))
        results.setdefault('_orig', None)
        r = {'ade_stair_iou': float(inter / max(union, 1)), 'ade_stair_hit': f'{hit}/{len(pos)}',
             'ade_false_alarm': f'{fa}/{len(neg)}', 'web_hit': f'{whit}/{len(web)}',
             'web_stair_frac_mean': float(np.mean(wfrac))}
        print(name, r, flush=True)
        results.setdefault('_stats', {})[name] = r
        del m; torch.cuda.empty_cache()
    for i, f in enumerate(web):
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        tiles = [c.overlay(img, np.zeros(img.shape[:2], np.uint8), 'original')] + results[i]
        h = min(t.shape[0] for t in tiles)
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
        g = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imwrite(f'{OUT}/{os.path.basename(f)}', cv2.resize(g, None, fx=.6, fy=.6, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(results['_stats'], open(OUT + '/stats.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
