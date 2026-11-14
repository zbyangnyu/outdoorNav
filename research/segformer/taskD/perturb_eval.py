"""Step 1: synthetic adverse conditions on the 100 Lynx frames, scored against taskB refs.
usage: python perturb_eval.py  (writes /root/autodl-tmp/seg/taskD/perturb/)"""
import glob, json, os, sys
import numpy as np, cv2, torch
sys.path.insert(0, '/root/autodl-tmp/seg'); sys.path.insert(0, '/root/autodl-tmp/seg/taskB')
import compare as C
from evaluate import counts, metrics
ROOT = '/root/autodl-tmp/seg'; OUT = ROOT + '/taskD/perturb'
MODELS = {'PIDNet-M (deployed 1x)': lambda: C.PIDNet(1.0),
          'SegF-B0 CS': lambda: C.SegF('b0-finetuned-cityscapes-1024-1024'),
          'SegF-B1 CS': lambda: C.SegF('b1-finetuned-cityscapes-1024-1024'),
          'SegF-B0 ADE': lambda: C.SegF('b0-finetuned-ade-512-512'),
          'SegF-B1 ADE': lambda: C.SegF('b1-finetuned-ade-512-512'),
          'SegF-B2 ADE': lambda: C.SegF('b2-finetuned-ade-512-512')}

def dark(im, g):  # lower exposure + gamma + sensor noise like a dim scene
    x = (im / 255.) ** g * (1 / g); x = x + np.random.default_rng(0).normal(0, .02, x.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8)
def bright(im, k):  # overexposure / backlight: gain then clip, plus a glare blob top-centre
    x = im * k
    h, w = im.shape[:2]; yy, xx = np.mgrid[:h, :w]
    glare = np.exp(-((xx - w / 2) ** 2 + (yy - h * .15) ** 2) / (2 * (w * .3) ** 2))[..., None] * 255 * (k - 1)
    return np.clip(x + glare, 0, 255).astype(np.uint8)
def blur(im, k):  # horizontal+forward motion blur
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1; ker /= k
    return cv2.filter2D(im, -1, ker)
def noise(im, s):
    return np.clip(im + np.random.default_rng(1).normal(0, s, im.shape), 0, 255).astype(np.uint8)
def rain(im, n):
    rng = np.random.default_rng(2); h, w = im.shape[:2]; L = np.zeros((h, w), np.float32)
    for _ in range(n):
        x, y, l = rng.integers(0, w), rng.integers(0, h), rng.integers(10, 30)
        cv2.line(L, (x, y), (x + l // 5, y + l), 1, 1)
    L = cv2.GaussianBlur(L, (3, 3), 0)[..., None]
    x = im * .85 + L * 200  # slightly darker sky + streaks
    return np.clip(x, 0, 255).astype(np.uint8)
def fog(im, t):  # depth-ish: farther (higher in image) gets more fog
    h, w = im.shape[:2]; d = np.linspace(1, .3, h)[:, None, None]
    a = 1 - np.exp(-t * d * 3)
    return np.clip(im * (1 - a) + 220 * a, 0, 255).astype(np.uint8)

CONDS = [('clean', None, 0)] + [('dark', dark, g) for g in (1.8, 2.5, 3.5)] + \
        [('overexpose', bright, k) for k in (1.4, 1.8, 2.4)] + [('motion_blur', blur, k) for k in (9, 17, 31)] + \
        [('noise', noise, s) for s in (10, 25, 45)] + [('rain', rain, n) for n in (300, 800, 1600)] + \
        [('fog', fog, t) for t in (.3, .6, 1.0)]

def main():
    os.makedirs(OUT + '/samples', exist_ok=True)
    files = sorted(glob.glob(ROOT + '/camera_gemini330__color__image_raw/*.png'))
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    refs = [cv2.imread(ROOT + '/taskB/ref/' + os.path.basename(f), 0) for f in files]
    data = {}
    for c, fn, s in CONDS:
        data[f'{c}:{s}'] = imgs if fn is None else [fn(im.astype(np.float32), s) if c != 'motion_blur' else fn(im, s) for im in imgs]
    show = [5, 40, 80]
    res, preds = {}, {}
    for n, mk in MODELS.items():
        torch.cuda.empty_cache(); m = mk(); res[n] = {}
        for k, ims in data.items():
            tot = 0; ps = []
            for i, (im, r) in enumerate(zip(ims, refs)):
                p = C.predict(m, im); tot = tot + counts(p, r)
                if i in show: ps.append(p)
            res[n][k] = {q: float(v) for q, v in metrics(tot).items()}
            preds[(n, k)] = ps
        print(n, {k: round(v['iou'], 3) for k, v in res[n].items()}, flush=True)
        del m
    json.dump({'note': 'refs = taskB teacher masks on clean frames', 'conditions': list(data), 'models': res},
              open(OUT + '/metrics.json', 'w'), indent=1)
    # sample grids: for each condition (strongest level) original | PIDNet | B0 CS | B2 ADE
    for c in ['dark', 'overexpose', 'motion_blur', 'noise', 'rain', 'fog']:
        k = [x for x in data if x.startswith(c + ':')][-1]
        rows = []
        for j, i in enumerate(show):
            t = [C.overlay(data[k][i], np.zeros(refs[i].shape, np.uint8), k)]
            for n in ['PIDNet-M (deployed 1x)', 'SegF-B0 CS', 'SegF-B2 ADE']:
                p = preds[(n, k)][j]; v = C.overlay(data[k][i], p, f"{n.split(' (')[0]} {res[n][k]['iou']:.2f}")
                v[(p == 1) & (refs[i] == 0)] = (0, 0, 255); t.append(v)
            rows.append(np.hstack(t))
        g = cv2.resize(np.vstack(rows), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{OUT}/samples/{c}.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 85])

if __name__ == '__main__':
    main()
