"""Step 2: hard scenes (RUGD off-road + ADE20K outdoor stairs/grass) with real labels.
GT (4 classes): 1 paved walkable, 2 grass/soil/unpaved, 3 stairs, 0 other, 255 ignore.
usage: python hard_eval.py   -> /root/autodl-tmp/seg/taskD/hard/"""
import glob, json, os, sys
import numpy as np, cv2, torch
sys.path.insert(0, '/root/autodl-tmp/seg')
import compare as C
from perturb_eval import MODELS
D = '/root/autodl-tmp/seg/taskD'; OUT = D + '/hard'
RUGD = {(64, 64, 64): 1, (101, 101, 11): 1,                       # asphalt, concrete
        (108, 64, 20): 2, (255, 229, 204): 2, (0, 102, 0): 2, (255, 128, 0): 2, (153, 76, 0): 2,  # dirt sand grass gravel mulch
        (0, 0, 0): 255}                                            # void
def rugd_gt(f):
    a = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); g = np.zeros(a.shape[:2], np.uint8)
    for c, v in RUGD.items(): g[np.all(a == c, -1)] = v
    return g
def load(ds):
    if ds == 'RUGD':
        fs = sorted(glob.glob(D + '/data/rugd/img/*.png'))
        return fs, [rugd_gt(D + '/data/rugd/ann/' + os.path.basename(f)) for f in fs]
    fs = sorted(glob.glob(D + '/data/ade/img/*.png'))
    return fs, [cv2.imread(D + '/data/ade/gt/' + os.path.basename(f), 0) for f in fs]
def resize(im, g, long=1024):  # ADE images vary in size; cap long side for speed
    s = min(1., long / max(im.shape[:2]))
    if s < 1: im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); g = cv2.resize(g, im.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    return im, g
def acc(p, g):
    v = g != 255; P, G = (p == 1) & v, g == 1
    return np.array([(P & G).sum(), (P | G).sum(), ((g == 2) & (p == 1)).sum(), (g == 2).sum(), ((g == 3) & (p == 1)).sum(),
                     ((g == 3) & (p == 3)).sum(), (g == 3).sum(), ((g == 0) & (p == 1)).sum(), (g == 0).sum(),
                     ((g == 2) & (p == 2)).sum(), ((G) & (p == 255)).sum(), G.sum()], np.int64)
def mets(a):
    d = lambda x, y: float(x / y) if y else None
    return {'paved_iou': d(a[0], a[1]), 'grass_as_road': d(a[2], a[3]), 'stairs_as_road': d(a[4], a[6]),
            'stairs_recall': d(a[5], a[6]), 'other_as_road': d(a[7], a[8]), 'grass_recall': d(a[9], a[3]),
            'paved_unsure': d(a[10], a[11]), 'px_paved': int(a[11]), 'px_grass': int(a[3]), 'px_stairs': int(a[6])}
def main():
    os.makedirs(OUT + '/samples', exist_ok=True); res = {}
    for ds in ['RUGD', 'ADE20K']:
        fs, gs = load(ds); data = [resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), g) for f, g in zip(fs, gs)]
        res[ds] = {'frames': len(fs)}; preds = {}
        for n, mk in MODELS.items():
            torch.cuda.empty_cache(); m = mk(); tot = 0; ps = []
            for im, g in data:
                p = C.predict(m, im); tot = tot + acc(p, g); ps.append(p)
            res[ds][n] = mets(tot); preds[n] = ps; del m
            print(ds, n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[ds][n].items()}, flush=True)
        # pick samples: worst B0 CS grass/stairs-as-road + paved-rich, 5 per group
        def frac(i, cls): g = data[i][1]; return ((g == cls) & (preds['SegF-B0 CS'][i] == 1)).sum() / max((g == cls).sum(), 1) * ((g == cls).mean() > .05)
        groups = {'grass': sorted(range(len(data)), key=lambda i: -frac(i, 2))[:5]}
        if ds == 'ADE20K': groups['stairs'] = sorted(range(len(data)), key=lambda i: -(data[i][1] == 3).mean())[:5]
        else: groups['paved'] = sorted(range(len(data)), key=lambda i: -(data[i][1] == 1).mean())[::40][:5]
        for gname, idx in groups.items():
            rows = []
            for i in idx:
                im, g = data[i]; t = [C.overlay(im, np.zeros(g.shape, np.uint8), os.path.basename(fs[i])[:18]), C.overlay(im, g, 'ground truth')]
                for n in ['PIDNet-M (deployed 1x)', 'SegF-B0 CS', 'SegF-B2 ADE']:
                    t.append(C.overlay(im, preds[n][i], n.split(' (')[0]))
                r = np.hstack([cv2.resize(x, (480, int(480 * x.shape[0] / x.shape[1]))) for x in t])
                rows.append(cv2.resize(r, (2400, 360)))
            cv2.imwrite(f'{OUT}/samples/{ds}_{gname}.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    res['legend'] = 'samples: green paved, yellow grass/soil, magenta stairs, grey unsure(<0.7)'
    json.dump(res, open(OUT + '/metrics.json', 'w'), indent=1)
if __name__ == '__main__':
    main()
