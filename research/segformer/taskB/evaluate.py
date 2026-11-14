"""Score road segmentation models against reference masks (from teacher.py or real labels).

Ref PNG: 1 road, 0 non-road, 255 ignore. Same file stem as the image.
usage: python evaluate.py --images DIR --refs DIR --out DIR [--models "PIDNet-M,SegF-B0 CS,..."] [--worst 8]
Metrics (model "unsure" pixels, conf<0.7, count as NOT road; also reported separately):
  iou   road IoU
  fp    false-road rate = non-road pixels the model calls road   (dangerous for navigation)
  fn    miss rate       = road pixels the model does not call road
  unsure_on_road / unsure_on_nonroad  share of pixels with conf<0.7
Also writes front-region metrics (bottom 120 rows, middle half) = ground right in front of the robot.
"""
import argparse, glob, json, os, sys
import numpy as np, cv2, torch

sys.path.insert(0, '/root/autodl-tmp/seg')
import compare as C  # model loaders + class mapping, unchanged

MODELS = {
    'PIDNet-M (deployed 1x)': lambda: C.PIDNet(1.0),
    'PIDNet-M 2x': lambda: C.PIDNet(2.0),
    'SegF-B0 CS 1x': lambda: C.SegF('b0-finetuned-cityscapes-1024-1024'),
    'SegF-B0 CS 0.5x': lambda: C.SegF('b0-finetuned-cityscapes-1024-1024', .5),
    'SegF-B1 CS 1x': lambda: C.SegF('b1-finetuned-cityscapes-1024-1024'),
    'SegF-B1 CS 0.5x': lambda: C.SegF('b1-finetuned-cityscapes-1024-1024', .5),
    'SegF-B0 ADE': lambda: C.SegF('b0-finetuned-ade-512-512'),
    'SegF-B1 ADE': lambda: C.SegF('b1-finetuned-ade-512-512'),
    'SegF-B2 ADE': lambda: C.SegF('b2-finetuned-ade-512-512'),
}


def counts(pred, ref, sl=np.s_[:, :]):
    p, r = pred[sl], ref[sl]
    road, non = r == 1, r == 0
    pr = p == 1
    return np.array([(pr & road).sum(), (pr & non).sum(), (~pr & road).sum(), road.sum(), non.sum(),
                     ((p == 255) & road).sum(), ((p == 255) & non).sum()], np.int64)


def metrics(c):
    tp, fp, fn, nr, nn, ur, un = c
    return {'iou': tp / max(tp + fp + fn, 1), 'fp': fp / max(nn, 1), 'fn': fn / max(nr, 1),
            'unsure_on_road': ur / max(nr, 1), 'unsure_on_nonroad': un / max(nn, 1)}


def ref_vis(rgb, ref):
    cat = np.where(ref == 1, 1, np.where(ref == 255, 255, 0)).astype(np.uint8)
    return C.overlay(rgb, cat, 'reference (teacher)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True); ap.add_argument('--refs', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--worst', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out + '/worst', exist_ok=True)
    files = sorted(f for f in glob.glob(a.images + '/*') if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    pairs = [(f, f'{a.refs}/{os.path.splitext(os.path.basename(f))[0]}.png') for f in files]
    pairs = [p for p in pairs if os.path.isfile(p[1])]
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f, _ in pairs]
    refs = [cv2.imread(r, cv2.IMREAD_GRAYSCALE) for _, r in pairs]
    names = [n.strip() for n in a.models.split(',')]
    res, preds = {}, {}
    for n in names:
        torch.cuda.empty_cache()
        m = MODELS[n]()
        tot, front, per = 0, 0, []
        ps = []
        for im, r in zip(imgs, refs):
            p = C.predict(m, im)
            ps.append(p)
            h, w = r.shape
            c = counts(p, r); tot = tot + c
            front = front + counts(p, r, np.s_[-120:, w // 4: 3 * w // 4])
            per.append(metrics(c)['fp'])
        preds[n] = ps
        res[n] = {'all': metrics(tot), 'front': metrics(front), 'per_image_fp': per}
        print(n, {k: round(float(v), 4) for k, v in res[n]['all'].items()}, flush=True)
        del m
    # worst images = highest mean false-road rate over all models
    mean_fp = np.mean([res[n]['per_image_fp'] for n in names], 0)
    worst = np.argsort(-mean_fp)[:a.worst]
    for i in worst:
        tiles = [C.overlay(imgs[i], np.zeros(refs[i].shape, np.uint8), 'original'), ref_vis(imgs[i], refs[i])]
        for n in names:
            p = preds[n][i].copy(); p[(p != 1) & (p != 255)] = 0
            v = C.overlay(imgs[i], p, f"{n} fp={res[n]['per_image_fp'][i]:.2f}")
            v[(preds[n][i] == 1) & (refs[i] == 0)] = (0, 0, 255)  # red = wrongly called road
            tiles.append(v)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        g = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
        g = cv2.resize(g, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{a.out}/worst/{os.path.basename(pairs[i][0])[:-4]}.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 85])
    js = {'frames': len(pairs), 'note': 'reference masks come from large models, not human labels',
          'models': {n: {k: (v if k == 'per_image_fp' else {q: float(x) for q, x in v.items()})
                         for k, v in res[n].items()} for n in names},
          'worst_frames': [os.path.basename(pairs[i][0]) for i in worst]}
    for n in names:
        js['models'][n]['per_image_fp'] = [float(x) for x in res[n]['per_image_fp']]
    json.dump(js, open(a.out + '/metrics.json', 'w'), indent=1)
    L = ['| model | road IoU | false-road (fp) | miss (fn) | unsure on road | unsure on non-road | front fp | front fn |',
         '|---|---|---|---|---|---|---|---|']
    for n in names:
        x, f = res[n]['all'], res[n]['front']
        L.append(f"| {n} | {x['iou']:.3f} | {x['fp']:.3f} | {x['fn']:.3f} | {x['unsure_on_road']:.3f} | "
                 f"{x['unsure_on_nonroad']:.3f} | {f['fp']:.3f} | {f['fn']:.3f} |")
    open(a.out + '/table.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
