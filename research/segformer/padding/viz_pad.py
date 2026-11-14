"""Dump PIDNet inputs/outputs for replicate vs black bottom padding (deployed size 640x512)."""
import glob, os, json, numpy as np, cv2, torch, compare as c

OUT = '/root/autodl-tmp/seg/out_pad'
CS = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
      'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def run(m, rgb, border):
    pad = cv2.copyMakeBorder(rgb, 0, 32, 0, 0, border, value=0)
    x = (pad.astype(np.float32) / 255 - c.MEAN) / c.STD
    x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(c.dev)
    with torch.inference_mode():
        y = torch.nn.functional.interpolate(m.m(x), size=pad.shape[:2], mode='bilinear', align_corners=True)
        conf, cls = y.softmax(1).max(1)
    return pad, cls[0].cpu().numpy().astype(np.uint8), conf[0].cpu().numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    m = c.PIDNet(1.0)
    files = sorted(glob.glob(c.IMG_DIR + '/*.png'))
    rows = []
    for i, f in enumerate(files):
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        r = {}
        for tag, b in [('rep', cv2.BORDER_REPLICATE), ('blk', cv2.BORDER_CONSTANT)]:
            pad, cls, conf = run(m, rgb, b)
            front_cls, front_conf = cls[360:480, 160:480], conf[360:480, 160:480]
            ok = (front_cls <= 1) & (front_conf >= c.THR)
            r[tag] = (pad, cls, conf, float(ok.mean()))
        rows.append((i, f, r))
    rows.sort(key=lambda t: t[2]['blk'][3] - t[2]['rep'][3], reverse=True)
    pick = rows[:4]
    summary = []
    for i, f, r in pick:
        info = {'frame': i}
        for tag in ('rep', 'blk'):
            pad, cls, conf, ok = r[tag]
            np.savez_compressed(f'{OUT}/f{i:03d}_{tag}.npz', pad=pad, cls=cls, conf=conf)
            fc, fconf = cls[360:480, 160:480], conf[360:480, 160:480]
            lab = np.where(fconf < c.THR, 255, fc)
            vals, cnt = np.unique(lab, return_counts=True)
            order = np.argsort(-cnt)[:4]
            info[tag] = {'front_ok': round(ok, 3),
                         'front_breakdown': {('unsure(<0.7)' if vals[k] == 255 else CS[vals[k]]): round(cnt[k] / lab.size, 3)
                                             for k in order},
                         'road_conf_mean_front': round(float(fconf[fc <= 1].mean()) if (fc <= 1).any() else 0, 3)}
        summary.append(info)
        print(json.dumps(info), flush=True)
    # whole-set breakdown of the front region under replicate padding
    agg = {}
    for i, f, r in rows:
        pad, cls, conf, ok = r['rep']
        lab = np.where(conf[360:480, 160:480] < c.THR, 255, cls[360:480, 160:480])
        for v, n in zip(*np.unique(lab, return_counts=True)):
            k = 'unsure(<0.7)' if v == 255 else CS[v]
            agg[k] = agg.get(k, 0) + int(n)
    tot = sum(agg.values())
    print('ALL_FRAMES_FRONT_REPLICATE', {k: round(v / tot, 3) for k, v in sorted(agg.items(), key=lambda t: -t[1])[:6]})
    json.dump(summary, open(OUT + '/summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
