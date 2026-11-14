import glob, numpy as np, cv2, torch, compare as c
from viz_pad import run
m = c.PIDNet(1.0)
files = sorted(glob.glob(c.IMG_DIR + '/*.png'))
R = {t: {'cls': [], 'conf': []} for t in ('rep', 'blk')}
for f in files:
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    for t, b in [('rep', cv2.BORDER_REPLICATE), ('blk', cv2.BORDER_CONSTANT)]:
        _, cls, conf = run(m, rgb, b)
        R[t]['cls'].append(cls[:480]); R[t]['conf'].append(conf[:480])
for t in R: R[t] = {k: np.stack(v) for k, v in R[t].items()}
road = {t: R[t]['cls'] <= 1 for t in R}
front = (slice(None), slice(360, 480), slice(160, 480))
print('== front region (bottom 120 rows, middle half), mean over 100 frames')
for thr in [0, .5, .6, .7, .8]:
    line = []
    for t in ('rep', 'blk'):
        ok = road[t] & (R[t]['conf'] >= thr)
        fr = ok[front].reshape(100, -1).mean(1)
        line.append(f'{t}: road {fr.mean()*100:5.1f}%  bad(<50%) {int((fr<.5).sum()):3d}')
    print(f'conf>={thr:.1f}  ' + ' | '.join(line))
print('== argmax only (no threshold), by horizontal band of the image')
for a, b in [(240, 300), (300, 360), (360, 420), (420, 450), (450, 480)]:
    s = (slice(None), slice(a, b), slice(None))
    print(f'rows {a}-{b}: rep road {road["rep"][s].mean()*100:5.1f}%  blk road {road["blk"][s].mean()*100:5.1f}%  '
          f'mean conf rep {R["rep"]["conf"][s].mean():.2f} blk {R["blk"]["conf"][s].mean():.2f}')
agree = (R['rep']['cls'] == R['blk']['cls']).mean()
iou = (road['rep'] & road['blk']).sum() / (road['rep'] | road['blk']).sum()
print(f'== whole image: same class at {agree*100:.1f}% of pixels; road-mask IoU rep vs blk {iou:.3f}')
