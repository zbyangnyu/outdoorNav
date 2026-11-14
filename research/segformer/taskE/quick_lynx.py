"""Quick check: Lynx clean road IoU/fp/fn + mean class map for a checkpoint. usage: python quick_lynx.py CKPT [CKPT...]"""
import glob, os, sys, numpy as np, cv2
S = '/root/autodl-tmp/seg'; sys.path.insert(0, S); sys.path.insert(0, S + '/taskB'); sys.path.insert(0, S + '/taskE')
import compare as C
C.THR = float(os.environ.get('THR', C.THR))
from evaluate import counts, metrics
from eval_all import FT
fs = sorted(glob.glob(S + '/camera_gemini330__color__image_raw/*.png'))
for ck in sys.argv[1:]:
    m = FT(ck); tot = 0; frac = np.zeros(256)
    for f in fs:
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); r = cv2.imread(S + '/taskB/ref/' + os.path.basename(f), 0)
        p = C.predict(m, im); tot = tot + counts(p, r); frac += np.bincount(p[r == 1], minlength=256)
    frac /= frac.sum()
    print(ck, {k: round(float(v), 3) for k, v in metrics(tot).items()}, 'on-road pred: paved %.2f grass %.2f stairs %.2f other %.2f unsure %.2f' % (frac[1], frac[2], frac[3], frac[0], frac[255]), flush=True)
