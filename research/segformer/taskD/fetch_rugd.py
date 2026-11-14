"""Pick ~300 RUGD frames from the (small) annotation zip, then pull only those images out of the
5.6GB frames zip with HTTP range requests (remotezip). Data: http://rugd.vision (RUGD, Wigness et al. 2019).
usage: python fetch_rugd.py   (run in taskD/data, after unzip RUGD_annotations.zip)"""
import glob, os, random, numpy as np, cv2
from remotezip import RemoteZip
random.seed(0)
COL = {10: (64, 64, 64), 23: (101, 101, 11), 3: (0, 102, 0), 1: (108, 64, 20), 13: (153, 76, 0), 11: (255, 128, 0), 2: (255, 229, 204)}
paved, soft = [], []
fs = sorted(glob.glob('RUGD_annotations/*/*.png'))
for f in fs[::3]:  # every 3rd annotated frame is plenty (sequences are video)
    a = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    fr = {k: np.all(a == c, -1).mean() for k, c in COL.items()}
    pv = fr[10] + fr[23]; sf = fr[3] + fr[1] + fr[13] + fr[11] + fr[2]
    if pv > .05: paved.append(f)
    elif sf > .3: soft.append(f)
print('candidates paved', len(paved), 'soft', len(soft))
pick = random.sample(paved, min(200, len(paved))) + random.sample(soft, min(100, len(soft)))
os.makedirs('rugd/img', exist_ok=True); os.makedirs('rugd/ann', exist_ok=True)
with RemoteZip('http://rugd.vision/data/RUGD_frames-with-annotations.zip') as z:
    names = {os.path.basename(n): n for n in z.namelist() if n.endswith('.png')}
    for f in pick:
        b = os.path.basename(f)
        open('rugd/img/' + b, 'wb').write(z.read(names[b]))
        os.replace(f, 'rugd/ann/' + b) if False else cv2.imwrite('rugd/ann/' + b, cv2.imread(f))
print('fetched', len(pick))
