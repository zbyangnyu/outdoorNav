"""Build taskE training data (4 classes: 0 other, 1 paved, 2 grass/soil, 3 stairs, 255 ignore).
ADE20K: only ADEChallengeData2016 *training* split (taskD tests on ADE validation -> no overlap).
RUGD: frames from the same videos as taskD test but >= GAP frames away from any taskD test frame;
     sequences trail-9 and creek are held out completely as a RUGD val set.
Writes SEG_DATA/{img,gt}/ + split.json"""
import glob, json, os, random, numpy as np, cv2
from pathlib import Path
from remotezip import RemoteZip
random.seed(0)
HERE = Path(__file__).resolve().parent
E = os.environ.get('SEG_DATA', str(HERE / 'data'))
D = os.environ.get('SEG_TASKD_DATA', str(HERE.parent / 'taskD' / 'data'))
ADE = os.environ.get('SEG_ADE', str(HERE / 'ADEChallengeData2016'))
RUGD_FRAMES = os.environ.get('SEG_RUGD_FRAMES', 'http://rugd.vision/data/RUGD_frames-with-annotations.zip')
if not Path(ADE, 'annotations', 'training').is_dir():
    raise FileNotFoundError('Set SEG_ADE to the extracted ADEChallengeData2016 directory')
if not Path(D, 'RUGD_annotations').is_dir():
    raise FileNotFoundError('Set SEG_TASKD_DATA to the directory containing RUGD_annotations')
os.makedirs(E + '/img', exist_ok=True); os.makedirs(E + '/gt', exist_ok=True)
# ---- ADE (annotation value = class id + 1, 0 = unlabeled); mapping identical to taskD prep_ade.py
LUT = np.zeros(256, np.uint8); LUT[0] = 255
for k in (6, 11, 52, 54): LUT[k + 1] = 1             # road sidewalk path runway
for k in (9, 13, 29, 46, 91, 94): LUT[k + 1] = 2     # grass earth field sand dirt-track land
for k in (53, 59, 121): LUT[k + 1] = 3               # stairs stairway step
LUT[3 + 1] = 255                                     # indoor floor: ignore (neither road nor obstacle)
items = []
for f in sorted(glob.glob(ADE + '/annotations/training/*.png')):
    a = cv2.imread(f, 0); g = LUT[a]; n = os.path.basename(f)[:-4]
    st, sf, pv, sky = (g == 3).mean(), (g == 2).mean(), (g == 1).mean(), (a == 3).mean()
    if st >= .01: kind = 'stairs'
    elif sky < .01: continue                        # no sky -> indoor-ish, skip
    elif sf >= .05 and pv >= .02: kind = 'mixed'
    elif sf >= .15: kind = 'soft'
    elif pv >= .15: kind = 'street'
    else: continue
    items.append((n, kind))
by = {}
for n, k in items: by.setdefault(k, []).append(n)
print({k: len(v) for k, v in by.items()})
cap = {'stairs': 10 ** 9, 'mixed': 10 ** 9, 'soft': 1500, 'street': 1500}
split = {'train': [], 'val': []}
for k, v in by.items():
    random.shuffle(v); v = v[:cap[k]]; nv = max(20, len(v) // 20)
    for i, n in enumerate(v):
        split['val' if i < nv else 'train'].append({'n': n, 'src': 'ade', 'kind': k})
        if os.path.exists(f'{E}/gt/{n}.png'): continue
        im = cv2.imread(f'{ADE}/images/training/{n}.jpg'); g = LUT[cv2.imread(f'{ADE}/annotations/training/{n}.png', 0)]
        cv2.imwrite(f'{E}/img/{n}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(f'{E}/gt/{n}.png', g)
# ---- RUGD
COL = {(64, 64, 64): 1, (101, 101, 11): 1, (108, 64, 20): 2, (255, 229, 204): 2, (0, 102, 0): 2,
       (255, 128, 0): 2, (153, 76, 0): 2, (0, 0, 0): 255}
test = {}
for f in glob.glob(D + '/rugd/img/*.png'):
    s, i = os.path.basename(f)[:-4].rsplit('_', 1); test.setdefault(s, []).append(int(i))
GAP, VALSEQ = 60, {'trail-9', 'creek'}
cand = []
for f in sorted(glob.glob(D + '/RUGD_annotations/*/*.png'))[::3]:
    s, i = os.path.basename(f)[:-4].rsplit('_', 1); i = int(i)
    if s not in VALSEQ and any(abs(i - t) < GAP for t in test.get(s, [])): continue
    cand.append((f, s))
random.shuffle(cand)
tr = [c for c in cand if c[1] not in VALSEQ][:1400]; va = [c for c in cand if c[1] in VALSEQ][:120]
import threading
from concurrent.futures import ThreadPoolExecutor
tl = threading.local()
def fetch(job):
    part, f, s = job; b = os.path.basename(f); n = 'rugd_' + b[:-4]
    if not os.path.exists(f'{E}/gt/{n}.png'):
        if not hasattr(tl, 'z'): tl.z = RemoteZip(RUGD_FRAMES)
        names = {os.path.basename(x): x for x in tl.z.namelist() if x.endswith('.png')}
        for k in range(5):
            try: data = tl.z.read(names[b]); break
            except Exception: tl.z = RemoteZip(RUGD_FRAMES)
        else: raise RuntimeError(f'Could not download RUGD frame {b}')
        im = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        a = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); g = np.zeros(a.shape[:2], np.uint8)
        for c, v in COL.items(): g[np.all(a == c, -1)] = v
        cv2.imwrite(f'{E}/img/{n}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(f'{E}/gt/{n}.png', g)
    return part, {'n': n, 'src': 'rugd', 'kind': s}
jobs = [('train', f, s) for f, s in tr] + [('val', f, s) for f, s in va]
with ThreadPoolExecutor(16) as ex:
    for part, it in ex.map(fetch, jobs): split[part].append(it)
split['rules'] = {'ade': 'ADEChallengeData2016 training only; taskD uses ADE validation',
                  'rugd': f'frames >= {GAP} frame ids from any taskD test frame; seqs {sorted(VALSEQ)} = val only',
                  'lynx': 'the 100 Lynx frames are never used for training'}
json.dump(split, open(E + '/split.json', 'w'), indent=1)
print('train', len(split['train']), 'val', len(split['val']))
