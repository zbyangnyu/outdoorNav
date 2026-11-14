"""Add ITLP-Campus-Outdoor (CC-BY-4.0, HF OPR-Project/ITLP-Campus-Outdoor): mobile-robot front camera on a
university campus; masks are OneFormer (Mapillary Vistas 65 classes) pseudo-labels shipped with the dataset.
Mapped to 4 classes; robot arm (labelled bicycle/car-mount/ego) and snow -> ignore.
Per track, the last 10% of frames (by time, contiguous) go to val. Appends to data/split.json."""
import json, os, cv2, numpy as np, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
E = os.environ.get('SEG_DATA', str(Path(__file__).resolve().parent / 'data'))
HUB = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
B = HUB + '/datasets/OPR-Project/ITLP-Campus-Outdoor/resolve/main/'
API = HUB + '/api/datasets/OPR-Project/ITLP-Campus-Outdoor/tree/main/'
TRACKS = {'05_2023-08-15-day': 2, '07_2023-10-04-day': 2, '06_2023-08-18-night': 4, '08_2023-10-11-night': 4}  # every k-th frame
LUT = np.zeros(256, np.uint8)
for k in (7, 8, 9, 10, 11, 13, 14, 15, 23, 24, 36, 41, 43): LUT[k] = 1   # bike lane, crosswalk, curb cut, parking, ped. area, road, service lane, sidewalk, markings, catch basin, manhole, pothole
for k in (26, 29): LUT[k] = 2                                           # sand, terrain
for k in (28, 52, 63, 64, 65): LUT[k] = 255                             # snow, bicycle(=robot arm), car mount, ego vehicle, unlabeled
def get(url):
    for _ in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.ok: return r.content
        except Exception: pass
    raise RuntimeError(url)
def job(a):
    t, p, part = a; n = 'itlp_' + t[:2] + '_' + os.path.basename(p)[:-4]
    if not os.path.exists(f'{E}/gt/{n}.png'):
        im = cv2.imdecode(np.frombuffer(get(B + p), np.uint8), 1)
        m = cv2.imdecode(np.frombuffer(get(B + p.replace('front_cam', 'masks/front_cam')), np.uint8), -1)
        g = LUT[m]; h = g.shape[0]; arm = np.isin(m, (52, 63, 64)); arm[:int(h * .5)] = False; g[arm] = 255
        im = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA); g = cv2.resize(g, (960, 540), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{E}/img/{n}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(f'{E}/gt/{n}.png', g)
    return part, {'n': n, 'src': 'itlp', 'kind': t}
jobs = []
for t, k in TRACKS.items():
    fs = sorted(x['path'] for x in json.loads(get(API + t + '/front_cam')) if x['path'].endswith('.png'))[::k]
    nv = len(fs) // 10; jobs += [(t, p, 'val' if i >= len(fs) - nv else 'train') for i, p in enumerate(fs)]
sp = json.load(open(E + '/split.json'))
for part in ('train', 'val'): sp[part] = [d for d in sp[part] if d['src'] != 'itlp']
with ThreadPoolExecutor(16) as ex:
    for part, it in ex.map(job, jobs): sp[part].append(it)
sp['rules']['itlp'] = 'ITLP-Campus-Outdoor tracks ' + ', '.join(TRACKS) + '; last 10% of each track = val; not in any test set'
json.dump(sp, open(E + '/split.json', 'w'), indent=1)
print('itlp', sum(j[2] == 'train' for j in jobs), 'train', sum(j[2] == 'val' for j in jobs), 'val')
