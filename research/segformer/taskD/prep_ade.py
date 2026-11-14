"""ADE20K full-index validation (HF mirror 1aurent/ADE20K, validation parquet, 2000 imgs) -> 4-class GT.
GT png: 1 paved, 2 grass/soil, 3 stairs, 0 other, 255 unlabeled. Keep outdoor frames containing
stairs or grass/soil/path (>=2% of pixels), max 300."""
import io, os, json, numpy as np, pyarrow.parquet as pq
from PIL import Image
PAVED = {'road', 'route', 'sidewalk', 'pavement', 'path', 'footpath', 'walkway', 'street', 'driveway', 'parking lot', 'runway', 'crosswalk'}
SOFT = {'grass', 'earth', 'ground', 'field', 'sand', 'dirt track', 'land', 'soil', 'lawn', 'mud', 'gravel'}
STAIRS = {'stairs', 'steps', 'step', 'stair', 'stairway', 'staircase', 'stairs, steps'}
def cat(name):
    toks = {t.strip() for t in name.split(',')}
    if toks & STAIRS: return 3
    if toks & PAVED: return 1
    if toks & SOFT: return 2
    return 0
os.makedirs('ade/img', exist_ok=True); os.makedirs('ade/gt', exist_ok=True)
t = pq.ParquetFile('ade_val.parquet'); kept = []; names = {}
for batch in t.iter_batches(batch_size=50):
    for r in batch.to_pylist():
        if 'outdoor' not in r['scene']: continue
        s = np.array(Image.open(io.BytesIO(r['segmentations'][0]['bytes']))).astype(np.int32)
        c = s[..., 0] // 10 * 256 + s[..., 1]
        lut = {o['name_ndx']: cat(o['name']) for o in r['objects'] if o['parts']['part_level'] == 0}
        for o in r['objects']:
            if o['parts']['part_level'] == 0 and cat(o['name']): names[o['name']] = cat(o['name'])
        g = np.zeros(c.shape, np.uint8)
        for k, v in lut.items(): g[c == k] = v
        g[c == 0] = 255
        st, sf, pv = [(g == i).mean() for i in (3, 2, 1)]
        if st >= .02 or (sf >= .02 and pv >= .02) or sf >= .1:
            kept.append({'f': r['filename'], 'stairs': float(st), 'soft': float(sf), 'paved': float(pv)})
            im = Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB')
            im.save('ade/img/' + r['filename'][:-4] + '.png'); Image.fromarray(g).save('ade/gt/' + r['filename'][:-4] + '.png')
# cap: all stairs frames first, then the rest up to 300
st = [k for k in kept if k['stairs'] >= .02]; rest = [k for k in kept if k['stairs'] < .02]
keep = {k['f'][:-4] for k in (st + rest)[:300]}
for k in kept:
    if k['f'][:-4] not in keep:
        os.remove('ade/img/' + k['f'][:-4] + '.png'); os.remove('ade/gt/' + k['f'][:-4] + '.png')
json.dump({'n': len(keep), 'n_stairs': len(st), 'mapped_names': names}, open('ade/info.json', 'w'), indent=1)
print(len(kept), 'candidates;', len(keep), 'kept;', len(st), 'with stairs')
