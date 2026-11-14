"""Per-source val metrics for every saved checkpoint (run_*/it*), validation split only (no test data).
Selection rule (fixed before looking at test): among checkpoints whose val stairs->road and grass->road are both
<= 5%, take the one with the highest ITLP-val paved IoU (the robot-viewpoint campus data closest to the Lynx camera).
usage: python select_ckpt.py run_v5 run_v6 -> select.json"""
import glob, json, sys, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import train as T
dev = torch.device('cuda'); va = json.load(open(T.DATA + '/split.json'))['val']
srcs = sorted({d['src'] for d in va}); dls = {s: DataLoader(T.DS([d for d in va if d['src'] == s], False), 16, num_workers=8) for s in srcs}
def cm_of(m, dl):
    cm = torch.zeros(4, 4, dtype=torch.long, device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x, g in dl:
            x, g = x.to(dev), g.to(dev); p = F.interpolate(m(pixel_values=x).logits.float(), size=g.shape[1:], mode='bilinear').argmax(1)
            v = g != 255; cm += torch.bincount(g[v] * 4 + p[v], minlength=16).view(4, 4)
    return cm.double().cpu()
def mets(cm):
    iou = (cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())).tolist()
    return {'iou': [round(v, 4) for v in iou], 'stairs_as_road': (cm[3, 1] / cm[3].sum()).item() if cm[3].sum() else None,
            'grass_as_road': (cm[2, 1] / cm[2].sum()).item() if cm[2].sum() else None}
res = {}
for run in sys.argv[1:]:
    for ck in sorted(glob.glob(run + '/it*'), key=lambda p: int(p.split('it')[-1])):
        m = SegformerForSemanticSegmentation.from_pretrained(ck).to(dev).eval(); cms = {s: cm_of(m, dl) for s, dl in dls.items()}
        r = {s: mets(c) for s, c in cms.items()}; r['all'] = mets(sum(cms.values())); res[ck] = r
        print(ck, 'ITLP paved', r['itlp']['iou'][1], 'all stairs->road %.3f grass->road %.3f' % (r['all']['stairs_as_road'], r['all']['grass_as_road']), flush=True)
ok = [k for k, r in res.items() if r['all']['stairs_as_road'] <= .05 and r['all']['grass_as_road'] <= .05]
pick = max(ok, key=lambda k: res[k]['itlp']['iou'][1])
json.dump({'rule': __doc__, 'pick': pick, 'per_ckpt': res}, open('select.json', 'w'), indent=1); print('PICK', pick)
