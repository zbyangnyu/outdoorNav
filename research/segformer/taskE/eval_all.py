"""Side-by-side eval: original B0 CS vs taskE fine-tuned B0 on
(1) 100 Lynx frames vs taskB refs (+ front region), (2) taskD perturbations, (3) taskD hard scenes.
usage: python eval_all.py [ckpt_dir]  -> /root/autodl-tmp/seg/taskE/eval/metrics.json + samples/"""
import glob, json, os, sys, numpy as np, cv2, torch
S = '/root/autodl-tmp/seg'
for p in (S, S + '/taskB', S + '/taskD'): sys.path.insert(0, p)
import compare as C
C.THR = float(os.environ.get('THR', C.THR))
from evaluate import counts, metrics
import perturb_eval as PE, hard_eval as HE
CKS = sys.argv[1:] or [S + '/taskE/run_v5/best']; OUT = os.environ.get('OUT', S + '/taskE/eval')

class FT(C.SegF):
    def __init__(self, path):
        from transformers import SegformerForSemanticSegmentation
        self.m = SegformerForSemanticSegmentation.from_pretrained(path).to(C.dev).eval()
        self.scale, self.map = 1.0, {1: 1, 2: 2, 3: 3}
MODELS = {'B0 CS (original)': lambda: C.SegF('b0-finetuned-cityscapes-1024-1024')}
for ck in CKS: MODELS['B0 taskE ' + ck.rstrip('/').split('/')[-2] + '/' + ck.rstrip('/').split('/')[-1]] = (lambda c: lambda: FT(c))(ck)
FRONT = np.s_[-120:, 160:480]

def main():
    os.makedirs(OUT + '/samples', exist_ok=True); res = {'ckpts': CKS, 'thr': C.THR}
    files = sorted(glob.glob(S + '/camera_gemini330__color__image_raw/*.png'))
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    refs = [cv2.imread(S + '/taskB/ref/' + os.path.basename(f), 0) for f in files]
    cond = {f'{c}:{s}': (imgs if fn is None else [fn(im.astype(np.float32), s) if c != 'motion_blur' else fn(im, s) for im in imgs])
            for c, fn, s in PE.CONDS}
    hard = {}
    for ds in ['RUGD', 'ADE20K']:
        fs, gs = HE.load(ds); hard[ds] = (fs, [HE.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), g) for f, g in zip(fs, gs)])
    preds = {}
    for n, mk in MODELS.items():
        torch.cuda.empty_cache(); m = mk(); r = res.setdefault(n, {'lynx': {}, 'perturb': {}, 'hard': {}})
        for k, ims in cond.items():
            tot, fr = 0, 0
            for i, (im, ref) in enumerate(zip(ims, refs)):
                p = C.predict(m, im); tot = tot + counts(p, ref); fr = fr + counts(p, ref, FRONT)
                if k.startswith('clean'): preds[(n, 'lynx', i)] = p
            r['perturb'][k] = {q: float(v) for q, v in metrics(tot).items()}
            if k.startswith('clean'):
                r['lynx'] = dict(r['perturb'][k]); r['lynx'].update({'front_' + q: float(v) for q, v in metrics(fr).items()})
        for ds, (fs, data) in hard.items():
            tot = 0
            for i, (im, g) in enumerate(data):
                p = C.predict(m, im); tot = tot + HE.acc(p, g); preds[(n, ds, i)] = p
            r['hard'][ds] = HE.mets(tot)
        print(n, json.dumps({'lynx': r['lynx'], 'hard': r['hard']}), flush=True)
        print(n, {k: round(v['iou'], 3) for k, v in r['perturb'].items()}, flush=True); del m
    json.dump(res, open(OUT + '/metrics.json', 'w'), indent=1)
    # samples: rows = image | reference/GT | original | fine-tuned
    names = list(MODELS)
    def grid(rows_spec, fname):
        rows = []
        for im, ref, key, title in rows_spec:
            t = [C.overlay(im, np.zeros(ref.shape, np.uint8), title), C.overlay(im, ref, 'reference')]
            for n in names: t.append(C.overlay(im, preds[(n,) + key], n))
            rows.append(np.hstack([cv2.resize(x, (480, 360)) for x in t]))
        cv2.imwrite(f'{OUT}/samples/{fname}.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    # lynx: 4 frames where the two models differ most
    diff = sorted(range(len(imgs)), key=lambda i: -(preds[(names[0], 'lynx', i)] != preds[(names[1], 'lynx', i)]).mean())
    grid([(imgs[i], np.where(refs[i] == 255, 255, refs[i]).astype(np.uint8), ('lynx', i), os.path.basename(files[i])[:18]) for i in diff[:4]], 'lynx')
    fs, data = hard['ADE20K']
    st = sorted(range(len(data)), key=lambda i: -(data[i][1] == 3).mean())[:5]
    grid([(data[i][0], data[i][1], ('ADE20K', i), os.path.basename(fs[i])[:18]) for i in st], 'ade_stairs')
    def worst(ds, cls):
        fs, data = hard[ds]; n0 = names[0]
        return sorted(range(len(data)), key=lambda i: -((data[i][1] == cls) & (preds[(n0, ds, i)] == 1)).mean())[:4]
    grid([(hard['ADE20K'][1][i][0], hard['ADE20K'][1][i][1], ('ADE20K', i), 'ADE grass') for i in worst('ADE20K', 2)], 'ade_grass')
    grid([(hard['RUGD'][1][i][0], hard['RUGD'][1][i][1], ('RUGD', i), 'RUGD grass') for i in worst('RUGD', 2)], 'rugd_grass')
    print('EVAL_DONE')
if __name__ == '__main__':
    main()
