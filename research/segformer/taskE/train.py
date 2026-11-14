"""Fine-tune SegFormer-B0 Cityscapes -> 4 classes (0 other, 1 paved, 2 grass/soil, 3 stairs).
env: ITERS (default 20000), OUT. Picks best checkpoint on val mean IoU(paved, grass, stairs)
minus stairs/grass->road confusion. Logs to OUT/log.jsonl."""
import json, os, time, numpy as np, cv2, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import SegformerForSemanticSegmentation
E = str(Path(__file__).resolve().parent); DATA = os.environ.get('SEG_DATA', E + '/data')
OUT = os.environ.get('OUT', E + '/run'); ITERS = int(os.environ.get('ITERS', 20000)); BS = 16
H, W = 480, 640
MEAN = np.array([.485, .456, .406], np.float32); STD = np.array([.229, .224, .225], np.float32)
CS = os.environ.get('SEG_PRETRAINED', 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024')

# ---------- photometric augmentation (random strengths; similar families to taskD but random params)
def shadow_leaves(im, g, r):
    """dappled tree shadow over the whole image + fallen leaves scattered on paved pixels (label kept)."""
    h, w = im.shape[:2]; x = im.astype(np.float32)
    if r.random() < .5:
        n = r.random((h // 16 + 1, w // 16 + 1)).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC); n = cv2.GaussianBlur(n, (0, 0), r.uniform(2, 8))
        thr = r.uniform(.35, .65); m = (n > thr).astype(np.float32); m = cv2.GaussianBlur(m, (0, 0), r.uniform(1, 4))
        top = int(r.uniform(0, .5) * h); m[:top] *= np.linspace(0, 1, top)[:, None] if top else 1
        x = x * (1 - m[..., None] * r.uniform(.35, .7))
    if r.random() < .5 and (g == 1).mean() > .03:
        L = np.zeros((h, w), np.uint8); cols = []
        for _ in range(int(r.uniform(20, 300))):
            cy, cx = r.integers(0, h), r.integers(0, w)
            if g[cy, cx] != 1: continue
            s = 1.5 + 12 * cy / h * r.uniform(.5, 1.5); v = r.uniform(60, 200)
            c = [v, v * r.uniform(.5, .85), v * r.uniform(.15, .45)]
            cv2.ellipse(L, (int(cx), int(cy)), (int(s), max(1, int(s * r.uniform(.3, .7)))), r.uniform(0, 180), 0, 360, 1, -1); cols.append(c)
        if cols:
            c = np.array(cols).mean(0) * r.uniform(.8, 1.2, 3); a = (L > 0)[..., None] * r.uniform(.5, .9)
            tex = x * .4 + np.array(c) * .6
            x = x * (1 - a) + tex * a
    return np.clip(x, 0, 255).astype(np.uint8)

def aug_photo(im, r):
    x = im.astype(np.float32)
    k = r.random()
    if k < .15:   # dark
        g = r.uniform(1.3, 3.5); x = (x / 255) ** g / g * 255 + r.normal(0, r.uniform(2, 8), x.shape)
    elif k < .27:  # overexposure + glare
        s = r.uniform(1.2, 2.4); h, w = x.shape[:2]; yy, xx = np.mgrid[:h, :w]
        cx, cy = r.uniform(.2, .8) * w, r.uniform(0, .4) * h
        x = x * s + (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (w * r.uniform(.15, .4)) ** 2)) * 255 * (s - 1))[..., None]
    elif k < .35:  # fog
        t = r.uniform(.2, 1.0); d = np.linspace(1, .3, x.shape[0])[:, None, None]; a = 1 - np.exp(-t * d * 3)
        x = x * (1 - a) + r.uniform(180, 240) * a
    elif k < .45:  # rain
        h, w = x.shape[:2]; L = np.zeros((h, w), np.float32); sl = r.uniform(-.4, .4)
        for _ in range(int(r.uniform(100, 1500))):
            px, py, l = r.integers(0, w), r.integers(0, h), r.integers(8, 30); cv2.line(L, (px, py), (int(px + l * sl), py + l), 1, 1)
        x = x * r.uniform(.7, .95) + cv2.GaussianBlur(L, (3, 3), 0)[..., None] * r.uniform(120, 220)
    x = np.clip(x, 0, 255)
    if r.random() < .5:  # colour jitter
        x = (x - 128) * r.uniform(.7, 1.3) + 128 + r.uniform(-30, 30); x = x * r.uniform(.85, 1.15, 3)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if r.random() < .25:  # motion / gaussian blur
        k = int(r.integers(3, 25)) | 1
        if r.random() < .6:
            ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1
            M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), r.uniform(-30, 30), 1); ker = cv2.warpAffine(ker, M, (k, k))
            x = cv2.filter2D(x, -1, ker / max(ker.sum(), 1e-6))
        else: x = cv2.GaussianBlur(x, (k, k), 0)
    if r.random() < .25: x = np.clip(x + r.normal(0, r.uniform(3, 40), x.shape), 0, 255).astype(np.uint8)
    if r.random() < .15:  # jpeg
        x = cv2.imdecode(cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, int(r.integers(20, 70))])[1], 1)
    return x

class DS(Dataset):
    def __init__(self, items, train, data=DATA, size=(H, W), with_edges=False):
        self.items, self.train = items, train
        self.data = Path(data)
        self.size = size
        self.with_edges = with_edges
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        item = self.items[i]
        n = item['n']
        H, W = self.size
        image = self.data / item.get('image', f'img/{n}.jpg')
        mask = self.data / item.get('mask', f'gt/{n}.png')
        im = cv2.cvtColor(cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        g = cv2.imdecode(np.fromfile(mask, np.uint8), cv2.IMREAD_GRAYSCALE)
        if self.with_edges:
            edge = np.zeros_like(g)
            if "edge" in item:
                path = self.data / item["edge"]
                edge = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
            g = np.stack((g, edge), axis=-1)
        if self.train:
            # PyTorch seeds NumPy in each data worker.
            r = np.random.default_rng(np.random.randint(0, 2**32, dtype=np.uint32))
            low = r.random() < .3   # low-viewpoint: zoom into the bottom part of the image
            s = (r.uniform(1.5, 2.8) if low else r.uniform(.6, 1.4)) * max(H / im.shape[0], W / im.shape[1])
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR); g = cv2.resize(g, im.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            if low: im, g = im[-int(max(H, im.shape[0] * .5)):], g[-int(max(H, im.shape[0] * .5)):]
            ph, pw = max(H - im.shape[0], 0), max(W - im.shape[1], 0)
            im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_CONSTANT, 0)
            g = cv2.copyMakeBorder(g, 0, ph, 0, pw, cv2.BORDER_CONSTANT, (255, 0) if self.with_edges else 255)
            y, x = r.integers(0, im.shape[0] - H + 1), r.integers(0, im.shape[1] - W + 1)
            im, g = im[y:y + H, x:x + W], g[y:y + H, x:x + W]
            if r.random() < .5: im, g = im[:, ::-1], g[:, ::-1]
            im = shadow_leaves(np.ascontiguousarray(im), g[..., 0] if self.with_edges else g, r)
            im = aug_photo(im, r)
        else:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR); g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
        x = (im.astype(np.float32) / 255 - MEAN) / STD
        if self.with_edges:
            return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                    torch.from_numpy(np.ascontiguousarray(g[..., 0], dtype=np.int64)),
                    torch.from_numpy(np.ascontiguousarray(g[..., 1] > 0)))
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(np.ascontiguousarray(g).astype(np.int64))

def build_model(pretrained=CS):
    m = SegformerForSemanticSegmentation.from_pretrained(pretrained)
    old = m.decode_head.classifier; new = torch.nn.Conv2d(old.in_channels, 4, 1)
    with torch.no_grad():  # init: paved=road, grass=terrain, stairs=sidewalk-ish (small), other=building
        new.weight[1] = old.weight[0]; new.bias[1] = old.bias[0]
        new.weight[2] = old.weight[9]; new.bias[2] = old.bias[9]
        new.weight[0] = old.weight[2]; new.bias[0] = old.bias[2]
        new.weight[3] = old.weight[1] * .5; new.bias[3] = old.bias[1] - 2
    m.decode_head.classifier = new
    m.config.num_labels = 4; m.config.id2label = {0: 'other', 1: 'paved', 2: 'grass_soil', 3: 'stairs'}
    m.config.label2id = {v: k for k, v in m.config.id2label.items()}
    return m

def evaluate(m, dl, dev):
    m.eval(); cm = torch.zeros(4, 4, dtype=torch.long, device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x, g in dl:
            x, g = x.to(dev), g.to(dev)
            p = F.interpolate(m(pixel_values=x).logits.float(), size=g.shape[1:], mode='bilinear').argmax(1)
            v = g != 255; cm += torch.bincount(g[v] * 4 + p[v], minlength=16).view(4, 4)
    cm = cm.double().cpu(); iou = (cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())).tolist()
    res = {'iou': iou, 'stairs_as_road': (cm[3, 1] / cm[3].sum()).item(), 'grass_as_road': (cm[2, 1] / cm[2].sum()).item(),
           'other_as_road': (cm[0, 1] / cm[0].sum()).item()}
    res['score'] = (iou[1] + iou[2] + iou[3]) / 3 - res['stairs_as_road'] - res['grass_as_road']
    m.train(); return res

def main():
    os.makedirs(OUT, exist_ok=True); dev = torch.device('cuda')
    sp = json.load(open(os.environ.get('SPLIT', DATA + '/split.json')))
    tr = sp['train']; va = sp['val']
    # sampling weight: stairs images x3, rugd x1, rest x1 (keep road-rich streets common to preserve paved skill)
    wt = [3. if t['kind'] == 'stairs' else .5 if t['src'] == 'rugd' else 1. for t in tr]
    dl = DataLoader(DS(tr, True), BS, sampler=WeightedRandomSampler(wt, ITERS * BS, replacement=True),
                    num_workers=14, drop_last=True, persistent_workers=True, prefetch_factor=4)
    vdl = DataLoader(DS(va, False), 16, num_workers=8)
    m = build_model().to(dev).train()
    bb = [p for n, p in m.named_parameters() if 'decode_head' not in n]; hd = [p for n, p in m.named_parameters() if 'decode_head' in n]
    opt = torch.optim.AdamW([{'params': bb, 'lr': 3e-5}, {'params': hd, 'lr': 6e-4}], weight_decay=.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 500) * (1 - i / ITERS) ** .9)
    cw = torch.tensor([1., 1., 1.2, 3.], device=dev)  # stairs weighted up (rare)
    KD = float(os.environ.get('KD', 0))  # distill paved prob from frozen original B0 CS on GT-ignore/paved/other pixels
    if KD: tch = SegformerForSemanticSegmentation.from_pretrained(CS).to(dev).eval().requires_grad_(False)
    ema = torch.optim.swa_utils.AveragedModel(m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.999)) if os.environ.get('EMA') else None
    best, t0, log = -9, time.time(), open(OUT + '/log.jsonl', 'a')
    for it, (x, g) in enumerate(dl):
        x, g = x.to(dev, non_blocking=True), g.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = m(pixel_values=x).logits
        y = F.interpolate(y.float(), size=g.shape[1:], mode='bilinear')
        loss = F.cross_entropy(y, g, weight=cw, ignore_index=255)
        if KD:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                t = F.interpolate(tch(pixel_values=x).logits.float(), size=g.shape[1:], mode='bilinear').softmax(1)
            tp = (t[:, 0] + t[:, 1]).clamp(1e-4, 1 - 1e-4); sp = y.softmax(1)[:, 1].clamp(1e-4, 1 - 1e-4)
            km = ((g == 255) | (g == 1) | (g == 0)).float()  # never on grass/stairs GT (teacher is wrong there)
            loss = loss + KD * (F.binary_cross_entropy(sp, tp, reduction='none') * km).sum() / km.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if ema is not None: ema.update_parameters(m)
        if it % 50 == 0: log.write(json.dumps({'it': it, 'loss': loss.item(), 't': time.time() - t0}) + '\n'); log.flush()
        if (it + 1) % int(os.environ.get('EVAL', 1000)) == 0 or it + 1 == ITERS:
            em = ema.module if ema is not None else m
            r = evaluate(em, vdl, dev); r['it'] = it + 1; log.write(json.dumps({'val': r}) + '\n'); log.flush()
            if os.environ.get('SAVE_ALL'): em.save_pretrained(f'{OUT}/it{it + 1}')
            print(it + 1, f'{time.time() - t0:.0f}s loss {loss.item():.3f}', {k: (np.round(v, 3).tolist()) for k, v in r.items()}, flush=True)
            if r['score'] > best:
                best = r['score']; em.save_pretrained(OUT + '/best'); json.dump(r, open(OUT + '/best/val.json', 'w'))
    print('TRAIN_DONE best', best)

if __name__ == '__main__':
    main()
