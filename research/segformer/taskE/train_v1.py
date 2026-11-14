"""Fine-tune SegFormer-B0 Cityscapes -> 4 classes (0 other, 1 paved, 2 grass/soil, 3 stairs).
env: ITERS (default 20000), OUT. Picks best checkpoint on val mean IoU(paved, grass, stairs)
minus stairs/grass->road confusion. Logs to OUT/log.jsonl."""
import glob, json, os, sys, time, random, numpy as np, cv2, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import SegformerForSemanticSegmentation
E = '/root/autodl-tmp/seg/taskE'; DATA = E + '/data'
OUT = os.environ.get('OUT', E + '/run'); ITERS = int(os.environ.get('ITERS', 20000)); BS = 16
H, W = 480, 640
MEAN = np.array([.485, .456, .406], np.float32); STD = np.array([.229, .224, .225], np.float32)
CS = glob.glob('/root/autodl-tmp/hf/hub/models--nvidia--segformer-b0-finetuned-cityscapes-1024-1024/snapshots/*')[0]

# ---------- photometric augmentation (random strengths; similar families to taskD but random params)
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
    def __init__(self, items, train):
        self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        n = self.items[i]['n']; r = np.random.default_rng()
        im = cv2.cvtColor(cv2.imread(f'{DATA}/img/{n}.jpg'), cv2.COLOR_BGR2RGB); g = cv2.imread(f'{DATA}/gt/{n}.png', 0)
        if self.train:
            s = r.uniform(.6, 1.4) * max(H / im.shape[0], W / im.shape[1])
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR); g = cv2.resize(g, im.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            ph, pw = max(H - im.shape[0], 0), max(W - im.shape[1], 0)
            im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_CONSTANT, 0); g = cv2.copyMakeBorder(g, 0, ph, 0, pw, cv2.BORDER_CONSTANT, 255)
            y, x = r.integers(0, im.shape[0] - H + 1), r.integers(0, im.shape[1] - W + 1)
            im, g = im[y:y + H, x:x + W], g[y:y + H, x:x + W]
            if r.random() < .5: im, g = im[:, ::-1], g[:, ::-1]
            im = aug_photo(np.ascontiguousarray(im), r)
        else:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR); g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
        x = (im.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(np.ascontiguousarray(g).astype(np.int64))

def build_model():
    m = SegformerForSemanticSegmentation.from_pretrained(CS)
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
    wt = [3. if t['kind'] == 'stairs' else 1. for t in tr]
    dl = DataLoader(DS(tr, True), BS, sampler=WeightedRandomSampler(wt, ITERS * BS, replacement=True),
                    num_workers=14, drop_last=True, persistent_workers=True, prefetch_factor=4)
    vdl = DataLoader(DS(va, False), 16, num_workers=8)
    m = build_model().to(dev).train()
    bb = [p for n, p in m.named_parameters() if 'decode_head' not in n]; hd = [p for n, p in m.named_parameters() if 'decode_head' in n]
    opt = torch.optim.AdamW([{'params': bb, 'lr': 6e-5}, {'params': hd, 'lr': 6e-4}], weight_decay=.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 500) * (1 - i / ITERS) ** .9)
    cw = torch.tensor([1., 1., 1.2, 3.], device=dev)  # stairs weighted up (rare)
    best, t0, log = -9, time.time(), open(OUT + '/log.jsonl', 'a')
    for it, (x, g) in enumerate(dl):
        x, g = x.to(dev, non_blocking=True), g.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = m(pixel_values=x).logits
        y = F.interpolate(y.float(), size=g.shape[1:], mode='bilinear')
        loss = F.cross_entropy(y, g, weight=cw, ignore_index=255)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if it % 50 == 0: log.write(json.dumps({'it': it, 'loss': loss.item(), 't': time.time() - t0}) + '\n'); log.flush()
        if (it + 1) % int(os.environ.get('EVAL', 1000)) == 0 or it + 1 == ITERS:
            r = evaluate(m, vdl, dev); r['it'] = it + 1; log.write(json.dumps({'val': r}) + '\n'); log.flush()
            print(it + 1, f'{time.time() - t0:.0f}s loss {loss.item():.3f}', {k: (np.round(v, 3).tolist()) for k, v in r.items()}, flush=True)
            if r['score'] > best:
                best = r['score']; m.save_pretrained(OUT + '/best'); json.dump(r, open(OUT + '/best/val.json', 'w'))
    print('TRAIN_DONE best', best)

if __name__ == '__main__':
    main()
