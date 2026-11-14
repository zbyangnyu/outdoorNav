"""Fine-tune SegFormer-B0 (ADE20K init) to 4 classes: 0 other, 1 paved ground, 2 grass/soil, 3 stairs.

Data: ADE20K training (stair / road / random images, labels remapped) + first 60 Lynx frames
(pseudo-labelled by SegFormer-B1 Cityscapes). Lynx frames 60-99, ADE validation and the
web stair photos are kept out for testing.
"""
import glob, os, sys, time, numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, '/root/autodl-tmp/seg')
import compare as c

ADE = '/root/autodl-tmp/ade/ADEChallengeData2016'
OUT = '/root/autodl-tmp/seg/ft_b0'
LYNX_LBL = '/root/autodl-tmp/seg/lynx_pseudo'
N_LYNX_TRAIN = 60
ITERS, BATCH, CROP = int(os.environ.get('ITERS', 8000)), 16, 512
PAVED = [6, 11, 52, 54, 91, 3]       # road sidewalk path runway dirt-track floor
SOFT = [9, 13, 29, 46, 94]           # grass earth field sand land
STAIR = [53, 59, 121]
LUT = np.zeros(256, np.uint8)        # ADE raw value (1-based, 0 = unlabeled) -> 4 classes
LUT[0] = 255
for k in PAVED: LUT[k + 1] = 1
for k in SOFT: LUT[k + 1] = 2
for k in STAIR: LUT[k + 1] = 3
NAMES = ['other', 'paved', 'grass_soil', 'stairs']


def make_lynx_pseudo():
    os.makedirs(LYNX_LBL, exist_ok=True)
    files = sorted(glob.glob(c.IMG_DIR + '/*.png'))[:N_LYNX_TRAIN]
    m = c.SegF('b1-finetuned-cityscapes-1024-1024')
    for f in files:
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        cat = c.predict(m, rgb)              # 1 paved, 2 terrain, 0 other, 255 unsure
        cv2.imwrite(f'{LYNX_LBL}/{os.path.basename(f)}', cat)
    del m; torch.cuda.empty_cache()
    return [(f, f'{LYNX_LBL}/{os.path.basename(f)}', 'lynx') for f in files]


class Seg(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_p, lbl_p, kind = self.items[i]
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(lbl_p, cv2.IMREAD_UNCHANGED)
        if kind == 'ade':
            lbl = LUT[lbl]
        rng = np.random.default_rng()
        s = CROP / min(img.shape[:2]) * rng.uniform(.5, 2.0)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ph, pw = max(0, CROP - img.shape[0]), max(0, CROP - img.shape[1])
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(124, 116, 104))
            lbl = cv2.copyMakeBorder(lbl, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
        y = rng.integers(0, img.shape[0] - CROP + 1); x = rng.integers(0, img.shape[1] - CROP + 1)
        img, lbl = img[y:y + CROP, x:x + CROP], lbl[y:y + CROP, x:x + CROP]
        if rng.random() < .5:
            img, lbl = img[:, ::-1], lbl[:, ::-1]
        img = img.astype(np.float32)
        img = np.clip(img * rng.uniform(.75, 1.25) + rng.uniform(-25, 25), 0, 255)   # brightness/contrast
        img = (img / 255 - c.MEAN) / c.STD
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), torch.from_numpy(lbl.astype(np.int64).copy())


def build_model():
    from transformers import SegformerForSemanticSegmentation
    p = glob.glob(f'{c.HF}/models--nvidia--segformer-b0-finetuned-ade-512-512/snapshots/*')[0]
    base = SegformerForSemanticSegmentation.from_pretrained(p)
    m = SegformerForSemanticSegmentation.from_pretrained(
        p, num_labels=4, id2label=dict(enumerate(NAMES)), label2id={n: i for i, n in enumerate(NAMES)},
        ignore_mismatched_sizes=True)
    # warm-start the new 4-way classifier from the ADE rows it merges
    W, B = base.decode_head.classifier.weight.data, base.decode_head.classifier.bias.data
    groups = [[k for k in range(150) if k not in PAVED + SOFT + STAIR], PAVED, SOFT, STAIR]
    for j, g in enumerate(groups):
        m.decode_head.classifier.weight.data[j] = W[g].mean(0)
        m.decode_head.classifier.bias.data[j] = B[g].max()
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    fr = np.load('/root/autodl-tmp/ade/train_fracs.npy')
    anns = sorted(glob.glob(f'{ADE}/annotations/training/*.png'))
    def item(a): return (a.replace('annotations', 'images').replace('.png', '.jpg'), a, 'ade')
    stair = [item(a) for a, f in zip(anns, fr) if f[0] > .005]
    road = [item(a) for a, f in zip(anns, fr) if f[0] <= .005 and f[1] > .05]
    rest = [item(a) for a, f in zip(anns, fr) if f[0] <= .005 and f[1] <= .05]
    rest = [rest[i] for i in np.random.default_rng(0).choice(len(rest), 3000, replace=False)]
    lynx = make_lynx_pseudo()
    items = stair + road + rest + lynx
    share = {'stair': .35, 'road': .35, 'rest': .15, 'lynx': .15}
    w = ([share['stair'] / len(stair)] * len(stair) + [share['road'] / len(road)] * len(road)
         + [share['rest'] / len(rest)] * len(rest) + [share['lynx'] / len(lynx)] * len(lynx))
    print(f'train items: stair {len(stair)} road {len(road)} rest {len(rest)} lynx {len(lynx)}', flush=True)
    dl = DataLoader(Seg(items), batch_size=BATCH, num_workers=24, drop_last=True, persistent_workers=True,
                    sampler=WeightedRandomSampler(w, ITERS * BATCH, replacement=True))
    m = build_model().to(c.dev).train()
    head = [p for n, p in m.named_parameters() if n.startswith('decode_head')]
    body = [p for n, p in m.named_parameters() if not n.startswith('decode_head')]
    opt = torch.optim.AdamW([{'params': body, 'lr': 6e-5}, {'params': head, 'lr': 6e-4}], weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (1 - it / ITERS) ** 1.0)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1., 1., 2.], device=c.dev), ignore_index=255)
    t0, run = time.time(), 0.
    for it, (x, y) in enumerate(dl):
        x, y = x.to(c.dev, non_blocking=True), y.to(c.dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = m(pixel_values=x).logits
        lg = torch.nn.functional.interpolate(lg.float(), size=y.shape[1:], mode='bilinear', align_corners=False)
        loss = lossf(lg, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = .98 * run + .02 * loss.item() if it else loss.item()
        if it % 200 == 0 or it == ITERS - 1:
            print(f'iter {it}/{ITERS} loss {run:.3f} {time.time() - t0:.0f}s', flush=True)
    m.save_pretrained(OUT)
    print('TRAIN_DONE', flush=True)


if __name__ == '__main__':
    main()
