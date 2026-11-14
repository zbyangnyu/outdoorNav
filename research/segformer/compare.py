"""Compare PIDNet-M (as deployed) and SegFormer variants on Lynx camera frames.

Outputs (in OUT): grid/*.jpg side-by-side overlays, stats.json, summary.txt.
Categories: 1 paved (road/sidewalk/path), 2 grass/soil, 3 stairs/step,
0 other, 255 unknown (confidence < THR, same 0.7 as deployment).
"""
import glob, json, os, sys, time
import numpy as np, cv2, torch

ROOT = '/root/autodl-tmp/seg'
IMG_DIR = f'{ROOT}/camera_gemini330__color__image_raw'
OUT = f'{ROOT}/out'
PIDNET_W = f'{ROOT}/w/PIDNet_M_Cityscapes_test.pt'
HF = '/root/autodl-tmp/hf/hub'
THR = 0.7
dev = torch.device('cuda')
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)

# class id -> category
CS = {0: 1, 1: 1, 9: 2}  # road, sidewalk | terrain
ADE = {6: 1, 11: 1, 52: 1, 54: 1, 91: 1,      # road sidewalk path runway dirt-track
       9: 2, 13: 2, 29: 2, 46: 2, 94: 2,      # grass earth field sand land
       53: 3, 59: 3, 121: 3}                  # stairs stairway step


def to_tensor(rgb, scale=1.0, pad=1):
    if scale != 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = rgb.shape[:2]
    ph, pw = (-h) % pad, (-w) % pad
    rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    x = (rgb.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(dev), (h, w)


class PIDNet:
    def __init__(self, scale=1.0):
        sys.path.insert(0, ROOT + '/vendor')
        from semantic_local_grid.inference import Segmenter  # deployed loader, unchanged
        self.seg = Segmenter(PIDNET_W, 'cuda', 'pidnet-m', THR)
        self.m, self.scale, self.map = self.seg.model, scale, CS

    def logits(self, rgb):
        x, (h, w) = to_tensor(rgb, self.scale, 64)
        y = self.m(x)
        y = torch.nn.functional.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
        return y[:, :, :h, :w]


class SegF:
    def __init__(self, name, scale=1.0):
        from transformers import SegformerForSemanticSegmentation
        p = glob.glob(f'{HF}/models--nvidia--segformer-{name}/snapshots/*')[0]
        self.m = SegformerForSemanticSegmentation.from_pretrained(p).to(dev).eval()
        self.scale, self.map = scale, (ADE if 'ade' in name else CS)

    def logits(self, rgb):
        x, (h, w) = to_tensor(rgb, self.scale, 32)
        y = self.m(pixel_values=x).logits
        y = torch.nn.functional.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y[:, :, :h, :w]


def predict(model, rgb):
    with torch.inference_mode():
        y = model.logits(rgb)
        y = torch.nn.functional.interpolate(y, size=rgb.shape[:2], mode='bilinear', align_corners=False)
        conf, cls = y.softmax(1).max(1)
    conf, cls = conf[0].cpu().numpy(), cls[0].cpu().numpy()
    cat = np.zeros(cls.shape, np.uint8)
    for k, v in model.map.items():
        cat[cls == k] = v
    cat[conf < THR] = 255
    return cat


def bench(model, rgb, half=False, n=50):
    ctx = torch.autocast('cuda', dtype=torch.float16) if half else torch.autocast('cuda', enabled=False)
    with torch.inference_mode(), ctx:
        for _ in range(10):
            model.logits(rgb)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); model.logits(rgb); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


COLORS = {1: (0, 220, 0), 2: (255, 200, 0), 3: (255, 0, 255), 255: (128, 128, 128)}


def overlay(rgb, cat, title):
    out = rgb.astype(np.float32)
    for k, c in COLORS.items():
        m = cat == k
        a = .35 if k == 255 else .5
        out[m] = out[m] * (1 - a) + np.array(c) * a
    out = out.astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(bgr, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    return bgr


def main():
    os.makedirs(OUT + '/grid', exist_ok=True)
    files = sorted(glob.glob(IMG_DIR + '/*.png'))
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    specs = []
    if os.path.isfile(PIDNET_W):
        specs += [('PIDNet-M CS (deployed)', lambda: PIDNet(1.0)),
                  ('PIDNet-M CS x2', lambda: PIDNet(2.0))]
    specs += [('SegF-B0 CS', lambda: SegF('b0-finetuned-cityscapes-1024-1024')),
              ('SegF-B1 CS', lambda: SegF('b1-finetuned-cityscapes-1024-1024')),
              ('SegF-B1 CS x2', lambda: SegF('b1-finetuned-cityscapes-1024-1024', 2.0)),
              ('SegF-B0 ADE', lambda: SegF('b0-finetuned-ade-512-512')),
              ('SegF-B1 ADE', lambda: SegF('b1-finetuned-ade-512-512')),
              ('SegF-B2 ADE', lambda: SegF('b2-finetuned-ade-512-512'))]
    results, stats = {}, {}
    for name, make in specs:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m = make()
        cats = [predict(m, im) for im in imgs]
        results[name] = cats
        a = np.stack(cats)
        stats[name] = {
            'ms_fp32': bench(m, imgs[0]), 'ms_fp16': bench(m, imgs[0], True),
            'peak_mem_mb': torch.cuda.max_memory_allocated() / 2**20,
            'params_m': sum(p.numel() for p in m.m.parameters()) / 1e6,
            'paved': float((a == 1).mean()), 'grass_soil': float((a == 2).mean()),
            'stairs': float((a == 3).mean()), 'unknown': float((a == 255).mean()),
            # bottom-centre strip = ground right in front of the robot
            'paved_front': float((a[:, -120:, 160:480] == 1).mean()),
            'unknown_front': float((a[:, -120:, 160:480] == 255).mean()),
            'frames_with_stairs': int(((a == 3).reshape(len(a), -1).mean(1) > .01).sum()),
        }
        print(name, {k: round(v, 3) for k, v in stats[name].items()}, flush=True)
        del m
    names = list(results)
    # agreement between models on "paved" mask (IoU)
    agree = {}
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            a, b = np.stack(results[p]) == 1, np.stack(results[q]) == 1
            agree[f'{p} | {q}'] = float((a & b).sum() / max((a | b).sum(), 1))
    show = [n for n in names if not n.endswith('x2') and n != 'SegF-B0 CS'][:5]
    for i, f in enumerate(files):
        tiles = [overlay(imgs[i], np.zeros(imgs[i].shape[:2], np.uint8), 'original')]
        tiles += [overlay(imgs[i], results[n][i], n) for n in show]
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)]
        g = np.vstack(rows)
        g = cv2.resize(g, None, fx=.6, fy=.6, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{OUT}/grid/{i:03d}_{os.path.basename(f)[:-4]}.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump({'frames': len(files), 'models': stats, 'paved_iou': agree,
               'legend': 'green=paved road/sidewalk/path, yellow=grass/soil, magenta=stairs, grey=unknown(<0.7)'},
              open(OUT + '/stats.json', 'w'), indent=1)
    print('done', len(files), 'frames')


if __name__ == '__main__':
    main()
