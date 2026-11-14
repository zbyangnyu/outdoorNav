"""Dump full-class predictions of every method on a few example images (for the PDF)."""
import glob, json, os, numpy as np, cv2, torch, compare as c
from transformers import SegformerForSemanticSegmentation

OUT = '/root/autodl-tmp/seg/out_full'
ADE = '/root/autodl-tmp/ade/ADEChallengeData2016'
CS = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
      'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def tens(rgb):
    x = (rgb.astype(np.float32) / 255 - c.MEAN) / c.STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(c.dev)


def finish(y, h, w):
    y = torch.nn.functional.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
    conf, cls = y.softmax(1).max(1)
    return cls[0].cpu().numpy().astype(np.uint8), conf[0].cpu().numpy().astype(np.float16)


def fit(rgb, mult):
    """Lynx 640x480 stays as is; other photos: short side 480, sides rounded to `mult` (no padding)."""
    h, w = rgb.shape[:2]
    if (h, w) == (480, 640):
        return rgb
    s = 480 / min(h, w)
    return cv2.resize(rgb, (round(w * s / mult) * mult, round(h * s / mult) * mult))


def pidnet(m, border):
    def f(rgb):
        h, w = rgb.shape[:2]
        x = fit(rgb, 64) if (h, w) != (480, 640) else rgb
        ph = (-x.shape[0]) % 64
        x = cv2.copyMakeBorder(x, 0, ph, 0, 0, border, value=0)
        with torch.inference_mode():
            y = m.m(tens(x))
            y = torch.nn.functional.interpolate(y, size=x.shape[:2], mode='bilinear', align_corners=True)
            y = y[:, :, :x.shape[0] - ph]
            return finish(y, h, w)
    return f


def segf(path):
    m = SegformerForSemanticSegmentation.from_pretrained(path).to(c.dev).eval()
    def f(rgb):
        h, w = rgb.shape[:2]
        with torch.inference_mode():
            return finish(m(pixel_values=tens(fit(rgb, 32))).logits, h, w)
    return f, [m.config.id2label[i] for i in range(m.config.num_labels)]


def hf(n):
    return glob.glob(f'{c.HF}/models--nvidia--segformer-{n}/snapshots/*')[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    lynx = sorted(glob.glob(c.IMG_DIR + '/*.png'))
    imgs = {f'lynx_{i:03d}': lynx[i] for i in (66, 89, 96)}
    for k in ('042', '013', '038', '041'):
        imgs[f'web_{k}'] = f'/root/autodl-tmp/seg/stairs_web/web_{k}.jpg'
    # two ADE validation photos with both stairs and road/sidewalk in view, plus ground truth
    picked = []
    for a in sorted(glob.glob(f'{ADE}/annotations/validation/*.png')):
        g = cv2.imread(a, cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
        if np.isin(g, [53, 59, 121]).mean() > .08 and np.isin(g, [6, 11, 52]).mean() > .05:
            picked.append((a, g))
        if len(picked) == 2:
            break
    for a, g in picked:
        k = 'ade_' + os.path.basename(a)[:-4].split('_')[-1]
        imgs[k] = a.replace('annotations', 'images').replace('.png', '.jpg')
        np.save(f'{OUT}/{k}_gt.npy', g.astype(np.int16))
    rgbs = {k: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for k, p in imgs.items()}
    for k, v in rgbs.items():
        cv2.imwrite(f'{OUT}/{k}_rgb.jpg', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    pm = c.PIDNet(1.0)
    b0cs, cs_names = segf(hf('b0-finetuned-cityscapes-1024-1024'))
    b0ade, ade_names = segf(hf('b0-finetuned-ade-512-512'))
    ft, ft_names = segf('/root/autodl-tmp/seg/ft_b0')
    methods = {'pidnet_rep': (pidnet(pm, cv2.BORDER_REPLICATE), CS),
               'pidnet_blk': (pidnet(pm, cv2.BORDER_CONSTANT), CS),
               'b0_cs': (b0cs, cs_names), 'b0_ade': (b0ade, ade_names), 'b0_ft': (ft, ft_names)}
    for k, rgb in rgbs.items():
        for mk, (f, names) in methods.items():
            if mk == 'pidnet_blk' and not k.startswith('lynx'):
                continue                       # padding only differs on 640x480 camera frames
            cls, conf = f(rgb)
            np.savez_compressed(f'{OUT}/{k}__{mk}.npz', cls=cls, conf=conf)
        print('done', k, rgb.shape, flush=True)
    json.dump({'cs': CS, 'ade': ade_names, 'ft': ft_names, 'images': list(imgs)}, open(f'{OUT}/labels.json', 'w'))
    print('DUMP_DONE')


if __name__ == '__main__':
    main()
