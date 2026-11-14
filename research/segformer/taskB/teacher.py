"""Make road / non-road reference masks with big "teacher" models (NOT human labels).

PNG per image: 1 = road (road+sidewalk), 0 = non-road, 255 = ignore (teachers disagree).
usage: python teacher.py IMG_DIR OUT_DIR
"""
import glob, os, sys
import numpy as np, cv2, torch
from transformers import (SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation)

HERE = os.path.dirname(os.path.abspath(__file__))
dev = torch.device('cuda')
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
ROAD = [0, 1]  # cityscapes road, sidewalk


def prep(rgb, long_side=1024, pad=32):
    s = long_side / max(rgb.shape[:2])
    x = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    h, w = x.shape[:2]
    x = cv2.copyMakeBorder(x, 0, (-h) % pad, 0, (-w) % pad, cv2.BORDER_REPLICATE)
    x = (x.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(dev), (h, w)


def probs_to_road(p, hw, size):
    p = p[:, :, :hw[0], :hw[1]]
    p = torch.nn.functional.interpolate(p, size=size, mode='bilinear', align_corners=False)
    return (p[0].argmax(0)[None] == torch.tensor(ROAD, device=dev)[:, None, None]).any(0).cpu().numpy()


def main(img_dir, out):
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(img_dir + '/*.png') + glob.glob(img_dir + '/*.jpg'))
    b5 = SegformerForSemanticSegmentation.from_pretrained(f'{HERE}/m/b5').to(dev).eval()
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(f'{HERE}/m/m2f').to(dev).eval()
    agree = []
    for f in files:
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        size = rgb.shape[:2]
        with torch.inference_mode():
            x, hw = prep(rgb)
            a = probs_to_road(b5(pixel_values=x).logits.softmax(1), hw, size)
            o = m2f(pixel_values=x)
            # semantic map = sum over queries of class prob * mask prob
            cls = o.class_queries_logits.softmax(-1)[..., :-1]
            msk = o.masks_queries_logits.sigmoid()
            sem = torch.einsum('bqc,bqhw->bchw', cls, msk)
            sem = torch.nn.functional.interpolate(sem, size=x.shape[2:], mode='bilinear', align_corners=False)
            b = probs_to_road(sem, hw, size)
        ref = np.full(size, 255, np.uint8)
        ref[a & b] = 1
        ref[~a & ~b] = 0
        agree.append((ref != 255).mean())
        cv2.imwrite(f'{out}/{os.path.splitext(os.path.basename(f))[0]}.png', ref)
    print('teacher agreement (non-ignored share): mean %.3f min %.3f' % (np.mean(agree), np.min(agree)))


if __name__ == '__main__':
    main(*sys.argv[1:3])
