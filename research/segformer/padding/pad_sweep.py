import glob, numpy as np, cv2, torch, compare as c
files = sorted(glob.glob(c.IMG_DIR + '/*.png'))
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
class PadN(c.PIDNet):
    def __init__(self, H, W, mode):
        super().__init__(1.0); self.H, self.W, self.mode = H, W, mode
    def logits(self, rgb):
        h, w = rgb.shape[:2]
        y0, x0 = (self.H - h) // 2, (self.W - w) // 2
        if self.mode == 'black':
            canvas = np.zeros((self.H, self.W, 3), np.uint8); canvas[y0:y0+h, x0:x0+w] = rgb
        else:
            canvas = cv2.copyMakeBorder(rgb, y0, self.H-h-y0, x0, self.W-w-x0, cv2.BORDER_REPLICATE)
        x, _ = c.to_tensor(canvas, 1.0, 64)
        y = torch.nn.functional.interpolate(self.m(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        return y[:, :, y0:y0+h, x0:x0+w]
for H, W, mode in [(640, 768, 'black'), (768, 1024, 'black'), (1024, 1024, 'black'), (1024, 2048, 'black'), (1024, 2048, 'replicate')]:
    m = PadN(H, W, mode)
    f = np.array([(c.predict(m, im)[-120:, 160:480] == 1).mean() for im in imgs])
    print(f'{mode:9s} {W}x{H}  pixels x{H*W/(512*640):.1f}  front_road={f.mean():.3f} front<50%: {int((f<.5).sum()):3d}/100')
    del m; torch.cuda.empty_cache()
