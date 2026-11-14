import glob, numpy as np, cv2, torch, compare as c
files = sorted(glob.glob(c.IMG_DIR + '/*.png'))
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
class P(c.PIDNet):
    def __init__(self, top, bottom, side):
        super().__init__(1.0); self.t, self.b, self.s = top, bottom, side
    def logits(self, rgb):
        h, w = rgb.shape[:2]
        canvas = cv2.copyMakeBorder(rgb, self.t, self.b, self.s, self.s, cv2.BORDER_CONSTANT, value=0)
        x, _ = c.to_tensor(canvas, 1.0, 64)
        y = torch.nn.functional.interpolate(self.m(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        return y[:, :, self.t:self.t+h, self.s:self.s+w]
for name, args in [('black bottom 32 rows -> 640x512 (same size as deployed)', (0, 32, 0)),
                   ('black frame 32px all round -> 704x576', (32, 32+32, 32)),
                   ('black frame 64px all round -> 768x640', (64, 64, 64))]:
    m = P(*args)
    f = np.array([(c.predict(m, im)[-120:, 160:480] == 1).mean() for im in imgs])
    print(f'{name:55s} front_road={f.mean():.3f} front<50%: {int((f<.5).sum()):3d}/100')
    del m; torch.cuda.empty_cache()
