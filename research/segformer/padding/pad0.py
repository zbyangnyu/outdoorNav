import glob, numpy as np, cv2, torch, compare as c
files = sorted(glob.glob(c.IMG_DIR + '/*.png'))
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
class Pad0(c.PIDNet):
    def logits(self, rgb):
        h, w = rgb.shape[:2]
        canvas = np.zeros((1024, 2048, 3), np.uint8)       # black canvas, Cityscapes size
        y0, x0 = (1024 - h) // 2, (2048 - w) // 2           # put image in the centre
        canvas[y0:y0+h, x0:x0+w] = rgb
        x, _ = c.to_tensor(canvas, 1.0, 64)
        y = torch.nn.functional.interpolate(self.m(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        return y[:, :, y0:y0+h, x0:x0+w]
for name, m in [('as deployed (replicate pad to 640x512)', c.PIDNet(1.0)), ('black canvas 2048x1024', Pad0(1.0))]:
    f = np.array([(c.predict(m, im)[-120:, 160:480] == 1).mean() for im in imgs])
    print(f'{name:40s} front_road={f.mean():.3f} front<50%: {int((f<.5).sum()):3d}/100')
