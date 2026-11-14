import glob, numpy as np, cv2, torch, compare as c
files = sorted(glob.glob(c.IMG_DIR + '/*.png'))
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
for name, make in [('PIDNet-M deployed', lambda: c.PIDNet(1.0)), ('PIDNet-M x2', lambda: c.PIDNet(2.0)),
                   ('SegF-B1 CS', lambda: c.SegF('b1-finetuned-cityscapes-1024-1024')),
                   ('SegF-B0 ADE', lambda: c.SegF('b0-finetuned-ade-512-512')),
                   ('SegF-B1 ADE', lambda: c.SegF('b1-finetuned-ade-512-512')),
                   ('SegF-B2 ADE', lambda: c.SegF('b2-finetuned-ade-512-512'))]:
    m = make()
    f = np.array([(c.predict(m, im)[-120:, 160:480] == 1).mean() for im in imgs])
    print(f'{name:18s} front<50%: {int((f<.5).sum()):3d}/100  front<80%: {int((f<.8).sum()):3d}/100  worst={f.min():.2f}')
    del m; torch.cuda.empty_cache()
