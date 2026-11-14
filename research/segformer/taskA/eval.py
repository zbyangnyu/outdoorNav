import glob, sys, cv2, numpy as np
sys.path.insert(0, '/root/autodl-tmp/seg/taskA/repo')
from outdoor_nav.segmentation import validate_segmentation, create_backend, predict_mask
from pathlib import Path
w = glob.glob('/root/autodl-tmp/hf/hub/models--nvidia--segformer-b0-finetuned-cityscapes-1024-1024/snapshots/*')[0]
files = sorted(glob.glob('/root/autodl-tmp/seg/camera_gemini330__color__image_raw/*.png'))
for s in (1.0, .5):
    cfg = validate_segmentation({'enabled': True, 'backend': 'outdoor_nav.segformer_backend:create_backend', 'weights': w, 'options': {'input_scale': s}}, Path('/'), True)
    b = create_backend(cfg)
    r = []
    for f in files:
        m = predict_mask(b, cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), cfg['confidence_threshold'])
        r.append((m[-120:, 160:480] == 1).mean())
    r = np.array(r); print(f'scale={s} n={len(r)} mean={r.mean():.4f} lt50={(r<.5).sum()}')
