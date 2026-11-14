"""Export fine-tuned B0 to ONNX (input 1x3x480x640 normalized RGB, output logits 1x4x120x160) and
check against PyTorch on the 100 Lynx frames. usage: python export_onnx.py CKPT OUT.onnx"""
import glob, sys, json, numpy as np, cv2, torch, onnxruntime as ort
from transformers import SegformerForSemanticSegmentation
CK, OUT = sys.argv[1], sys.argv[2]
MEAN = np.array([.485, .456, .406], np.float32); STD = np.array([.229, .224, .225], np.float32)
m = SegformerForSemanticSegmentation.from_pretrained(CK).eval()
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(pixel_values=x).logits
x0 = torch.randn(1, 3, 480, 640)
torch.onnx.export(Wrap(m), x0, OUT, input_names=['pixel_values'], output_names=['logits'], opset_version=17, dynamo=False)
m = SegformerForSemanticSegmentation.from_pretrained(CK).eval()  # fresh copy: export can leave traced state behind
sess = ort.InferenceSession(OUT, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
md, agree, sess_cpu = 0, [], ort.InferenceSession(OUT, providers=['CPUExecutionProvider']); mdc, agc = 0, []
for f in sorted(glob.glob('/root/autodl-tmp/seg/camera_gemini330__color__image_raw/*.png')):
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); x = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    with torch.no_grad(): a = m(pixel_values=torch.from_numpy(x)).logits.numpy()
    b = sess.run(None, {'pixel_values': x})[0]
    md = max(md, float(np.abs(a - b).max())); agree.append(float((a.argmax(1) == b.argmax(1)).mean()))
    c = sess_cpu.run(None, {'pixel_values': x})[0]; mdc = max(mdc, float(np.abs(a - c).max())); agc.append(float((a.argmax(1) == c.argmax(1)).mean()))
r = {'onnx': OUT, 'max_abs_logit_diff': md, 'argmax_agreement_min': min(agree), 'argmax_agreement_mean': float(np.mean(agree)),
     'providers': sess.get_providers(), 'cpu_max_abs_logit_diff': mdc, 'cpu_argmax_agreement_min': min(agc)}
print(r); json.dump(r, open(OUT[:-5] + '_check.json', 'w'), indent=1)
