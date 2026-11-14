"""Export SegFormer-B0 Cityscapes (HF) to ONNX.

Input : 'input'  float32 [1,3,H,W], RGB, (x/255-mean)/std (ImageNet mean/std)
Output: 'logits' float32 [1,19,H/4,W/4]  (Cityscapes 19 classes; road=0 sidewalk=1 terrain=9)
Usage : python export_onnx.py [out_dir]   (needs torch, transformers, onnx; downloads weights if not cached)
Produces segf_b0_480x640.onnx, segf_b0_240x320.onnx, segf_b0_dyn.onnx
"""
import sys, os, torch
from transformers import SegformerForSemanticSegmentation

out = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(out, exist_ok=True)
m = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-1024-1024').eval()


class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(pixel_values=x).logits


w = Wrap(m)
for name, (h, ww), dyn in [('480x640', (480, 640), False), ('240x320', (240, 320), False), ('dyn', (480, 640), True)]:
    f = f'{out}/segf_b0_{name}.onnx'
    torch.onnx.export(w, torch.randn(1, 3, h, ww), f, input_names=['input'], output_names=['logits'],
                      opset_version=17, dynamo=False,
                      dynamic_axes={'input': {2: 'H', 3: 'W'}, 'logits': {2: 'h', 3: 'w'}} if dyn else None)
    print('wrote', f)
