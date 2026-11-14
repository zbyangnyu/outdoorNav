"""Check ONNX/ORT/TRT outputs match PyTorch on 100 frames and benchmark (RTX 4090D server).
Run in /root/autodl-tmp/seg/taskC after export_onnx.py.  Prints a markdown table.
"""
import glob, sys, time, json, os
import numpy as np, cv2, torch, torch.nn.functional as F
import onnxruntime as ort, tensorrt as trt

ROOT = '/root/autodl-tmp/seg'
dev = torch.device('cuda')
MEAN = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)
THR = 0.7
files = sorted(glob.glob(ROOT + '/camera_gemini330__color__image_raw/*.png'))[:100]
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
N_WARM, N_RUN = 30, 300


def pre(rgb, H, W, pad_to=None):
    """uint8 HxWx3 numpy -> normalized 1x3xHxW cuda (GPU resize)."""
    x = torch.from_numpy(rgb).to(dev).permute(2, 0, 1)[None].float() / 255
    if x.shape[2:] != (H, W) and pad_to is None:
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
    x = (x - MEAN) / STD
    if pad_to:
        x = F.pad(x, (0, pad_to[1] - x.shape[3], 0, pad_to[0] - x.shape[2]), mode='replicate')
    return x.contiguous()


def post(logits, crop=None):
    """logits 1xCxhxw cuda -> category uint8 480x640 numpy (1 paved, 2 terrain, 255 unknown)."""
    y = logits.float()
    if crop:  # PIDNet: upsample to padded size then crop
        y = F.interpolate(y, size=crop[0], mode='bilinear', align_corners=True)[:, :, :480, :640]
    y = F.interpolate(y, size=(480, 640), mode='bilinear', align_corners=False)
    conf, cls = y.softmax(1).max(1)
    cat = torch.zeros_like(cls, dtype=torch.uint8)
    cat[(cls == 0) | (cls == 1)] = 1
    cat[cls == 9] = 2
    cat[conf < THR] = 255
    return cat[0].cpu().numpy()


def front(cats):
    a = np.stack(cats)
    return float((a[:, -120:, 160:480] == 1).mean())


def med(fn):
    for _ in range(N_WARM): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(N_RUN):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


# ---------- backends: each is f(x_cuda) -> logits_cuda ----------
def torch_fn(model, half):
    def f(x):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=half):
            return model(x)
    return f


def ort_fn(path):
    s = ort.InferenceSession(path, providers=['CUDAExecutionProvider'])
    outs = {}
    def f(x):
        H, W = x.shape[2:]
        if (H, W) not in outs:
            outs[(H, W)] = torch.empty(1, 19, -(-H // 4), -(-W // 4), device=dev)
        o = outs[(H, W)]
        b = s.io_binding()
        b.bind_input('input', 'cuda', 0, np.float32, list(x.shape), x.data_ptr())
        b.bind_output('logits', 'cuda', 0, np.float32, list(o.shape), o.data_ptr())
        s.run_with_iobinding(b)
        return o
    return f


def trt_fn(onnx_path, plan_path, shape):
    lg = trt.Logger(trt.Logger.WARNING)
    if not os.path.exists(plan_path):
        b = trt.Builder(lg); net = b.create_network(0); p = trt.OnnxParser(net, lg)
        assert p.parse(open(onnx_path, 'rb').read()), p.get_error(0)
        c = b.create_builder_config(); c.set_flag(trt.BuilderFlag.FP16)
        c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        open(plan_path, 'wb').write(b.build_serialized_network(net, c))
    eng = trt.Runtime(lg).deserialize_cuda_engine(open(plan_path, 'rb').read())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    outs = {n: torch.empty(tuple(eng.get_tensor_shape(n)), device=dev, dtype=torch.float32)
            for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT}
    inp = [n for n in names if n not in outs][0]
    for n, t in outs.items(): ctx.set_tensor_address(n, t.data_ptr())
    out0 = list(outs.values())[0]
    stream = torch.cuda.Stream()
    def f(x):
        ctx.set_tensor_address(inp, x.data_ptr())
        ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
        return out0
    return f


def run_all(fn, H, W, pad=None, crop=None):
    return [post(fn(pre(im, H, W, pad)), crop) for im in imgs]


def e2e(fn, H, W, pad=None, crop=None):
    return med(lambda: post(fn(pre(imgs[0], H, W, pad)), crop))


rows, checks = [], []
from transformers import SegformerForSemanticSegmentation
sf = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-1024-1024').to(dev).eval()
sfm = lambda x: sf(pixel_values=x).logits

for res, (H, W) in [('480x640', (480, 640)), ('240x320', (240, 320))]:
    x0 = pre(imgs[0], H, W)
    pt32, pt16 = torch_fn(sfm, False), torch_fn(sfm, True)
    ref = run_all(pt32, H, W); fref = front(ref)
    backends = [('PyTorch fp32', pt32), ('PyTorch fp16 (autocast)', pt16),
                ('ORT CUDA fp32', ort_fn(f'segf_b0_{res}.onnx')),
                ('ORT CUDA fp32 (dynamic onnx)', ort_fn('segf_b0_dyn.onnx')),
                ('TensorRT fp16', trt_fn(f'segf_b0_{res}.onnx', f'segf_b0_{res}_fp16.plan', (H, W)))]
    for name, fn in backends:
        cats = ref if name == 'PyTorch fp32' else run_all(fn, H, W)
        a, b = np.stack(ref) == 1, np.stack(cats) == 1
        maxerr = max(float((fn(pre(im, H, W)).float() - pt32(pre(im, H, W)).float()).abs().max()) for im in imgs[:20])
        checks.append(dict(model='SegF-B0', res=res, backend=name, paved_agree=float((a == b).mean()),
                           paved_iou=float((a & b).sum() / max((a | b).sum(), 1)),
                           front=front(cats), front_ref=fref, max_logit_err=maxerr))
        rows.append(dict(model='SegF-B0', res=res, backend=name, ms_model=med(lambda: fn(x0)), ms_e2e=e2e(fn, H, W)))
        print(checks[-1], rows[-1], flush=True)
del sf; torch.cuda.empty_cache()

# ---------- PIDNet-M 640x512 (deployed size: 480 padded to 512) ----------
sys.path.insert(0, ROOT + '/vendor')
from semantic_local_grid.inference import Segmenter
pm = Segmenter(ROOT + '/w/PIDNet_M_Cityscapes_test.pt', 'cuda', 'pidnet-m', THR).model.eval()
if not os.path.exists('pidnet_m_512x640.onnx'):
    torch.onnx.export(pm, torch.randn(1, 3, 512, 640, device=dev), 'pidnet_m_512x640.onnx',
                      input_names=['input'], output_names=['logits'], opset_version=17, dynamo=False)
P, C = (512, 640), ((512, 640),)
pt32 = torch_fn(pm, False)
ref = run_all(pt32, 480, 640, P, C[0]); fref = front(ref)
x0 = pre(imgs[0], 480, 640, P)
for name, fn in [('PyTorch fp32', pt32), ('PyTorch fp16 (autocast)', torch_fn(pm, True)),
                 ('TensorRT fp16', trt_fn('pidnet_m_512x640.onnx', 'pidnet_m_512x640_fp16.plan', P))]:
    cats = ref if name == 'PyTorch fp32' else run_all(fn, 480, 640, P, C[0])
    a, b = np.stack(ref) == 1, np.stack(cats) == 1
    checks.append(dict(model='PIDNet-M', res='512x640', backend=name, paved_agree=float((a == b).mean()),
                       paved_iou=float((a & b).sum() / max((a | b).sum(), 1)), front=front(cats), front_ref=fref,
                       max_logit_err=float((fn(x0).float() - pt32(x0).float()).abs().max())))
    rows.append(dict(model='PIDNet-M', res='512x640', backend=name, ms_model=med(lambda: fn(x0)),
                     ms_e2e=e2e(fn, 480, 640, P, C[0])))
    print(checks[-1], rows[-1], flush=True)

json.dump(dict(speed=rows, checks=checks), open('results.json', 'w'), indent=1)
print('\n| 模型 | 输入(H×W) | 后端 | 仅模型 ms | 端到端 ms | 路面掩码一致率 | 前方路面比例 (PyTorch fp32) | logits 最大误差 |')
print('|---|---|---|---|---|---|---|---|')
for r, c in zip(rows, checks):
    print(f"| {r['model']} | {r['res']} | {r['backend']} | {r['ms_model']:.2f} | {r['ms_e2e']:.2f} | "
          f"{c['paved_agree']*100:.2f}% | {c['front']:.3f} ({c['front_ref']:.3f}) | {c['max_logit_err']:.3f} |")
