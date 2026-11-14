"""Paired offline SegFormer stair experiment using Task E's model and augmentations."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import SegformerForSemanticSegmentation

from stair_split import STAIR_SCENES, load_split
from train import DS, build_model


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def stair_metrics(cm):
    return {
        "stairs_as_road": ratio(cm[3, 1], cm[3].sum()),
        "stairs_recall": ratio(cm[3, 3], cm[3].sum()),
        "stairs_iou": ratio(cm[3, 3], cm[3].sum() + cm[:3, 3].sum()),
        "stairs_pixels": int(cm[3].sum()),
    }


def summarize(rows, matrices):
    zeros = lambda: np.zeros((4, 4), dtype=np.int64)
    scenes = defaultdict(zeros)
    by_stair = defaultdict(zeros)
    counts = defaultdict(int)
    stair_counts = defaultdict(int)
    negatives = zeros()
    negative_frames = false_alarm_frames = 0
    for row, cm in zip(rows, matrices, strict=True):
        scene = row["scene"]
        counts[scene] += 1
        scenes[scene] += cm
        if scene in {"road", "no_stairs"}:
            negatives += cm
            negative_frames += 1
            false_alarm_frames += int(cm[:, 3].sum() > 0)
        if scene in STAIR_SCENES:
            key = (row["stair_id"], scene)
            by_stair[key] += cm
            stair_counts[key] += 1

    road = scenes["road"]
    return {
        "frames": len(rows),
        "scene_counts": dict(counts),
        "downstairs": stair_metrics(scenes["downstairs"]),
        "upstairs": stair_metrics(scenes["upstairs"]),
        "generic_stairs": stair_metrics(scenes["stairs"]),
        "road_iou": ratio(road[1, 1], road[1].sum() + road[:, 1].sum() - road[1, 1]),
        "no_stairs_frames": negative_frames,
        "no_stairs_false_positive_pixels": ratio(negatives[:, 3].sum(), negatives.sum()),
        "no_stairs_false_alarm_frames": ratio(false_alarm_frames, negative_frames),
        "by_stair": {
            group: {scene: {"frames": stair_counts[group, scene], **stair_metrics(by_stair[group, scene])}
                    for scene in sorted(s for g, s in by_stair if g == group)}
            for group in sorted({g for g, _ in by_stair})
        },
    }


def score(metrics, smoke):
    stairs = metrics["generic_stairs" if smoke else "downstairs"]
    values = (stairs["stairs_iou"], metrics["road_iou"], stairs["stairs_as_road"],
              metrics["no_stairs_false_positive_pixels"])
    if any(value is None for value in values):
        raise ValueError("Validation needs valid stair and ordinary road pixels after resize")
    stair_iou, road_iou, confusion, false_positive = values
    return (stair_iou + road_iou) / 2 - confusion - false_positive


def edge_loss(logits, labels, human, radius, annotated_edges=None):
    # Only boundaries between known stair and known non-stair pixels are positive.
    known = (labels != 255) & human[:, None, None]
    stairs = labels == 3
    edges = torch.zeros_like(known)
    vertical = known[:, 1:] & known[:, :-1] & (stairs[:, 1:] != stairs[:, :-1])
    horizontal = known[:, :, 1:] & known[:, :, :-1] & (stairs[:, :, 1:] != stairs[:, :, :-1])
    edges[:, 1:] |= vertical
    edges[:, :-1] |= vertical
    edges[:, :, 1:] |= horizontal
    edges[:, :, :-1] |= horizontal
    if annotated_edges is not None:
        edges |= annotated_edges & known
    kernel = 2 * radius + 1
    target = F.max_pool2d(edges[:, None].float(), kernel, stride=1, padding=radius)[:, 0].bool()
    unknown = F.max_pool2d((~known)[:, None].float(), kernel, stride=1, padding=radius)[:, 0].bool()
    positive = known & target & ~unknown
    negative = known & ~target & ~unknown
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    pos_count = positive.sum()
    neg_count = negative.sum()
    loss = (pixel_loss * positive).sum() / pos_count.clamp_min(1)
    loss = loss + (pixel_loss * negative).sum() / neg_count.clamp_min(1)
    return loss / ((pos_count > 0).int() + (neg_count > 0).int()).clamp_min(1), pos_count


class TrainingData(DS):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row = self.items[index]
        edge = sample[2] if self.with_edges else torch.empty(0, dtype=torch.bool)
        return sample[0], sample[1], edge, row["label_source"] == "human", row["src"] == "beihang"


def resize(logits, size):
    return F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False)


def amp(device, cfg):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg["bf16"])


@torch.inference_mode()
def evaluate(model, loader, device, cfg, thresholds):
    model.eval()
    matrices = {threshold: [] for threshold in thresholds}
    for image, label in loader:
        label = label.to(device)
        with amp(device, cfg):
            logits = model(pixel_values=image.to(device)).logits
        confidence, cls = resize(logits, label.shape[-2:]).softmax(1).max(1)
        valid = label != 255
        batch_offset = torch.arange(len(label), device=device)[:, None, None] * 16
        for threshold in thresholds:
            predicted = torch.where(confidence >= threshold, cls, 0)
            bins = batch_offset + label * 4 + predicted
            counts = torch.bincount(bins[valid], minlength=len(label) * 16)
            matrices[threshold].extend(counts.reshape(-1, 4, 4).cpu().numpy())
    return matrices


def eval_loader(rows, data, cfg):
    dataset = DS(rows, False, data, tuple(cfg["size"]))
    return DataLoader(dataset, batch_size=cfg["eval_batch_size"], num_workers=cfg["workers"])


def train(args, split, cfg, device):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    seed = cfg["seed"]
    edge_weight = cfg["edge_weight"] if args.edge_weight is None else args.edge_weight
    seed_everything(seed)
    rows = split["train"]
    weights = [3.0 if row["scene"] in STAIR_SCENES
               else .5 if row["src"] == "rugd" else 1.0 for row in rows]
    sampler = WeightedRandomSampler(weights, cfg["steps"] * cfg["batch_size"], replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    dataset = TrainingData(rows, True, args.data, tuple(cfg["size"]), with_edges=args.variant == "B")
    loader = DataLoader(dataset, cfg["batch_size"], sampler=sampler, drop_last=True,
                        num_workers=cfg["workers"], pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(seed + 1))
    val = eval_loader(split["val"], args.data, cfg)
    model = build_model(args.pretrained).to(device)
    teacher = SegformerForSemanticSegmentation.from_pretrained(args.pretrained).to(device)
    teacher.eval().requires_grad_(False)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]))
    edge_head = None
    if args.variant == "B":
        edge_head = nn.Conv2d(model.decode_head.classifier.in_channels, 1, 1).to(device)
        # The segmentation classifier already consumes shared, fused decoder features.
        def predict_edges(_classifier, inputs):
            edge_outputs.append(edge_head(inputs[0]))
        edge_outputs = []
        model.decode_head.classifier.register_forward_pre_hook(predict_edges)
    seed_everything(seed + 2)  # Keep dropout and zero-worker augmentation paired.
    backbone = list(model.segformer.parameters())
    head = list(model.decode_head.parameters())
    if edge_head is not None:
        head.extend(edge_head.parameters())
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": cfg["backbone_lr"]},
                                   {"params": head, "lr": cfg["head_lr"]}],
                                  weight_decay=cfg["weight_decay"])
    steps = cfg["steps"]
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(1, (i + 1) / cfg["warmup_steps"]) * (1 - i / steps) ** .9
    )
    weights = torch.tensor(cfg["class_weights"], device=device)
    common = {
        "config": cfg, "split_sha256": digest(args.split),
        "pretrained_config_sha256": digest(args.pretrained / "config.json"),
        "pretrained_sha256": digest(next(
            args.pretrained / filename for filename in ("model.safetensors", "pytorch_model.bin")
            if (args.pretrained / filename).is_file()
        )),
        "torch": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    start = perf_counter()
    with (output / "log.jsonl").open("w", encoding="utf-8") as log:
        for step, (image, label, edge, human, stair_only) in enumerate(loader, 1):
            model.train()
            image, label, human = image.to(device), label.to(device), human.to(device)
            with amp(device, cfg):
                logits = model(pixel_values=image).logits
                with torch.no_grad():
                    teacher_logits = teacher(pixel_values=image).logits
            size = label.shape[-2:]
            logits = resize(logits, size)
            # Stair-only masks annotate where stairs end, but do not classify road/soil.
            semantic = label.masked_fill(stair_only.to(device)[:, None, None] & (label == 0), 255)
            pixels = semantic != 255
            # Normalise by the class-weight sum to match Task E's weighted mean.
            segmentation = F.cross_entropy(logits, semantic, weight=weights, ignore_index=255, reduction="sum")
            segmentation = segmentation / weights[semantic[pixels]].sum().clamp_min(1)
            teacher_probs = resize(teacher_logits, size).softmax(1)
            target = (teacher_probs[:, 0] + teacher_probs[:, 1]).clamp(1e-4, 1 - 1e-4)
            road = logits.softmax(1)[:, 1].clamp(1e-4, 1 - 1e-4)
            distill_mask = (label == 255) | (label == 1) | (label == 0)
            distill = (F.binary_cross_entropy(road, target, reduction="none") * distill_mask).sum()
            distill = distill / distill_mask.sum().clamp_min(1)
            loss = segmentation + cfg["distill_weight"] * distill
            stats = {"step": step}
            if edge_head is not None:
                boundary, positives = edge_loss(resize(edge_outputs.pop(), size)[:, 0], label, human,
                                                cfg["edge_radius"], edge.to(device))
                loss = loss + edge_weight * boundary
                if step == 1 or step % cfg["log_every"] == 0 or step == steps:
                    stats.update(edge_loss=boundary.item(), edge_positive_pixels=positives.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            ema.update_parameters(model)
            if step == 1 or step % cfg["log_every"] == 0 or step == steps:
                stats["loss"] = loss.item()
                print(json.dumps(stats), file=log, flush=True)
            if step % cfg["validate_every"] == 0 or step == steps:
                candidates = evaluate(ema.module, val, device, cfg, cfg["thresholds"])
                results = []
                for threshold, matrices in candidates.items():
                    metrics = summarize(split["val"], matrices)
                    rating = score(metrics, args.smoke)
                    result = dict(variant=args.variant, step=step, threshold=threshold,
                                  score=rating, val=metrics)
                    print(json.dumps(result), file=log, flush=True)
                    results.append(result)
                # Pin the selected weights to the same training step in both variants.
                best = max(results, key=lambda result: result["score"])
                if step == steps:
                    ema.module.save_pretrained(output / "selected")
                    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    write_json(output / "selected.json",
                               {**common, **best, "peak_allocated_bytes": peak,
                                "seconds": perf_counter() - start,
                                "edge_weight": edge_weight if edge_head is not None else None})
                print(f"{args.variant} step {step}/{steps}, best validation score at this step="
                      f"{best['score']:.4f}", flush=True)


@torch.inference_mode()
def benchmark(model, device, cfg):
    image = torch.zeros((1, 3, *cfg["size"]), device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    for _ in range(10):
        with amp(device, cfg):
            model(pixel_values=image)
    samples = []
    for _ in range(40):
        sync()
        start = perf_counter()
        with amp(device, cfg):
            model(pixel_values=image)
        sync()
        samples.append((perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95))}


@torch.inference_mode()
def draw_cases(file, indices, rows, data, models, selections, device, cfg):
    palette = np.array([[90, 90, 90], [37, 200, 96], [218, 145, 43], [186, 89, 236]], np.uint8)
    height, width = 168, 224
    canvas = Image.new("RGB", (width * 4, (height + 23) * len(indices) + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for col, heading in enumerate(("RGB", "Annotation", "A", "B")):
        draw.text((col * width + 5, 5), heading, fill="black")
    dataset = DS(rows, False, data, tuple(cfg["size"]))
    for line, (index, description) in enumerate(indices):
        row = rows[index]
        rgb = Image.open(data / row["image"]).convert("RGB").resize((width, height))
        image, label = dataset[index]
        maps = [label.numpy()]
        for model, selection in zip(models, selections, strict=True):
            with amp(device, cfg):
                logits = model(pixel_values=image[None].to(device)).logits
            conf, cls = resize(logits, label.shape).softmax(1).max(1)
            maps.append(torch.where(conf >= selection["threshold"], cls, 0)[0].cpu().numpy())
        y = 24 + line * (height + 23)
        canvas.paste(rgb, (0, y))
        for col, ids in enumerate(maps, 1):
            mask = palette[np.minimum(ids, 3)]
            mask[ids == 255] = 32
            tile = Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST)
            canvas.paste(Image.blend(rgb, tile, .55), (col * width, y))
        draw.text((5, y + height + 4), f"{description} | {row['n']}", fill="black")
    canvas.save(file)


def compare(args, split, cfg, device):
    selections = [read_json(run / "selected.json") for run in (args.a, args.b)]
    identity = ("step", "split_sha256", "config", "pretrained_sha256", "pretrained_config_sha256", "torch", "gpu")
    if [entry["variant"] for entry in selections] != ["A", "B"] or any(
        selections[0][key] != selections[1][key] for key in identity
    ) or selections[0]["config"] != cfg or selections[0]["split_sha256"] != digest(args.split) \
            or selections[0]["step"] != cfg["steps"]:
        raise ValueError("A and B must use the same split, initialization, environment and shared configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = split["test"]
    loader = eval_loader(rows, args.data, cfg)
    models = [SegformerForSemanticSegmentation.from_pretrained(run / "selected").to(device).eval()
              for run in (args.a, args.b)]
    results = {}
    per_model = []
    for key, run, model, selection in zip(("A", "B"), (args.a, args.b), models, selections, strict=True):
        threshold = selection["threshold"]
        matrices = evaluate(model, loader, device, cfg, [threshold])[threshold]
        per_model.append(matrices)
        results[key] = {"step": selection["step"], "threshold": threshold,
                        "peak_training_allocated_bytes": selection["peak_allocated_bytes"],
                        "edge_weight": selection.get("edge_weight", cfg["edge_weight"] if key == "B" else None),
                        "weights_sha256": digest(run / "selected/model.safetensors"),
                        **summarize(rows, matrices), "forward_speed": benchmark(model, device, cfg)}
    a, b = results["A"], results["B"]
    error_a, error_b = (value["downstairs"]["stairs_as_road"] for value in (a, b))
    results["relative_downstairs_error_reduction"] = ratio(error_a - error_b, error_a) if error_a else None
    for value in (a, b):
        for metric in ("stairs_iou", "stairs_recall"):
            down = value["downstairs"][metric]
            up = value["upstairs"][metric]
            value[f"down_over_up_{metric}"] = ratio(down, up) if down is not None else None
    results["purpose"] = split["purpose"]
    results["eval_device"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    cases = []
    scene = "stairs" if args.smoke else "downstairs"
    for i, row in enumerate(rows):
        if row["scene"] == scene:
            first, second = (stair_metrics(matrices[i])["stairs_as_road"] for matrices in per_model)
            if first is not None and second is not None:
                cases.append((i, first, second))
    panels = {}
    if cases:
        for case, label in (
            (max(cases, key=lambda x: x[1] - x[2]), "Largest A - B stair-to-road gap"),
            (min(cases, key=lambda x: x[1] - x[2]), "Smallest A - B stair-to-road gap"),
            (max(cases, key=lambda x: x[2]), "Largest remaining B stair-to-road rate"),
        ):
            i, first, second = case
            panels.setdefault(i, f"{label}: {first:.1%} / {second:.1%}")
    for i, row in enumerate(rows):
        if row["scene"] == "upstairs":
            panels[i] = "Upward example"
            break
    for i, row in enumerate(rows):
        if row["scene"] == "road":
            panels[i] = "Ordinary road"
            break
    draw_cases(args.output / "comparisons.jpg", list(panels.items()), rows, data=args.data,
               models=models, selections=selections, device=device, cfg=cfg)
    write_json(args.output / "metrics.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "compare"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--beihang", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--variant", choices=("A", "B"))
    parser.add_argument("--edge-weight", type=float, help="Override B's edge loss weight for a validation trial")
    parser.add_argument("--a", type=Path)
    parser.add_argument("--b", type=Path)
    args = parser.parse_args()
    cfg = read_json(args.config)
    purpose = "beihang_ab" if args.beihang else "public_smoke" if args.smoke else "field_experiment"
    if (args.smoke and args.beihang) or cfg["purpose"] != purpose:
        parser.error("Config purpose does not match requested experiment")
    if args.action == "train" and (not args.variant or not args.pretrained):
        parser.error("train needs --variant and --pretrained")
    if args.edge_weight is not None and (args.action != "train" or args.variant != "B"
                                         or not 0 < args.edge_weight < float("inf")):
        parser.error("--edge-weight requires a positive finite value and train --variant B")
    if args.action == "compare" and (not args.a or not args.b):
        parser.error("compare needs --a and --b")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    cv2.setNumThreads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if device.type == "cuda" and cfg["bf16"] and not torch.cuda.is_bf16_supported():
        parser.error("GPU does not support bfloat16")
    split = load_split(args.split, args.data, args.smoke,
                       ("train", "val") if args.action == "train" else ("test",), beihang=args.beihang)
    if args.action == "train":
        train(args, split, cfg, device)
    else:
        compare(args, split, cfg, device)


if __name__ == "__main__":
    main()
