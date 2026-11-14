"""Build and visualize the reviewed Beihang RGB-only stair benchmark."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from PIL import Image, ImageDraw
import numpy as np

from prepare_stair_smoke import LUT
from stair_split import PARTS, load_split

# Conservatively keep visually similar contiguous runs and matching materials together.
# The publication supplies no physical stair IDs. These are reviewed estimates of scene groups.
PARK_GROUPS = (
    ("train", "early_granite", 0, 11),
    ("train", "blue_slate", 12, 23),
    ("test", "grass_brick_path", 24, 34),
    ("train", "curved_and_leaves", 35, 54),
    ("val", "moss_brick_path", 55, 60),
    ("test", "building_landings", 61, 94),
    ("train", "garden_granite", 95, 189),
    ("test", "ornate_stone_complex", 190, 238),
    ("val", "beige_stone_complex", 239, 266),
    ("test", "rough_stone_steps", 267, 275),
    ("val", "beige_stone_complex", 276, 355),
)
EXAMPLES = (
    ("park28", "Top view | path between grass banks"),
    ("park55", "Top view | weathered stairs and brick landing"),
    ("park272", "Top view | dappled sunlight on rough stone"),
    ("park36", "Upward view | visible vertical risers"),
)


def load_image(archive, member):
    with Image.open(io.BytesIO(archive.read(member))) as im:
        return im.copy()


def beihang_rows(archive):
    members = {}
    for file in archive.namelist():
        path = PurePosixPath(file)
        if len(path.parts) < 4 or path.suffix.lower() not in {".jpg", ".png", ".txt"}:
            continue
        part, kind = path.parts[-3:-1]
        members[part, kind, path.stem] = file

    originals = {}
    frame_counts = Counter()
    triples = 0
    for (part, kind, name), image in sorted(members.items()):
        if kind != "images" or part not in ("train", "val"):
            continue
        mask = members[part, "segmentations", name]
        label = members[part, "labels", name]
        triples += 1
        coords = [tuple(map(int, line.split())) for line in archive.read(label).decode("utf-8-sig").splitlines()
                  if line.strip()]
        ids = np.asarray(load_image(archive, mask))
        if not set(np.unique(ids)) <= {0, 128, 255} or ids.shape != (512, 512):
            raise ValueError(f"Unexpected mask: {mask}")
        risers = bool((ids == 128).any())
        concave = any(values[0] == 1 for values in coords)
        if risers and concave:
            scene = "upstairs"
        elif not risers and not concave and len(coords) >= 2:
            scene = "downstairs"
        else:
            scene = "unclear"
        frame_counts[scene] += 1
        match = re.fullmatch(r"(park(\d+))(?:_[lr])?", name)
        if match and (match[1] not in originals or name == match[1]):
            originals[match[1]] = (int(match[2]), image, mask, coords, scene)
    return originals, dict(frame_counts), triples


def make_row(n, prefix, scene, group, *, edge=False):
    row = dict(n=n, image=f"img/{n}.jpg", mask=f"gt/{n}.png", src=prefix, scene=scene,
               label_source="human", sequence=group, stair_id=group if prefix == "beihang" else "")
    if edge:
        row["edge"] = f"edges/{n}.png"
    return row


def select_park(archive, rows, split, out):
    by_group = {}
    for part, group, first, last in PARK_GROUPS:
        sides = by_group.setdefault((part, group), {"upstairs": [], "downstairs": []})
        for key, (number, _, _, _, scene) in sorted(rows.items(), key=lambda pair: pair[1][0]):
            if first <= number <= last and scene in sides:
                sides[scene].append(key)

    chosen_counts = {}
    for (part, group), sides in by_group.items():
        count = min(map(len, sides.values()))
        chosen_counts[group] = {"part": part, "per_direction": count}
        for scene, names in sides.items():
            # Keep the full sequence span when one direction has more examples.
            for index in np.linspace(0, len(names) - 1, num=count).round().astype(int):
                key = names[index]
                _, image, mask, coords, _ = rows[key]
                n = f"beihang_{key}"
                row = make_row(n, "beihang", scene, group, edge=True)
                rgb = load_image(archive, image).convert("RGB")
                ids = np.asarray(load_image(archive, mask))
                line_map = Image.new("L", rgb.size)
                draw = ImageDraw.Draw(line_map)
                for _, x1, y1, x2, y2 in coords:
                    draw.line((x1, y1, x2, y2), fill=255, width=1)
                # Remove publisher's square letterbox using black row/column coverage.
                active = (np.asarray(rgb).max(axis=2) > 12)
                valid_y = np.flatnonzero((active | (ids != 0)).mean(axis=1) > .07)
                valid_x = np.flatnonzero((active | (ids != 0)).mean(axis=0) > .07)
                x1, x2, y1, y2 = valid_x[0], valid_x[-1] + 1, valid_y[0], valid_y[-1] + 1
                rgb.crop((x1, y1, x2, y2)).save(out / row["image"], quality=96)
                Image.fromarray(np.where(ids[y1:y2, x1:x2], 3, 0).astype(np.uint8)).save(out / row["mask"])
                line_map.crop((x1, y1, x2, y2)).save(out / row["edge"])
                split[part].append(row)
    return chosen_counts


def add_ade(archive, split, out):
    for source in ("training", "validation"):
        limits = {"road": 240, "mixed": 120, "soft": 80} if source == "training" else {"road": 80}
        counts = Counter()
        prefix = f"ADEChallengeData2016/annotations/{source}/"
        names = sorted(p for p in archive.namelist() if p.startswith(prefix) and p.endswith(".png"))
        # Deterministic shuffled candidates avoid collecting neighboring ID ranges.
        rng = np.random.default_rng(20261115)
        for i in rng.permutation(len(names)):
            p = names[i]
            raw = np.asarray(load_image(archive, p))
            ids = LUT[raw]
            if (ids == 3).any() or (raw == 3).mean() < .02:
                continue
            road, soft = (ids == 1).mean(), (ids == 2).mean()
            scene = "road" if road >= .2 and soft < .15 else "mixed" if road >= .05 and soft >= .05 else "soft"
            if scene == "soft" and soft < .20:
                continue
            if counts[scene] >= limits.get(scene, 0):
                continue
            part = "train" if source == "training" else "val" if counts[scene] < 40 else "test"
            counts[scene] += 1
            n = PurePosixPath(p).stem
            row = make_row(n, "ade", "road" if scene == "road" else "mixed", n)
            (out / row["image"]).write_bytes(archive.read(f"ADEChallengeData2016/images/{source}/{n}.jpg"))
            Image.fromarray(ids).save(out / row["mask"])
            split[part].append(row)
            if counts == limits:
                break
        if counts != limits:
            raise ValueError(f"Insufficient ADE sample counts: {counts}")


def examples(archive, rows, output):
    tile = 288
    canvas = Image.new("RGB", (tile * 3, 27 + len(EXAMPLES) * (tile + 22)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(("Original RGB", "Published segmentation", "Published stair edges")):
        draw.text((col * tile + 7, 7), title, fill="black")
    for i, (name, desc) in enumerate(EXAMPLES):
        _, image, mask, coords, _ = rows[name]
        rgb = load_image(archive, image).convert("RGB")
        source_mask = np.asarray(load_image(archive, mask))
        mask_rgb = np.zeros((512, 512, 3), np.uint8)
        mask_rgb[source_mask == 128] = (245, 163, 68)
        mask_rgb[source_mask == 255] = (88, 203, 239)
        overlay = rgb.copy()
        lines = ImageDraw.Draw(overlay)
        for kind, x1, y1, x2, y2 in coords:
            lines.line((x1, y1, x2, y2), fill="#66ff66" if kind == 0 else "#ff9f30", width=4)
        y = 27 + i * (tile + 22)
        for column, im in enumerate((rgb, Image.fromarray(mask_rgb), overlay)):
            canvas.paste(im.resize((tile, tile), Image.Resampling.NEAREST), (column * tile, y))
        draw.text((6, y + tile + 4), f"{name} | {desc}", fill="black")
    canvas.save(output, quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beihang", type=Path, required=True)
    parser.add_argument("--ade", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    for path in ("img", "gt", "edges"):
        (out / path).mkdir(parents=True, exist_ok=True)
    split = {"purpose": "beihang_ab", **{part: [] for part in PARTS}}
    with ZipFile(args.beihang) as archive:
        rows, frame_counts, triples = beihang_rows(archive)
        groups = select_park(archive, rows, split, out)
        examples(archive, rows, out / "annotation_examples.jpg")
    with ZipFile(args.ade) as archive:
        add_ade(archive, split, out)
    path = out / "split.json"
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    load_split(path, out, smoke=False, check_parts=PARTS, beihang=True)
    audit = {"matched_rgb_masks_edges": triples, "original_direction_candidates": frame_counts,
             "park_deduplicated_candidates": dict(Counter(values[-1] for values in rows.values())),
             "balanced_groups": groups,
             "split": {part: dict(Counter(row["scene"] for row in split[part])) for part in PARTS}}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
