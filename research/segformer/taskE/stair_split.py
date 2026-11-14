"""Validate a fixed split for the stair experiment."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

PARTS = ("train", "val", "test")
SCENES = {"downstairs", "stairs", "road", "no_stairs", "mixed"}
LABELS = {0, 1, 2, 3, 255}


def load_split(path, data, smoke, check_parts):
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    if split["purpose"] != ("public_smoke" if smoke else "field_experiment"):
        raise ValueError("Split purpose does not match the requested experiment")
    seen = {field: {} for field in ("n", "image", "mask", "sequence", "stair_id")}
    for part in PARTS:
        rows = split[part]
        if not rows:
            raise ValueError(f"Empty split: {part}")
        if not smoke:
            field_scenes = {row["scene"] for row in rows if row["src"] == "lynx"}
            if not {"downstairs", "road"} <= field_scenes:
                raise ValueError(f"{part} needs human-labelled Lynx downstairs and road frames")
        for row in rows:
            name = row["n"]
            if row["scene"] not in SCENES:
                raise ValueError(f"{name}: unknown scene {row['scene']}")
            if not row["sequence"] or (row["scene"] in {"stairs", "downstairs"} and not row["stair_id"]):
                raise ValueError(f"{name}: missing sequence or physical stair ID")
            if not smoke and part != "train" and row["src"] != "lynx":
                raise ValueError(f"{name}: formal validation and test require Lynx frames")
            if row["label_source"] not in {"human", "dataset_pseudo"}:
                raise ValueError(f"{name}: unknown annotation source")
            if (row["src"] == "lynx" or part != "train") and row["label_source"] != "human":
                raise ValueError(f"{name}: human annotation required")
            for field, owners in seen.items():
                value = row[field]
                if field in {"image", "mask"}:
                    relative = Path(value)
                    if not value or relative.anchor or ".." in relative.parts:
                        raise ValueError(f"{name}: {field} must be relative to the data root")
                    value = relative.as_posix().casefold()
                if value and value in owners and (
                    field in {"n", "image", "mask"} or owners[value] != part
                ):
                    raise ValueError(f"{name}: reused {field}={value!r} in {part} and {owners[value]}")
                if value:
                    owners[value] = part
            if part in check_parts:
                check_mask(data, row)
    return split


def check_mask(data, row):
    with Image.open(data / row["image"]) as image, Image.open(data / row["mask"]) as mask:
        if image.size != mask.size or mask.mode != "L":
            raise ValueError(f"{row['n']}: mask must be an aligned greyscale ID PNG")
        labels = set(np.unique(np.asarray(mask)))
    if not labels <= LABELS or labels == {255}:
        raise ValueError(f"{row['n']}: invalid or empty annotation")
    scene = row["scene"]
    if scene in {"stairs", "downstairs"} and 3 not in labels:
        raise ValueError(f"{row['n']}: stairs scene has no stairs pixels")
    if scene in {"road", "no_stairs"} and (3 in labels or (scene == "road" and 1 not in labels)):
        raise ValueError(f"{row['n']}: annotation contradicts scene={scene}")
