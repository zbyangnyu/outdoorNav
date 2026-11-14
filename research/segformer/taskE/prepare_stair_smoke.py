"""Prepare a small, fixed public ADE20K workflow check from the original ZIP."""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

LUT = np.zeros(256, np.uint8)
LUT[0] = LUT[4] = 255
LUT[np.array([6, 11, 52, 54]) + 1] = 1
LUT[np.array([9, 13, 29, 46, 91, 94]) + 1] = 2
LUT[np.array([53, 59, 121]) + 1] = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    (root / "img").mkdir(parents=True, exist_ok=True)
    (root / "gt").mkdir(exist_ok=True)
    split = {"purpose": "public_smoke", "train": [], "val": [], "test": []}
    with ZipFile(args.archive) as archive:
        for source, partitions in (("training", ("train",)), ("validation", ("val", "test"))):
            prefix = f"ADEChallengeData2016/annotations/{source}/"
            paths = sorted(p for p in archive.namelist() if p.startswith(prefix) and p.endswith(".png"))
            buckets = {"stairs": [], "road": []}
            for path in paths:
                n = Path(path).stem
                raw = np.asarray(Image.open(io.BytesIO(archive.read(path))))
                ids = LUT[raw]
                if (ids == 3).mean() >= .06:
                    scene = "stairs"
                elif (ids == 3).sum() == 0 and (raw == 3).mean() >= .02 and (ids == 1).mean() >= .20:
                    scene = "road"
                else:
                    continue
                if len(buckets[scene]) == 6:
                    continue
                image = f"img/{n}.jpg"
                mask = f"gt/{n}.png"
                (root / image).write_bytes(archive.read(f"ADEChallengeData2016/images/{source}/{n}.jpg"))
                Image.fromarray(ids).save(root / mask)
                buckets[scene].append(dict(n=n, image=image, mask=mask, src="ade", label_source="human",
                                           scene=scene, sequence=n, stair_id=n if scene == "stairs" else ""))
                if all(len(rows) == 6 for rows in buckets.values()):
                    break
            for index, part in enumerate(partitions):
                count = 6 // len(partitions)
                for rows in buckets.values():
                    if len(rows) != 6:
                        raise ValueError(f"Insufficient ADE examples in {source}")
                    split[part].extend(rows[index * count:(index + 1) * count])
    (root / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print({part: len(split[part]) for part in ("train", "val", "test")})


if __name__ == "__main__":
    main()
