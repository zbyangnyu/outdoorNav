"""Freeze reviewed Lynx split assignments, optionally adding Task E public training data."""
import argparse
import csv
import json
from pathlib import Path

from stair_split import PARTS, load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="CSV of human-annotated Lynx frames")
    parser.add_argument("--data", type=Path, required=True, help="Common root for image and mask paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-split", type=Path, help="Historical-format Task E public split.json")
    args = parser.parse_args()
    split = {"purpose": "field_experiment", **{part: [] for part in PARTS}}
    with args.annotations.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            part = row.pop("part")
            split[part].append({**row, "src": "lynx", "label_source": "human"})
    if args.public_split:
        root = args.public_split.parent.resolve().relative_to(args.data.resolve())
        public = json.loads(args.public_split.read_text(encoding="utf-8"))
        for row in public["train"]:
            n, src, kind = (row[key] for key in ("n", "src", "kind"))
            scene = "stairs" if src == "ade" and kind == "stairs" else "mixed"
            split["train"].append({
                "n": n, "image": (root / "img" / f"{n}.jpg").as_posix(),
                "mask": (root / "gt" / f"{n}.png").as_posix(),
                "src": src, "scene": scene,
                "sequence": f"{src}:{n if src == 'ade' else kind}",
                "stair_id": f"ade:{n}" if scene == "stairs" else "",
                "label_source": "dataset_pseudo" if src == "itlp" else "human",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(split, file, indent=2)
        file.write("\n")
    try:
        load_split(args.output, args.data, smoke=False, check_parts=PARTS)
    except Exception:
        args.output.unlink()
        raise
    print({part: len(split[part]) for part in PARTS})


if __name__ == "__main__":
    main()
