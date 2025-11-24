#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_catalog.py
----------------
Create per-split CSVs mapping each chip image to its damage label.

Inputs:
  - chips_post/<split>/*.png      (chips created by make_chips.py)
  - chips_post/<split>/*.json     (sidecar metadata for each chip)

Output:
  - chips_post/<split>/catalog.csv

Columns:
  image_path,label_int,label_text,tile_id,event_id,stem,idx
"""

import csv
import json
from pathlib import Path

ROOT_CHIPS = Path("chips_post")


def discover_splits(root_chips: Path):
    """Find available splits (tier1, tier3, test, etc.)."""
    return sorted([d.name for d in root_chips.iterdir() if d.is_dir()])


def build_catalog_for_split(split: str):
    """Read all chip JSONs and write a single catalog CSV."""
    split_dir = ROOT_CHIPS / split
    if not split_dir.exists():
        print(f"[{split}] missing dir {split_dir}")
        return

    sidecars = sorted(split_dir.glob("*.json"))
    if not sidecars:
        print(f"[{split}] no sidecar JSONs found (did you run make_chips.py?)")
        return

    rows = []
    for sc in sidecars:
        try:
            with open(sc, "r") as f:
                meta = json.load(f)
            img = meta.get("image_path")
            label_int = meta.get("label_int")
            label_text = meta.get("label_text", "unknown")
            event_id = meta.get("event_id", "")
            tile_id = meta.get("tile_id", "")
            stem = meta.get("stem", "")
            idx = meta.get("idx", 0)

            # Skip sidecar files that don't have an image (safety check)
            if not img or not sc.with_suffix(".png").exists():
                continue

            rows.append([
                img, label_int, label_text, tile_id, event_id, stem, idx
            ])
        except Exception:
            continue

    if not rows:
        print(f"[{split}] found 0 labeled chips")
        return

    out_csv = split_dir / "catalog.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label_int", "label_text",
                    "tile_id", "event_id", "stem", "idx"])
        w.writerows(rows)

    # Quick summary
    counts = {}
    for _, label_int, _, _, _, _, _ in rows:
        counts[label_int] = counts.get(label_int, 0) + 1

    print(f"[{split}] wrote {len(rows)} rows -> {out_csv}")
    print(f"[{split}] class counts: " +
          ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))


def main():
    splits = discover_splits(ROOT_CHIPS)
    if not splits:
        print("No splits found under chips_post/")
        return
    for split in splits:
        build_catalog_for_split(split)


if __name__ == "__main__":
    main()