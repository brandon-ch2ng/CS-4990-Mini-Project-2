#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chips.py
-------------
Generate aligned post-disaster chips (128×128) for xBD/xView2-style data,
including the "features": {"lng_lat": [...], "xy": [...]} schema.

Outputs per split:
  chips_post/<split>/
    ├── *_b00000.png     image chip
    ├── *_b00000.json    metadata (label, shift, etc.)
    ├── manifest.csv     summary table
    └── post_shifts.json per-tile alignment log
"""

import json, csv, re, warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import wkt
from shapely.geometry import shape, Polygon
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
import imageio.v2 as iio

# ---------------- CONFIG ----------------
ROOT = Path("../data/geotiffs")
OUT_DIR = Path("chips_post")
CROP_SIZE = 128
DOWNSAMPLE_FOR_SHIFT = 4
MAX_SHIFT_PX = 20
PAIR_RX = re.compile(r"(.*)_(pre|post)_disaster\.(?:tif|tiff|png)$", re.IGNORECASE)
SPLITS_OVERRIDE = ["hold", "tier1", "tier3", "test"]

DAMAP = {
    "no-damage": 0, "none": 0, "undamaged": 0,
    "minor-damage": 1, "minor_damage": 1, "minor": 1,
    "major-damage": 2, "major_damage": 2, "major": 2,
    "destroyed": 3, "total-destruction": 3
}

# ---------------- HELPERS ----------------
def discover_splits(root: Path):
    return [d.name for d in root.iterdir() if d.is_dir()]

def find_images_dir(split_root: Path):
    if (split_root / "images").exists():
        return split_root / "images"
    subs = list(split_root.rglob("images"))
    return subs[0] if subs else split_root

def read_preview_gray(path: Path, ds_factor=1):
    with rasterio.open(path) as ds:
        arr = ds.read([1,2,3], out_dtype=np.float32, boundless=True, fill_value=0)
        gray = 0.2989*arr[0] + 0.5870*arr[1] + 0.1140*arr[2]
        if ds_factor>1:
            h,w = gray.shape
            gray = resize(gray,(h//ds_factor,w//ds_factor),anti_aliasing=True)
        gray -= gray.mean(); gray /= (gray.std()+1e-6)
        return gray

def estimate_shift(pre_path: Path, post_path: Path)->Tuple[float,float]:
    g_pre = read_preview_gray(pre_path,DOWNSAMPLE_FOR_SHIFT)
    g_post = read_preview_gray(post_path,DOWNSAMPLE_FOR_SHIFT)
    shift,_,_ = phase_cross_correlation(g_pre,g_post,upsample_factor=10)
    dy,dx = shift
    return float(np.clip(dx*DOWNSAMPLE_FOR_SHIFT,-MAX_SHIFT_PX,MAX_SHIFT_PX)), \
           float(np.clip(dy*DOWNSAMPLE_FOR_SHIFT,-MAX_SHIFT_PX,MAX_SHIFT_PX))

def safe_window_centered(col,row,size,w,h):
    half=size//2
    c0=int(round(col-half)); r0=int(round(row-half))
    c0=max(0,min(c0,w-size)); r0=max(0,min(r0,h-size))
    return Window(c0,r0,size,size)

def save_png(path:Path,arr:np.ndarray):
    iio.imwrite(path.as_posix(),arr)

# -------- parse label JSON (supports nested lng_lat/xy + WKT) --------
def load_wkt_features(label_path:Path)->List[Dict[str,Any]]:
    """Parse label files with nested features.lng_lat / xy arrays."""
    try:
        gj=json.load(open(label_path))
    except Exception:
        return []
    feats=[]
    if isinstance(gj,dict) and "features" in gj:
        f=gj["features"]
        if isinstance(f,dict):
            # your schema: {"lng_lat":[...],"xy":[...]}
            feats += f.get("lng_lat",[])
            feats += f.get("xy",[])
        elif isinstance(f,list):
            feats=f
    elif isinstance(gj,list):
        feats=gj
    out=[]
    for ft in feats:
        if not isinstance(ft,dict): continue
        props=ft.get("properties",{})
        if props.get("feature_type")!="building": continue
        dmg_text=str(props.get("subtype","unknown")).lower()
        dmg_int=DAMAP.get(dmg_text,0)
        wkt_str=ft.get("wkt")
        try:
            geom=wkt.loads(wkt_str)
            if not geom.is_valid: continue
            out.append({"geom":geom,"label_int":dmg_int,"label_text":dmg_text})
        except Exception:
            continue
    return out

# ---------------- MAIN LOOP ----------------
def process_split(split:str):
    split_root=ROOT/split
    if not split_root.exists():
        print(f"[{split}] missing split dir");return
    images_dir=find_images_dir(split_root)
    label_dir=split_root/"labels"
    out_split=OUT_DIR/split
    out_split.mkdir(parents=True,exist_ok=True)
    pairs={}
    for p in images_dir.rglob("*_pre_disaster.*"):
        m=PAIR_RX.match(p.name)
        if not m: continue
        stem=m.group(1)
        post=p.with_name(stem+"_post_disaster"+p.suffix)
        if post.exists():
            pairs[stem]={"pre":p,"post":post}
    print(f"[pair_pre_post] {len(pairs)} pairs found")

    shifts,manifest={},[]
    for stem,paths in pairs.items():
        pre,post=paths["pre"],paths["post"]
        dx,dy=0.0,0.0
        try: dx,dy=estimate_shift(pre,post)
        except: pass
        shifts[stem]={"dx":dx,"dy":dy}
        label_path=label_dir/f"{stem}_post_disaster.json"
        if not label_path.exists():
            continue
        feats=load_wkt_features(label_path)
        print(f"[{split}] {stem}: buildings_found={len(feats)}")
        if not feats: continue
        with rasterio.open(post) as ds:
            w,h=ds.width,ds.height
            for i,b in enumerate(feats):
                cx,cy=ds.index(*b["geom"].centroid.xy)
                win=safe_window_centered(cx,cy,CROP_SIZE,w,h)
                chip=ds.read(window=win,out_dtype=np.uint8,boundless=True)
                chip=np.moveaxis(chip,0,2)
                if chip.ndim==2: chip=np.stack([chip]*3,axis=2)
                out_png=out_split/f"{stem}_b{i:05d}.png"
                save_png(out_png,chip)
                sidecar={
                    "image_path":str(out_png),
                    "label_int":b["label_int"],
                    "label_text":b["label_text"],
                    "stem":stem,
                    "idx":i,
                    "shift_dx":dx,"shift_dy":dy,
                    "label_file":str(label_path)
                }
                json.dump(sidecar,open(out_png.with_suffix(".json"),"w"),indent=2)
                manifest.append([str(out_png),b["label_int"],b["label_text"],stem,i,dx,dy])
    json.dump(shifts,open(out_split/"post_shifts.json","w"),indent=2)
    if manifest:
        with open(out_split/"manifest.csv","w",newline="") as f:
            w=csv.writer(f);w.writerow(["image_path","label_int","label_text","stem","idx","dx","dy"]);w.writerows(manifest)
    print(f"[{split}] chips written: {len(manifest)}")

def main():
    OUT_DIR.mkdir(exist_ok=True)
    splits=SPLITS_OVERRIDE or discover_splits(ROOT)
    print(f"Discovered splits: {splits}")
    for s in splits: process_split(s)

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=rasterio.errors.NotGeoreferencedWarning)
    main()