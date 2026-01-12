#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → YOLO label converter
--------------------------
Input CSV columns (header required):
filename,width,height,xmin,ymin,xmax,ymax,class

For each unique filename, this script writes a YOLO label file: <stem>.txt
Each line: <class_id> <xc> <yc> <w> <h>   (all normalized to [0,1])

Usage:
  python csv_to_yolo.py --csv annotations.csv --out labels/ --classes-out classes.txt
  # Optional: provide a fixed class mapping via JSON like {"Comp":0, "Phone":1}
  python csv_to_yolo.py --csv annotations.csv --out labels/ --class-map '{"Comp":0}'
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import csv

def yolo_from_xyxy(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_w: int, img_h: int
) -> Tuple[float, float, float, float] | None:
    # Clip to image bounds for safety
    xmin = max(0.0, min(float(xmin), img_w - 1))
    xmax = max(0.0, min(float(xmax), img_w - 1))
    ymin = max(0.0, min(float(ymin), img_h - 1))
    ymax = max(0.0, min(float(ymax), img_h - 1))

    # Ensure proper ordering
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None

    xc = xmin + w / 2.0
    yc = ymin + h / 2.0
    # Normalize
    return xc / img_w, yc / img_h, w / img_w, h / img_h

def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = ["filename","width","height","xmin","ymin","xmax","ymax","class"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {reader.fieldnames}")
        rows.extend(reader)
    return rows

def build_class_map(rows: List[Dict[str, str]], fixed_map: Dict[str,int]|None=None) -> Dict[str,int]:
    if fixed_map:
        return fixed_map
    classes = sorted(list({r["class"] for r in rows}))
    return {c: i for i, c in enumerate(classes)}

def write_classes_txt(classes_map: Dict[str,int], out_path: Path):
    inv = [None] * (max(classes_map.values()) + 1)
    for name, idx in classes_map.items():
        inv[idx] = name
    with open(out_path, "w", encoding="utf-8") as f:
        for name in inv:
            f.write(f"{name}\n")

def fmt(x: float) -> str:
    x = min(max(x, 0.0), 1.0)
    s = f"{x:.6f}"
    return s.rstrip('0').rstrip('.') if '.' in s else s

def group_by_filename(rows: List[Dict[str,str]]) -> Dict[str, List[Dict[str,str]]]:
    groups: Dict[str, List[Dict[str,str]]] = {}
    for r in rows:
        groups.setdefault(r["filename"], []).append(r)
    return groups

def main():
    ap = argparse.ArgumentParser(description="Convert CSV annotations to YOLO label files")
    ap.add_argument("--csv", required=True, type=Path, help="Path to input CSV (with header)")
    ap.add_argument("--out", required=True, type=Path, help="Output folder for YOLO .txt labels")
    ap.add_argument("--classes-out", type=Path, default=None, help="Path to write classes.txt (optional)")
    ap.add_argument("--class-map", type=str, default=None,
                    help='JSON string for fixed class mapping, e.g. {"Comp":0,"Phone":1}')
    args = ap.parse_args()

    rows = load_rows(args.csv)

    fixed_map = None
    if args.class_map:
        fixed_map = json.loads(args.class_map)
        for k,v in list(fixed_map.items()):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"class id for '{k}' must be a non-negative int")

    cls_map = build_class_map(rows, fixed_map)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.classes_out:
        args.classes_out.parent.mkdir(parents=True, exist_ok=True)
        write_classes_txt(cls_map, args.classes_out)

    groups = group_by_filename(rows)
    total_boxes = 0
    skipped = 0

    for fname, items in groups.items():
        stem = Path(fname).stem
        out_label = args.out / f"{stem}.txt"
        lines = []
        for r in items:
            try:
                img_w = int(float(r["width"]))
                img_h = int(float(r["height"]))
                xmin = float(r["xmin"]); ymin = float(r["ymin"])
                xmax = float(r["xmax"]); ymax = float(r["ymax"])
                cls_name = r["class"]
            except Exception:
                skipped += 1
                continue

            y = yolo_from_xyxy(xmin, ymin, xmax, ymax, img_w, img_h)
            if y is None:
                skipped += 1
                continue
            xc, yc, w, h = y
            cid = cls_map[cls_name]
            lines.append(f"{cid} {fmt(xc)} {fmt(yc)} {fmt(w)} {fmt(h)}")
            total_boxes += 1

        with open(out_label, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"✅ Done. Wrote {len(groups)} label files to: {args.out}")
    print(f"   Total boxes: {total_boxes}; Skipped: {skipped}")
    print("   Class mapping:", cls_map)
    if args.classes_out:
        print(f"   classes.txt written to: {args.classes_out}")

if __name__ == "__main__":
    main()
