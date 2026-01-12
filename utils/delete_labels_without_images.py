#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser("Delete label files without corresponding images")
    ap.add_argument("--images", required=True, help="Folder chứa ảnh YOLO")
    ap.add_argument("--labels", required=True, help="Folder chứa nhãn YOLO")
    ap.add_argument("--recursive", action="store_true", help="Duyệt đệ quy")
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)

    # lấy danh sách tên file ảnh (không đuôi)
    img_files = img_dir.rglob("*") if args.recursive else img_dir.glob("*")
    img_stems = {p.stem for p in img_files if p.suffix.lower() in ALLOWED_IMG}

    lbl_files = lbl_dir.rglob("*.txt") if args.recursive else lbl_dir.glob("*.txt")

    deleted = 0
    kept = 0

    for txt in lbl_files:
        if txt.stem not in img_stems:
            txt.unlink()
            print(f"[DEL] {txt}")
            deleted += 1
        else:
            kept += 1

    print(f"[INFO] Deleted {deleted} label(s) without images, kept {kept}")


if __name__ == "__main__":
    main()
