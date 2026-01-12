#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def main():
    ap = argparse.ArgumentParser("Đếm số ảnh không có label YOLO .txt")
    ap.add_argument("--images", required=True, help="Thư mục chứa ảnh")
    ap.add_argument("--labels", required=True, help="Thư mục chứa nhãn .txt")
    ap.add_argument("--recursive", action="store_true", help="Duyệt ảnh trong subfolders")
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)

    if args.recursive:
        images = [p for p in img_dir.rglob("*") if p.suffix.lower() in ALLOWED_IMG]
    else:
        images = [p for p in img_dir.glob("*") if p.suffix.lower() in ALLOWED_IMG]

    no_label = []
    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists() or lbl_path.stat().st_size == 0:  # không tồn tại hoặc rỗng
            no_label.append(img_path)

    print(f"Tổng số ảnh: {len(images)}")
    print(f"Số ảnh không có label: {len(no_label)}")
    if no_label:
        print("\nDanh sách ảnh thiếu label:")
        for p in no_label[:50]:   # in thử tối đa 50 ảnh đầu
            print("-", p)

if __name__ == "__main__":
    main()
