#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import shutil

ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_labels(txt_path: Path) -> bool:
    if not txt_path.exists():
        return False
    content = txt_path.read_text(encoding="utf-8").strip()
    return content != ""


def main():
    ap = argparse.ArgumentParser("Copy only images with labels + label files")
    ap.add_argument("--src", required=True, help="Folder chứa ảnh YOLO")
    ap.add_argument("--labels", required=True, help="Folder chứa nhãn YOLO")
    ap.add_argument("--dst", required=True, help="Folder đích để copy")
    ap.add_argument("--recursive", action="store_true", help="Duyệt đệ quy cả ảnh và nhãn")
    args = ap.parse_args()

    src = Path(args.src)
    lbl = Path(args.labels)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = src.rglob("*") if args.recursive else src.glob("*")
    img_files = [p for p in files if p.suffix.lower() in ALLOWED_IMG]

    moved = 0
    skipped = 0

    for img in img_files:
        try:
            rel = img.relative_to(src)          # đường dẫn tương đối
            txt = (lbl / rel).with_suffix(".txt")  # nhãn ở vị trí song song
        except ValueError:
            txt = lbl / (img.stem + ".txt")     # fallback nếu không nằm trong src

        if has_labels(txt):
            # tạo thư mục đích song song
            out_img = dst / rel if args.recursive else dst / img.name
            out_txt = out_img.with_suffix(".txt")
            out_img.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(img), str(out_img))
            if txt.exists():
                shutil.move(str(txt), str(out_txt))
            moved += 1
        else:
            skipped += 1

    print(f"[INFO] Copied {moved} images+labels to {dst}")
    print(f"[INFO] Skipped {skipped} images (no labels)")


if __name__ == "__main__":
    main()
