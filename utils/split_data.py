#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple

# ================== CONFIG ==================
DATASET_ROOT = Path("/root/trunglm8/mobile_phone_detection/roboflow/data-1/train")  
IMG_DIR = DATASET_ROOT / "images"              
LBL_DIR = DATASET_ROOT / "labels"              

OUT_ROOT = DATASET_ROOT / "data_26_9"
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Tỉ lệ split
R_TRAIN, R_VAL, R_TEST = 0.8, 0.2, 0.0
assert abs(R_TRAIN + R_VAL + R_TEST - 1.0) < 1e-6
# ============================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

def is_positive(label_path: Path) -> bool:
    """Dương nếu có file .txt và không rỗng."""
    return label_path.exists() and label_path.stat().st_size > 0

def split_indices(n: int, r_train: float, r_val: float, r_test: float) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(round(n * r_train))
    n_val   = int(round(n * r_val))
    n_test  = n - n_train - n_val  # đảm bảo tổng = n
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def copy_with_empty_label_if_missing(img_path: Path, lbl_src_dir: Path, split_dir: Path):
    """
    split_dir: OUT_ROOT / <train|val|test>
    Tạo thư mục images/ và labels/ bên trong split_dir và copy tương ứng.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    ensure_dir(images_dir)
    ensure_dir(labels_dir)

    # copy image
    dst_img = images_dir / img_path.name
    shutil.copy2(img_path, dst_img)

    # label path
    src_lbl = lbl_src_dir / (img_path.stem + ".txt")
    dst_lbl = labels_dir / (img_path.stem + ".txt")

    if src_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)
    else:
        # tạo file rỗng cho ảnh âm
        dst_lbl.touch()

def main():
    random.seed(SEED)

    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Không thấy thư mục ảnh: {IMG_DIR}")
    ensure_dir(LBL_DIR)  # có thể trống

    all_imgs = list_images(IMG_DIR)
    if not all_imgs:
        raise RuntimeError("Không tìm thấy ảnh nào trong thư mục images/.")

    # Phân loại dương/âm dựa vào label
    pos_imgs, neg_imgs = [], []
    for img in all_imgs:
        lbl = LBL_DIR / (img.stem + ".txt")
        (pos_imgs if is_positive(lbl) else neg_imgs).append(img)

    print(f"[INFO] Tổng ảnh: {len(all_imgs)} | Dương: {len(pos_imgs)} | Âm: {len(neg_imgs)}")

    # Stratified split: tách dương/âm rồi ghép
    pos_tr_idx, pos_va_idx, pos_te_idx = split_indices(len(pos_imgs), R_TRAIN, R_VAL, R_TEST)
    neg_tr_idx, neg_va_idx, neg_te_idx = split_indices(len(neg_imgs), R_TRAIN, R_VAL, R_TEST)

    train_imgs = [pos_imgs[i] for i in pos_tr_idx] + [neg_imgs[i] for i in neg_tr_idx]
    val_imgs   = [pos_imgs[i] for i in pos_va_idx] + [neg_imgs[i] for i in neg_va_idx]
    test_imgs  = [pos_imgs[i] for i in pos_te_idx] + [neg_imgs[i] for i in neg_te_idx]

    random.shuffle(train_imgs); random.shuffle(val_imgs); random.shuffle(test_imgs)

    # Tạo thư mục cho mỗi split với cấu trúc yêu cầu
    for split in ["train", "val", "test"]:
        ensure_dir(OUT_ROOT / split / "images")
        ensure_dir(OUT_ROOT / split / "labels")

    # Copy
    for img in train_imgs:
        copy_with_empty_label_if_missing(img, LBL_DIR, OUT_ROOT / "train")
    for img in val_imgs:
        copy_with_empty_label_if_missing(img, LBL_DIR, OUT_ROOT / "val")
    for img in test_imgs:
        copy_with_empty_label_if_missing(img, LBL_DIR, OUT_ROOT / "test")

    # Thống kê cuối
    def count_pos_neg(img_list):
        p = sum(is_positive(LBL_DIR / (im.stem + ".txt")) for im in img_list)
        return p, len(img_list) - p

    p_tr, n_tr = count_pos_neg(train_imgs)
    p_va, n_va = count_pos_neg(val_imgs)
    p_te, n_te = count_pos_neg(test_imgs)

    print(f"[DONE] Lưu tại: {OUT_ROOT}")
    print(f"Train: {len(train_imgs)} (pos {p_tr}, neg {n_tr})")
    print(f"Val  : {len(val_imgs)} (pos {p_va}, neg {n_va})")
    print(f"Test : {len(test_imgs)} (pos {p_te}, neg {n_te})")

if __name__ == "__main__":
    main()
