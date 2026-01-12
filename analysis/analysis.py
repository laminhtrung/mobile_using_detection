#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze a YOLO detection dataset (train/val/test).
- Hỗ trợ YOLO txt labels (normalized cx, cy, w, h).
- Tự đọc data.yaml (nếu có) để lấy tên lớp.
- Xuất biểu đồ (PNG), CSV, và report.json kèm cảnh báo.
- TÍNH THÊM: số file txt rỗng (empty_txt_files) ở summary.

Usage:
python analyze_yolo.py \
  --root /path/to/dataset \
  --outdir /path/to/out \
  --img-exts .jpg .png .jpeg .bmp .webp \
  --sample-vis 30
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

ALLOWED_IMG_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
SPLITS = ["train", "val", "test"]

# COCO thresholds theo diện tích pixel tuyệt đối
COCO_SMALL = 32 ** 2
COCO_MEDIUM = 96 ** 2


def read_yaml_if_exists(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception:
            return {}


def find_images(folder: Path, img_exts):
    files = []
    for ext in img_exts:
        files.extend(folder.rglob(f"*{ext.lower()}"))
        files.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def to_xyxy_norm(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def analyze_split(split_root: Path, img_exts, class_names):
    """
    Cấu trúc kỳ vọng trong split_root:
    - images/  (hoặc ảnh nằm rải rác)
    - labels/  (txt YOLO)
    """
    images_dir_candidates = [split_root / "images", split_root]
    labels_dir_candidates = [split_root / "labels", split_root]

    # Tập các thư mục thật sự có *.txt
    label_txt_dirs = set()
    for c in labels_dir_candidates:
        if c.exists() and any(p.suffix.lower() == ".txt" for p in c.rglob("*.txt")):
            label_txt_dirs.add(c)
    if not label_txt_dirs:
        for p in split_root.rglob("labels"):
            if any(x.suffix.lower() == ".txt" for x in p.rglob("*.txt")):
                label_txt_dirs.add(p)

    images = []
    for c in images_dir_candidates:
        if c.exists():
            images += find_images(c, img_exts)
    if not images:
        images = find_images(split_root, img_exts)

    rows = []      # per-box
    img_rows = []  # per-image
    problems = {
        "images_without_labels": [],
        "labels_without_images": [],
        "empty_label_files": [],
        "malformed_lines": [],
        "coords_out_of_range": [],
        "zero_or_tiny_wh": [],
    }

    # Tập toàn bộ txt trong split để lần cuối tìm orphan
    all_txts = set(p for p in split_root.rglob("*.txt"))

    size_cache = {}

    for img_path in tqdm(images, desc=f"[{split_root.name}] scanning images"):
        # Tìm label tương ứng
        found_label = None
        if label_txt_dirs:
            for lbl_dir in label_txt_dirs:
                candidate = lbl_dir / (img_path.stem + ".txt")
                if candidate.exists():
                    found_label = candidate
                    break
        else:
            candidate = img_path.with_suffix(".txt")
            if candidate.exists():
                found_label = candidate

        # Kích thước ảnh
        try:
            if img_path in size_cache:
                W, H = size_cache[img_path]
            else:
                with Image.open(img_path) as im:
                    W, H = im.size
                size_cache[img_path] = (W, H)
        except Exception:
            im = cv2.imread(str(img_path))
            if im is None:
                W = H = None
            else:
                H, W = im.shape[:2]
            size_cache[img_path] = (W, H)

        n_boxes = 0
        if found_label is None or not found_label.exists():
            problems["images_without_labels"].append(str(img_path))
        else:
            raw_lines = []
            try:
                with open(found_label, "r", encoding="utf-8") as f:
                    raw_lines = [ln.rstrip("\n") for ln in f.readlines()]
            except Exception:
                raw_lines = []

            for li, line in enumerate(raw_lines):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    problems["malformed_lines"].append(
                        {"file": str(found_label), "line_idx": li, "text": line}
                    )
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                except Exception:
                    problems["malformed_lines"].append(
                        {"file": str(found_label), "line_idx": li, "text": line}
                    )
                    continue

                out_of_range = not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0)
                zero_tiny = (w <= 0 or h <= 0 or (W and H and (w * W < 1 or h * H < 1)))

                if out_of_range:
                    problems["coords_out_of_range"].append(
                        {"img": str(img_path), "label": str(found_label), "line_idx": li,
                         "values": [cls_id, cx, cy, w, h]}
                    )
                if zero_tiny:
                    problems["zero_or_tiny_wh"].append(
                        {"img": str(img_path), "label": str(found_label), "line_idx": li, "w": w, "h": h}
                    )

                x1, y1, x2, y2 = to_xyxy_norm(cx, cy, w, h)
                abs_w = w * W if (W is not None) else None
                abs_h = h * H if (H is not None) else None
                abs_area = (abs_w * abs_h) if (abs_w is not None and abs_h is not None) else None
                size_bucket = None
                if abs_area is not None:
                    if abs_area < COCO_SMALL:
                        size_bucket = "small"
                    elif abs_area < COCO_MEDIUM:
                        size_bucket = "medium"
                    else:
                        size_bucket = "large"

                rows.append({
                    "split": split_root.name,
                    "image": str(img_path),
                    "label_file": str(found_label),
                    "class_id": cls_id,
                    "class_name": class_names.get(cls_id, str(cls_id)),
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "img_w": W, "img_h": H,
                    "abs_w": abs_w, "abs_h": abs_h, "abs_area": abs_area,
                    "size_bucket": size_bucket,
                    "out_of_range": out_of_range,
                    "zero_or_tiny": zero_tiny
                })
                n_boxes += 1

            if len(raw_lines) == 0:
                problems["empty_label_files"].append(str(found_label))

            if found_label in all_txts:
                all_txts.remove(found_label)

        img_rows.append({
            "split": split_root.name,
            "image": str(img_path),
            "img_w": W, "img_h": H,
            "aspect_ratio": (W / H) if (W and H and H != 0) else None,
            "n_boxes": n_boxes
        })

    # label mồ côi (không ghép được với ảnh)
    for orphan in list(all_txts):
        if orphan.suffix.lower() == ".txt":
            problems["labels_without_images"].append(str(orphan))

    return pd.DataFrame(rows), pd.DataFrame(img_rows), problems


def plot_and_save_hist(series, title, xlabel, out_path, bins=30, logy=False):
    try:
        plt.figure()
        plt.hist(series.dropna(), bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        if logy:
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to plot {title}: {e}")


def barh_counts(counter_dict, title, out_path, top=None):
    try:
        items = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)
        if top:
            items = items[:top]
        labels = [str(k) for k, _ in items]
        vals = [v for _, v in items]
        plt.figure()
        plt.barh(labels, vals)
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to plot barh {title}: {e}")


def summarize(df_boxes, df_images, class_names, problems=None):
    summary = {}

    total_images = len(df_images)
    total_boxes = len(df_boxes)
    summary["total_images"] = total_images
    summary["total_boxes"] = total_boxes

    boxes_by_split = df_boxes.groupby("split")["class_id"].count().to_dict()
    imgs_by_split = df_images.groupby("split")["image"].nunique().to_dict()
    summary["images_by_split"] = {k: int(v) for k, v in imgs_by_split.items()}
    summary["boxes_by_split"] = {k: int(v) for k, v in boxes_by_split.items()}

    cls_counts = df_boxes["class_id"].value_counts().to_dict()
    summary["class_counts"] = {int(k): int(v) for k, v in cls_counts.items()}
    if class_names:
        summary["class_counts_named"] = {class_names.get(int(k), str(k)): int(v) for k, v in cls_counts.items()}

    size_counts = df_boxes["size_bucket"].value_counts(dropna=True).to_dict()
    summary["size_buckets"] = {str(k): int(v) for k, v in size_counts.items()}

    out_of_range = int(df_boxes["out_of_range"].sum()) if "out_of_range" in df_boxes else 0
    zero_tiny = int(df_boxes["zero_or_tiny"].sum()) if "zero_or_tiny" in df_boxes else 0
    summary["out_of_range_boxes"] = out_of_range
    summary["zero_or_tiny_boxes"] = zero_tiny

    # SỐ FILE TXT RỖNG
    empty_txt_files = 0
    if problems and "empty_label_files" in problems:
        # problems có thể là merged (list), nên len() là số file rỗng
        empty_txt_files = len(problems["empty_label_files"])
    summary["empty_txt_files"] = int(empty_txt_files)

    # Thống kê bbox/ảnh
    if "n_boxes" in df_images:
        s = df_images["n_boxes"].describe().to_dict()
        summary["boxes_per_image"] = {k: float(v) for k, v in s.items()}

    # Kích thước ảnh & tỉ lệ khung
    sizes = df_images[["img_w", "img_h"]].dropna()
    if len(sizes):
        summary["image_w_median"] = float(sizes["img_w"].median())
        summary["image_h_median"] = float(sizes["img_h"].median())
    ar = df_images["aspect_ratio"].dropna()
    if len(ar):
        summary["aspect_ratio_median"] = float(np.median(ar))

    # Gợi ý
    hints = []
    if len(cls_counts) >= 2:
        most = max(cls_counts.values())
        least = min(cls_counts.values())
        if least > 0 and most / least >= 10:
            hints.append("Class imbalance severe (>=10x). Cân nhắc oversampling/augmentation có điều kiện hoặc class-weighted loss.")
        elif most / max(least, 1) >= 5:
            hints.append("Class imbalance (>=5x). Nên áp dụng cân bằng dữ liệu.")
    if summary.get("size_buckets", {}).get("small", 0) / max(total_boxes, 1) >= 0.5:
        hints.append("Tỷ lệ small object cao. Cân nhắc tăng img size train, stride nhỏ hơn hoặc tiling.")
    if out_of_range > 0:
        hints.append("Có bbox ngoài [0,1] (normalized). Cần sửa nhãn.")
    if zero_tiny > 0:
        hints.append("Có bbox width/height ~0 hoặc <=1px sau denorm. Kiểm tra lại annotation.")
    if empty_txt_files > 0:
        hints.append("Có file nhãn rỗng. Kiểm tra ảnh tương ứng hoặc xóa nếu là negative không chủ đích.")

    summary["hints"] = hints
    return summary


def save_split_products(outdir: Path, split: str, df_boxes_split, df_images_split, class_names):
    split_dir = outdir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    df_boxes_split.to_csv(split_dir / "boxes.csv", index=False)
    df_images_split.to_csv(split_dir / "images.csv", index=False)

    plot_and_save_hist(df_images_split["n_boxes"], f"{split}: BBoxes per image",
                       "boxes/img", split_dir / "hist_boxes_per_image.png", bins=30, logy=True)
    plot_and_save_hist(df_images_split["aspect_ratio"], f"{split}: Aspect ratio (W/H)",
                       "W/H", split_dir / "hist_aspect_ratio.png", bins=30)

    if "abs_area" in df_boxes_split:
        plot_and_save_hist(df_boxes_split["abs_area"], f"{split}: BBox area (px^2)",
                           "area", split_dir / "hist_bbox_area_px.png", bins=50, logy=True)

    cls_counts = df_boxes_split["class_id"].value_counts().to_dict()
    readable = {class_names.get(int(k), str(k)): v for k, v in cls_counts.items()}
    try:
        items = sorted(readable.items(), key=lambda x: x[1], reverse=True)
        labels = [str(k) for k, _ in items]
        vals = [v for _, v in items]
        plt.figure()
        plt.barh(labels, vals)
        plt.gca().invert_yaxis()
        plt.title(f"{split}: Class distribution")
        plt.tight_layout()
        plt.savefig(split_dir / "class_distribution.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to plot class distribution for {split}: {e}")

    size_counts = df_boxes_split["size_bucket"].value_counts().to_dict()
    try:
        items = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [str(k) for k, _ in items]
        vals = [v for _, v in items]
        plt.figure()
        plt.barh(labels, vals)
        plt.gca().invert_yaxis()
        plt.title(f"{split}: BBox size buckets")
        plt.tight_layout()
        plt.savefig(split_dir / "size_buckets.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to plot size buckets for {split}: {e}")


def visualize_samples(df_boxes, outdir: Path, k=20):
    vis_dir = outdir / "samples_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    imgs = df_boxes["image"].dropna().unique().tolist()
    if not imgs:
        return
    rng = np.random.default_rng(123)
    picks = rng.choice(imgs, size=min(k, len(imgs)), replace=False)

    df_by_img = df_boxes.groupby("image")
    for img_path in tqdm(picks, desc="Rendering samples"):
        try:
            im = cv2.imread(img_path)
            if im is None:
                continue
            H, W = im.shape[:2]
            subset = df_by_img.get_group(img_path)
            for _, r in subset.iterrows():
                x1 = int((r["cx"] - r["w"] / 2) * W)
                y1 = int((r["cy"] - r["h"] / 2) * H)
                x2 = int((r["cx"] + r["w"] / 2) * W)
                y2 = int((r["cy"] + r["h"] / 2) * H)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = str(r.get("class_name", r["class_id"]))
                cv2.putText(im, txt, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            outp = vis_dir / (Path(img_path).stem + "_vis.jpg")
            cv2.imwrite(str(outp), im)
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser("Analyze YOLO dataset (train/val/test)")
    ap.add_argument("--root", required=True, help="Dataset root có train/ val/ test")
    ap.add_argument("--outdir", required=True, help="Folder xuất charts/CSVs/report")
    ap.add_argument("--img-exts", nargs="+", default=ALLOWED_IMG_DEFAULT, help="Phần mở rộng ảnh")
    ap.add_argument("--sample-vis", type=int, default=20, help="Số ảnh vẽ bbox mẫu")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Đọc data.yaml để lấy names (nếu có)
    data_yaml = read_yaml_if_exists(root / "data.yaml")
    class_names = {}
    if "names" in data_yaml and isinstance(data_yaml["names"], (list, tuple)):
        class_names = {i: str(n) for i, n in enumerate(data_yaml["names"])}
    elif "names" in data_yaml and isinstance(data_yaml["names"], dict):
        class_names = {int(k): str(v) for k, v in data_yaml["names"].items()}

    all_boxes = []
    all_images = []
    all_problems = {}

    for split in SPLITS:
        split_root = root / split
        if not split_root.exists():
            print(f"[INFO] Split '{split}' not found at {split_root} — skipping.")
            continue
        df_b, df_i, probs = analyze_split(split_root, args.img_exts, class_names)
        all_boxes.append(df_b)
        all_images.append(df_i)
        all_problems[split] = probs

    if len(all_boxes) == 0:
        print("[ERROR] No splits found or no labels/images detected.")
        return

    df_boxes = pd.concat(all_boxes, ignore_index=True) if len(all_boxes) else pd.DataFrame()
    df_images = pd.concat(all_images, ignore_index=True) if len(all_images) else pd.DataFrame()

    # Lưu theo split
    for split in df_boxes["split"].unique():
        df_b_s = df_boxes[df_boxes["split"] == split]
        df_i_s = df_images[df_images["split"] == split]
        save_split_products(outdir, split, df_b_s, df_i_s, class_names)

    # Lưu tổng
    df_boxes.to_csv(outdir / "all_boxes.csv", index=False)
    df_images.to_csv(outdir / "all_images.csv", index=False)

    plot_and_save_hist(df_images["n_boxes"], "All: BBoxes per image", "boxes/img",
                       outdir / "all_hist_boxes_per_image.png", bins=30, logy=True)
    plot_and_save_hist(df_images["aspect_ratio"], "All: Aspect ratio (W/H)", "W/H",
                       outdir / "all_hist_aspect_ratio.png", bins=30)
    if "abs_area" in df_boxes:
        plot_and_save_hist(df_boxes["abs_area"], "All: BBox area (px^2)", "area",
                           outdir / "all_hist_bbox_area_px.png", bins=60, logy=True)

    # Gộp problems để tính empty_txt_files tổng
    problems_merged = defaultdict(list)
    for sp, pr in all_problems.items():
        for k, v in pr.items():
            problems_merged[k].extend(v)

    summary = summarize(df_boxes, df_images, class_names, problems=problems_merged)

    report = {
        "summary": summary,
        "problems": problems_merged,
        "notes": [
            "Size buckets dùng COCO thresholds: small < 32^2, medium < 96^2, còn lại là large.",
            "Tiny boxes: width/height <= 1px sau khi denormalize theo kích thước ảnh.",
            "Kiểm tra danh sách problems.* để sửa annotation/data."
        ]
    }
    with open(outdir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    crowded = df_images.sort_values("n_boxes", ascending=False).head(50)
    crowded.to_csv(outdir / "top_crowded_images.csv", index=False)

    if args.sample_vis > 0 and len(df_boxes):
        visualize_samples(df_boxes, outdir, k=args.sample_vis)

    print("\n=== DONE ===")
    print(f"- Output directory: {outdir}")
    print("- Key files:")
    print("  • report.json (có empty_txt_files)")
    print("  • all_boxes.csv, all_images.csv")
    print("  • per-split charts & CSVs")
    print("  • *_class_distribution.png, *_size_buckets.png, hist_*.png")
    if args.sample_vis > 0:
        print("  • samples_vis/*.jpg (ảnh gắn bbox để kiểm tra nhanh)")
    print("\nHints:")
    for h in summary.get("hints", []):
        print(" -", h)


if __name__ == "__main__":
    main()
