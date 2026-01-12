#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

from ultralytics import YOLO
import numpy as np

try:
    from tqdm import tqdm
    USE_TQDM = True
except Exception:
    USE_TQDM = False

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(source: Path, recursive: bool = False) -> List[Path]:
    if source.is_file():
        return [source]
    # Quét tất cả rồi lọc theo suffix.lower() để không bỏ sót .JPG, .PNG
    it = source.rglob("*") if recursive else source.glob("*")
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_classes_arg(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [c.strip().lower() for c in arg.split(",") if c.strip()]


def parse_class_ids_arg(arg: Optional[str]) -> Optional[Set[int]]:
    if not arg:
        return None
    ids = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.add(int(tok))
        except ValueError:
            pass
    return ids or None


def save_yolo_txt(label_path: Path, lines: List[str], allow_empty: bool) -> None:
    ensure_dir(label_path.parent)
    if not lines and not allow_empty:
        # Không ghi file rỗng nếu không được phép
        return
    with label_path.open("w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            # Ghi file rỗng theo yêu cầu (để downstream biết ảnh đã được xử lý)
            f.write("")


def build_keep_class_ids(
    model_names: Dict[int, str],
    keep_names: Optional[List[str]],
    keep_ids: Optional[Set[int]],
) -> Optional[Set[int]]:
    if keep_names is None and keep_ids is None:
        return None
    ids = set(keep_ids or [])
    if keep_names:
        name2id = {v.lower(): k for k, v in model_names.items()}
        for n in keep_names:
            if n in name2id:
                ids.add(name2id[n])
    return ids


def relative_label_path(img_path: Path, src_root: Path, out_root: Path) -> Path:
    """
    Giữ cấu trúc thư mục: src_root/sub/a.jpg -> out_root/sub/a.txt
    Nếu không nằm trong src_root, ghi phẳng vào out_root.
    """
    try:
        rel = img_path.relative_to(src_root)
        return (out_root / rel).with_suffix(".txt")
    except ValueError:
        return (out_root / img_path.stem).with_suffix(".txt")


def main():
    ap = argparse.ArgumentParser(description="YOLO11 -> YOLO label exporter (keep images, no empty labels by default)")
    ap.add_argument("--weights", required=True, type=Path,
                    help="Path to YOLO11 .pt (e.g., best.pt)")
    ap.add_argument("--source", required=True, type=Path, help="Image file or folder")
    ap.add_argument("--out", type=Path, default=Path("./labels"), help="Output labels dir (will mirror structure)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--device", type=str, default=None, help="e.g. '0' or 'cpu'")
    ap.add_argument("--classes", type=str, default=None, help="Comma-separated class names to keep (e.g., 'mobile phone')")
    ap.add_argument("--class-ids", type=str, default=None, help="Comma-separated class IDs to keep (e.g., '0,2')")
    ap.add_argument("--recursive", action="store_true", help="Search subdirectories")
    ap.add_argument("--save-conf", action="store_true", help="Write confidence as 6th column")
    ap.add_argument("--save-empty", action="store_true", help="Write empty .txt when no detections")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output .txt already exists")
    ap.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS")
    ap.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    ap.add_argument("--half", action="store_true", help="Use half precision (CUDA only)")
    args = ap.parse_args()

    # Load model
    model = YOLO(str(args.weights))

    # Collect images
    images = list_images(args.source, args.recursive)
    if not images:
        print(f"[WARN] No images found in: {args.source}", file=sys.stderr)
        sys.exit(0)

    # Warm-up single dry run helps on CUDA
    # (Ultralytics sẽ tự tối ưu; có thể bỏ nếu không cần)
    # model.to(args.device)  # Ultralytics tự chọn nếu None
    keep_names = parse_classes_arg(args.classes)
    keep_ids = parse_class_ids_arg(args.class_ids)

    saved_count = 0
    skipped_count = 0
    empty_count = 0
    err_count = 0

    iterator = tqdm(images, desc="Infer") if USE_TQDM and len(images) > 1 else images
    for img_path in iterator:
        try:
            out_txt = relative_label_path(img_path, args.source if args.source.is_dir() else img_path.parent, args.out)
            if args.skip_existing and out_txt.exists():
                skipped_count += 1
                continue

            results = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                agnostic_nms=args.agnostic_nms,
                max_det=args.max_det,
                half=args.half,
                verbose=False
            )

            if not results:
                # Không có kết quả nào trả về
                save_yolo_txt(out_txt, [], allow_empty=args.save_empty)
                empty_count += 1
                continue

            res = results[0]
            names: Dict[int, str] = res.names
            boxes = res.boxes

            if boxes is None or boxes.xywhn is None or len(boxes) == 0:
                # Không phát hiện
                save_yolo_txt(out_txt, [], allow_empty=args.save_empty)
                empty_count += 1
                continue

            xywhn = boxes.xywhn.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()

            # Xác định tập lớp cần giữ (một lần) dựa trên names + keep args
            keep_id_set = build_keep_class_ids(names, keep_names, keep_ids)

            lines = []
            for b, c, s in zip(xywhn, cls, conf):
                if keep_id_set is not None and int(c) not in keep_id_set:
                    continue

                x, y, w, h = [float(np.clip(float(t), 0.0, 1.0)) for t in b.tolist()]
                if args.save_conf:
                    lines.append(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {float(s):.6f}")
                else:
                    lines.append(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            save_yolo_txt(out_txt, lines, allow_empty=args.save_empty)
            if lines:
                saved_count += 1
            else:
                empty_count += 1

        except Exception as e:
            err_count += 1
            print(f"[ERR] {img_path}: {e}", file=sys.stderr)

    print(f"[OK] Finished.")
    print(f"  Saved label files : {saved_count}")
    print(f"  Empty/no detections: {empty_count} (written={args.save_empty})")
    print(f"  Skipped existing  : {skipped_count}")
    print(f"  Errors            : {err_count}")

    # Trả exit code != 0 nếu có lỗi I/O (hữu ích cho CI)
    if err_count > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
