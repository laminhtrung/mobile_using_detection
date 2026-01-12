#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video -> YOLOv11l person detection -> crop persons (+20% padding) -> save crops
- Processes every Nth frame (default: 3)
- Creates one subfolder per input video under --out
- File naming:
    {video_stem}/{parent_folder}_{video_stem}_frame_{frame_idx:06d}_crop_{k}.jpg
"""

import argparse, sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser("Crop persons from videos with YOLOv11l")
    ap.add_argument("--videos", required=True,
                    help="Path to a .mp4 file OR a folder containing .mp4 files")
    ap.add_argument("--out", required=True, help="Output directory for crops")
    ap.add_argument("--weights", default="yolov11l.pt", help="Path to YOLOv11l weights")
    ap.add_argument("--device", default="0", help="Device for inference (e.g., '0', 'cpu')")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--every", type=int, default=3, help="Process every Nth frame (1=every frame)")
    ap.add_argument("--pad", type=float, default=0.20, help="Padding ratio around person box (0.2 = 20%)")
    ap.add_argument("--class-id", type=int, default=0, help="Class id to keep (0 = person for COCO)")
    ap.add_argument("--min_side", type=int, default=20, help="Skip crops smaller than this (pixels)")
    return ap.parse_args()


def list_mp4s(videos_path: Path):
    if videos_path.is_file() and videos_path.suffix.lower() == ".mp4":
        return [videos_path]
    if videos_path.is_dir():
        return sorted([p for p in videos_path.rglob("*.mp4")])
    raise FileNotFoundError(f"'{videos_path}' is not a .mp4 file or a directory containing .mp4")


def expand_box(xyxy, pad, w, h):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    # expand by pad ratio
    bw2, bh2 = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    nx1 = max(0, int(round(cx - bw2 / 2.0)))
    ny1 = max(0, int(round(cy - bh2 / 2.0)))
    nx2 = min(w - 1, int(round(cx + bw2 / 2.0)))
    ny2 = min(h - 1, int(round(cy + bh2 / 2.0)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [nx1, ny1, nx2, ny2]


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {args.weights} (device={args.device})")
    model = YOLO(args.weights)

    videos_path = Path(args.videos)
    video_files = list_mp4s(videos_path)
    if not video_files:
        print("[WARN] No .mp4 files found.", file=sys.stderr)
        sys.exit(1)

    for vid_path in video_files:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {vid_path}", file=sys.stderr)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_out_dir = out_dir / vid_path.stem
        vid_out_dir.mkdir(parents=True, exist_ok=True)

        parent_name = vid_path.parent.stem or "root"
        print(f"[INFO] Processing: {vid_path} -> {vid_out_dir} | parent='{parent_name}'")

        frame_idx = 0
        saved_count = 0
        pbar = tqdm(total=total_frames if total_frames > 0 else None,
                    desc=f"{vid_path.name}", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            pbar.update(1)

            # Process every Nth frame: 1, 1+every, 1+2*every, ...
            if args.every > 1 and (frame_idx % args.every != 1):
                continue

            h, w = frame.shape[:2]
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                max_det=300
            )

            if not results:
                continue
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            persons = [(i, b) for i, (c, b) in enumerate(zip(cls, xyxy)) if c == args.class_id]
            if not persons:
                continue

            k = 0
            for _, box in persons:
                ex = expand_box(box, args.pad, w, h)
                if ex is None:
                    continue
                x1, y1, x2, y2 = ex
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                if min(ch, cw) < args.min_side:
                    continue

                # ---- tên file có thêm folder cha + tên video ----
                save_name = f"{vid_path.stem}_frame_{frame_idx:06d}_crop_{k}.jpg"
                save_path = vid_out_dir / save_name
                cv2.imwrite(str(save_path), crop)
                k += 1
                saved_count += 1

        pbar.close()
        cap.release()
        print(f"[INFO] Done {vid_path.name}: saved {saved_count} crops to {vid_out_dir}")

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
