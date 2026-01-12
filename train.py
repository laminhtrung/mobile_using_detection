#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data.yaml")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", type=str, default="runs/train")
    ap.add_argument("--name", type=str, default="exp11s")
    ap.add_argument("--optimizer", type=str, default="auto",
                    choices=["auto","SGD","Adam","AdamW","NAdam","RMSProp"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--overrides", type=str, default=None,
                    help="Path to a YAML of training overrides (lr0, lrf, hsv_h, ...).")
    ap.add_argument("--cos_lr", action="store_true")
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--weights", type=str, default="yolo11s.pt",
                    help="Pretrained .pt (e.g., yolo11n.pt, yolo11s.pt).")
    ap.add_argument("--scratch", action="store_true",
                    help="Train from scratch: use model YAML and pretrained=False")

    # ---- NEW: auto name ----
    ap.add_argument("--auto_name", action="store_true",
                    help="Tự động đặt tên kiểu <name_prefix>_<N> (ví dụ train_7).")
    ap.add_argument("--name_prefix", type=str, default="train",
                    help="Tiền tố khi auto name (mặc định: 'train').")

    # ---- NEW: nạp hyperparameters từ YAML ----
    ap.add_argument("--hyp", type=str, default="/root/trunglm8/mobile_phone_detection/config/hyp.yaml",
                    help="Đường dẫn YAML chứa hyperparameters (lr0, lrf, momentum, weight_decay, mosaic, hsv_h, ...).")
    ap.add_argument("--patience", type=int, default = 0, help="Early stopping patience (0 = tắt)")
    return ap.parse_args()

def next_run_name(project_dir: str, prefix: str) -> str:
    """
    Quét các thư mục trong project_dir có dạng '<prefix>_<số>' và trả về tên tiếp theo.
    Nếu project_dir chưa tồn tại hoặc chưa có run nào: trả về '<prefix>_1'.
    """
    p = Path(project_dir)
    if not p.exists():
        return f"{prefix}_1"

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_idx = 0
    for d in p.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    pass
    return f"{prefix}_{max_idx + 1}"

def main():
    args = parse_args()

    # Auto name nếu cần
    if args.auto_name and not args.resume:
        Path(args.project).mkdir(parents=True, exist_ok=True)
        args.name = next_run_name(args.project, args.name_prefix)

    # Chọn pretrained hay scratch
    if args.scratch:
        arch_yaml = args.weights.replace(".pt", ".yaml") if args.weights.endswith(".pt") else args.weights
        model = YOLO(arch_yaml)
        pretrained = False
    else:
        model = YOLO(args.weights)
        pretrained = True

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        resume=args.resume,
        freeze=args.freeze,
        cos_lr=args.cos_lr,
        patience=args.patience,  
        verbose=True,
        amp=True,
        exist_ok=True,
        plots=True,
        pretrained=pretrained,
    )

    # === NEW: Ưu tiên --hyp; nếu không có thì dùng --overrides ===
    cfg_path = args.hyp or args.overrides
    if cfg_path:
        cfg_path = str(Path(cfg_path).resolve())
        if not Path(cfg_path).exists():
            raise FileNotFoundError(f"[ERR] Không tìm thấy file YAML: {cfg_path}")
        train_kwargs["cfg"] = cfg_path
        print(f"[INFO] Using cfg: {cfg_path}")

    results = model.train(**train_kwargs)

    if args.val:
        model.val(project=args.project, name=f"{args.name}_val", plots=True, save_json=True)

    print("[OK] Done:", f"{args.project}/{args.name}")

if __name__ == "__main__":
    main()