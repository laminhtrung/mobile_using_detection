#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import yaml
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser("Evaluate YOLO on test split")
    ap.add_argument("--weights", type=str, required=True, help="Path to model .pt (e.g., best.pt)")
    ap.add_argument("--data", type=str, required=True, help="data.yaml với khóa 'test'")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.001, help="Confidence threshold cho NMS/eval")
    ap.add_argument("--iou", type=float, default=0.6, help="IoU threshold cho NMS")
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--project", type=str, default="runs/test")
    ap.add_argument("--name", type=str, default="eval_test")
    ap.add_argument("--plots", action="store_true", help="Lưu PR/F1/P/R curves, confusion matrix, ...")
    ap.add_argument("--out_json", type=str, default="test_metrics.json",
                    help="Tên file JSON chỉ chứa metrics test")
    return ap.parse_args()

def ensure_test_split(data_yaml: Path):
    with open(data_yaml, "r") as f:
        d = yaml.safe_load(f)
    if "test" not in d or not d["test"]:
        print(f"[WARN] data.yaml ({data_yaml}) không có 'test' split. "
              f"Thêm khóa 'test:' để evaluate trên test.")
    return d

def extract_metrics(results) -> dict:
    """
    Lấy các metric chuẩn từ Ultralytics results
    """
    md = results if isinstance(results, dict) else getattr(results, "results_dict", {})
    def pick(*keys):
        for k in keys:
            if k in md:
                return float(md[k])
        return None

    return {
        "precision":  pick("metrics/precision(B)", "metrics/precision", "precision", "mp"),
        "recall":     pick("metrics/recall(B)", "metrics/recall", "recall", "mr"),
        "mAP50":      pick("metrics/mAP50(B)", "metrics/mAP50", "map50", "mAP50"),
        "mAP50-95":   pick("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50-95", "mAP50-95"),
        "fitness":    pick("fitness", "metrics/fitness"),
    }

def main():
    args = parse_args()
    ensure_test_split(Path(args.data))

    model = YOLO(args.weights)

    # chỉ evaluate test, không save_json để tránh predictions từng ảnh
    results = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        plots=args.plots,
        verbose=True
    )

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = extract_metrics(results)
    out_path = run_dir / args.out_json
    out_path.write_text(json.dumps(metrics, indent=2))

    print("\n[SUMMARY METRICS]")
    for k, v in metrics.items():
        print(f"- {k:11s}: {v}")
    print(f"\n[OK] Saved metrics JSON: {out_path}")

if __name__ == "__main__":
    main()
