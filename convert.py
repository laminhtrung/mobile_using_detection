#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_writable_ultralytics_config() -> None:
    # In sandboxed environments, Ultralytics may fail writing to ~/.config/Ultralytics.
    # Point YOLO_CONFIG_DIR inside the repo to keep exports quiet and reproducible.
    repo_config_dir = (Path(__file__).resolve().parent / ".ultralytics").resolve()
    repo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(repo_config_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Ultralytics YOLO .pt model (e.g. yolov9m.pt) to ONNX."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolov9m.pt"),
        help="Path to .pt weights (default: yolov9m.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path (default: next to weights, same stem). If a directory, writes <stem>.onnx inside it.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size (single int, e.g. 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for export (e.g. "cpu", "0", "0,1").',
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes (batch/height/width).",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX graph (requires onnxsim).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 (may be unsupported on CPU).",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Export with NMS post-processing included (Ultralytics option).",
    )
    return parser.parse_args()


def main() -> int:
    _ensure_writable_ultralytics_config()

    args = _parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        print(
            "ERROR: Failed to import ultralytics. Install deps with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 3

    model = YOLO(str(weights))
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
        nms=args.nms,
    )

    exported_path = Path(str(exported_path)).expanduser().resolve()

    if args.output is None:
        print(f"Exported: {exported_path}")
        return 0

    output = args.output.expanduser()
    output_str = str(args.output)
    treat_as_dir = (
        (output.exists() and output.is_dir())
        or output_str.endswith(("/", "\\"))
        or output.suffix == ""
    )
    if treat_as_dir:
        output = output / f"{weights.stem}.onnx"
    if output.suffix.lower() != ".onnx":
        output = output.with_suffix(".onnx")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if exported_path != output:
        output.unlink(missing_ok=True)
        exported_path.replace(output)

    print(f"Exported: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
