#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"


python3 convert.py "$@"\
    --weights /root/trunglm8/mobile_phone_detection/runs/train/ver_26_9/weights/best.pt \
    --imgsz 640 --opset 17 --output ./best.onnx
