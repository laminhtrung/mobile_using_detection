#!/usr/bin/env python3
"""FastAPI demo for YOLO mobile phone detection."""

from __future__ import annotations

import base64
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("/root/trunglm8/mobile_phone_detection/runs/train/support_autolabel_version/weights/best.pt")

app = FastAPI(title="Mobile Phone Detection Demo")
templates = Jinja2Templates(directory="templates")


def load_model(weights_path: Path) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    return YOLO(str(weights_path))


model = load_model(DEFAULT_WEIGHTS)


def read_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("Empty file uploaded")
    return Image.open(BytesIO(data)).convert("RGB")


def run_prediction(image: Image.Image, conf: float = 0.25, iou: float = 0.7) -> Tuple[str, List[dict]]:
    results = model.predict(image, imgsz=640, conf=conf, iou=iou, verbose=False)
    if not results:
        return None, []

    res = results[0]
    plotted = res.plot()
    rgb_plot = Image.fromarray(plotted[:, :, ::-1])

    buffer = BytesIO()
    rgb_plot.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")

    detections = []
    boxes = res.boxes
    if boxes is None or boxes.data is None or len(boxes) == 0:
        return encoded_image, detections

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for box, score, cls_id in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            {
                "class": res.names.get(int(cls_id), str(cls_id)),
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )

    return encoded_image, detections


@app.get("/", response_class=HTMLResponse)
def get_index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error": None,
            "result_image": None,
            "detections": [],
            "summary": {"total": 0, "class_breakdown": []},
        },
    )


@app.post("/", response_class=HTMLResponse)
async def post_predict(request: Request, image: UploadFile = File(...)):
    error = None
    result_image = None
    detections = []
    summary: Dict[str, object] = {"total": 0, "class_breakdown": []}

    if image is None or image.filename == "":
        error = "Please choose an image file to upload."
    else:
        try:
            raw_bytes = await image.read()
            pil_image = read_image(raw_bytes)
            result_image, detections = run_prediction(pil_image)
            if result_image is None:
                error = "No detections were produced for this image."
            else:
                class_counts = Counter(det["class"] for det in detections)
                summary = {
                    "total": len(detections),
                    "class_breakdown": [
                        {"label": label, "count": count}
                        for label, count in class_counts.most_common()
                    ],
                }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = f"Failed to process image: {exc}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error": error,
            "result_image": result_image,
            "detections": detections,
            "summary": summary,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp:app", host="0.0.0.0", port=8000, reload=True)
