#!/usr/bin/env python3
import cv2, argparse
from pathlib import Path

def draw_one(img_path: Path, labels_dir: Path, out_dir: Path):
    txt_path = labels_dir / (img_path.stem + ".txt")
    img = cv2.imread(str(img_path))
    if img is None or not txt_path.exists():
        print("Skip", img_path.name); return
    H, W = img.shape[:2]
    for line in txt_path.read_text().splitlines():
        if not line.strip(): continue
        cls, cx, cy, w, h = map(float, line.split()[:5])
        x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, str(int(cls)), (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (img_path.stem + "_ann.jpg")
    cv2.imwrite(str(out_file), img)
    print("Saved:", out_file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Visualize YOLO labels on images")
    ap.add_argument("--images", required=True, help="Path to folder images")
    ap.add_argument("--labels", required=True, help="Path to folder labels (.txt)")
    ap.add_argument("--out", default="annotated", help="Output folder")
    args = ap.parse_args()

    images_dir, labels_dir, out_dir = Path(args.images), Path(args.labels), Path(args.out)

    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() in {".jpg",".jpeg",".png"}:
            draw_one(img_path, labels_dir, out_dir)
