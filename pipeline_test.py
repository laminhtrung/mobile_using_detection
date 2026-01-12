#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Pipeline (clean crop first): Person -> Crop -> Using-Phone
- Modes:
  * image : 1 ảnh hoặc thư mục ảnh (tuỳ chọn --recursive)
  * video : file video / webcam (0) / RTSP URL
- Model #1: person detector (.pt, ví dụ COCO person id=0)
- Model #2: using_phone detector (.pt của bạn)
- QUAN TRỌNG: Crop từ ảnh gốc (img_raw/frame_raw) trước khi vẽ box (img_draw/frame_draw)
- Outputs:
  * image mode:
      - /annotated: ảnh có overlay
      - /crops: crop người (nếu --save-crops)
      - /json: JSON (nếu --export-json)
      - /labels: YOLO TXT phone (nếu --export-yolo-phone)
  * video mode:
      - lưu video annotate nếu --save-video
      - lưu crops nếu --save-crops
"""

import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# thêm gần đầu file (sau import)
def safe_imshow(win_name, img):
    try:
        cv2.imshow(win_name, img)
        return True
    except cv2.error:
        return False

def safe_destroy_all():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ------------------------- utils -------------------------
def clip_xyxy(xyxy, W, H):
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    return [x1, y1, x2, y2]

def expand_box(xyxy, W, H, ratio=0.17):
    x1, y1, x2, y2 = map(int, xyxy)
    w = x2 - x1
    h = y2 - y1
    dx = int(w * ratio)
    dy = int(h * ratio)
    return clip_xyxy([x1 - dx, y1 - dy, x2 + dx, y2 + dy], W, H)

def draw_box(img, xyxy, color, label=None, thickness=2, fill=False, alpha=0.18):
    x1, y1, x2, y2 = map(int, xyxy)
    if fill:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        tf = max(thickness - 1, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, tf)[0]
        p2 = (x1 + t_size[0] + 6, y1 - t_size[1] - 6)
        cv2.rectangle(img, (x1, y1), p2, color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), tf, cv2.LINE_AA)

def put_fps(img, fps):
    txt = f"FPS: {fps:.1f}"
    cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1, cv2.LINE_AA)

def save_crop(img_raw, xyxy, out_dir, stem, idx, prefix="person"):
    x1, y1, x2, y2 = map(int, xyxy)
    crop = img_raw[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}_{prefix}{idx:03d}.jpg"
    cv2.imwrite(str(p), crop)
    return p

def list_images(path: Path, recursive: bool):
    if path.is_file() and path.suffix.lower() in ALLOWED_IMG:
        return [path]
    if path.is_dir():
        if recursive:
            return sorted([p for p in path.rglob("*") if p.suffix.lower() in ALLOWED_IMG])
        return sorted([p for p in path.glob("*") if p.suffix.lower() in ALLOWED_IMG])
    raise FileNotFoundError(f"Không tìm thấy ảnh/thư mục: {path}")

def yolo_xyxy_to_norm(xyxy, W, H):
    x1, y1, x2, y2 = map(float, xyxy)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / W, cy / H, bw / W, bh / H

# ------------------------- image mode -------------------------
def run_image_mode(args, person_model, phone_model):
    outdir = Path(args.outdir)
    out_anno = outdir / "annotated"
    out_crops = outdir / "crops"
    out_json  = outdir / "json"
    out_labels = outdir / "labels"
    out_anno.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        out_crops.mkdir(parents=True, exist_ok=True)
    if args.export_json:
        out_json.mkdir(parents=True, exist_ok=True)
    if args.export_yolo_phone:
        out_labels.mkdir(parents=True, exist_ok=True)

    images = list_images(Path(args.source), recursive=args.recursive)
    if not images:
        print("⚠️ Không tìm thấy ảnh hợp lệ.")
        return

    OK_COLOR = (60, 220, 60)
    ALERT_COLOR = (30, 30, 230)
    PHONE_BOX_COLOR = (0, 200, 255)

    total_people = 0
    total_phone  = 0
    t0_all = time.time()

    for im_path in images:
        img_raw = cv2.imread(str(im_path))
        if img_raw is None:
            print(f"⚠️ Lỗi đọc ảnh: {im_path}")
            continue
        img_draw = img_raw.copy()
        H, W = img_raw.shape[:2]
        stem = im_path.stem

        # Person detect
        res1 = person_model.predict(
            source=img_raw, imgsz=args.imgsz1, device=args.device,
            conf=args.conf1, verbose=False, classes=[args.person_class_id]
        )[0]

        person_boxes = []
        if res1.boxes is not None and len(res1.boxes) > 0:
            for b in res1.boxes:
                xyxy = b.xyxy[0].tolist()
                confp = float(b.conf[0].item()) if hasattr(b, "conf") else 1.0
                person_boxes.append((clip_xyxy(xyxy, W, H), confp))

        # JSON + YOLO lines
        per_image_json = {"image": str(im_path), "width": W, "height": H, "persons": []}
        yolo_lines = []

        any_phone = False
        for i, (xyxy, confp) in enumerate(person_boxes):
            xyxy_exp = expand_box(xyxy, W, H, ratio=0.2)
            x1, y1, x2, y2 = map(int, xyxy_exp)
            crop = img_raw[y1:y2, x1:x2]

            phone_found = False
            phone_boxes_global = []

            if crop.size > 0:
                res2 = phone_model.predict(
                    source=crop, imgsz=args.imgsz2, device=args.device,
                    conf=args.conf2, verbose=False
                )[0]
                if res2.boxes is not None and len(res2.boxes) > 0:
                    for pb in res2.boxes:
                        cx1, cy1, cx2, cy2 = pb.xyxy[0].tolist()
                        pconf = float(pb.conf[0].item()) if hasattr(pb, "conf") else 1.0
                        gx1, gy1 = int(cx1) + x1, int(cy1) + y1
                        gx2, gy2 = int(cx2) + x1, int(cy2) + y1
                        gxyxy = clip_xyxy([gx1, gy1, gx2, gy2], W, H)
                        phone_boxes_global.append((gxyxy, pconf))
                        phone_found, any_phone = True, True
                        if args.export_yolo_phone:
                            ncx, ncy, nw, nh = yolo_xyxy_to_norm(gxyxy, W, H)
                            yolo_lines.append(f"{args.yolo_phone_class} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

            # draw person
            total_people += 1
            color = ALERT_COLOR if phone_found else OK_COLOR
            draw_box(img_draw, xyxy_exp, color=color, label=f"person {confp:.2f}")

            # draw phone(s)
            if phone_found and args.draw_phone_box:
                for (gxyxy, pconf) in phone_boxes_global:
                    draw_box(img_draw, gxyxy, color=(0, 200, 255), label=f"phone {pconf:.2f}", fill=True)
                    total_phone += 1

            # json
            per_image_json["persons"].append({
                "xyxy": list(map(int, xyxy_exp)),
                "conf": confp,
                "using_phone": phone_found,
                "phones": [{"xyxy": list(map(int, pb[0])), "conf": float(pb[1])} for pb in phone_boxes_global]
            })

            # crops
            if args.save_crops:
                save_crop(img_raw, xyxy_exp, out_crops, stem, i)

        # save annotated
        out_img = out_anno / f"{stem}_annotated.jpg"
        cv2.imwrite(str(out_img), img_draw)

        # save json
        if args.export_json:
            out_js = out_json / f"{stem}.json"
            with open(out_js, "w", encoding="utf-8") as f:
                json.dump(per_image_json, f, ensure_ascii=False, indent=2)

        # save yolo labels
        if args.export_yolo_phone:
            lbl = out_labels / f"{stem}.txt"
            if yolo_lines:
                with open(lbl, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines) + "\n")
            else:
                open(lbl, "w", encoding="utf-8").close()

        print(f"[OK][IMG] {im_path.name}: persons={len(person_boxes)} | any_phone={any_phone}")

    print(f"\n[SUMMARY IMG] total images={len(images)} | total persons={total_people} | phone boxes drawn={total_phone} | t={time.time()-t0_all:.2f}s")

# ------------------------- video mode -------------------------
def run_video_mode(args, person_model, phone_model):
    # open source
    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {args.source}")

    # writer
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    OK_COLOR = (60, 220, 60)
    ALERT_COLOR = (30, 30, 230)
    PHONE_BOX_COLOR = (0, 200, 255)

    frame_idx = 0
    prev_t = time.time()
    fps = 0.0
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    crop_dir = Path(args.outdir) / "crops" if args.save_crops else None
    window_name = "Using-Phone Pipeline (video)"

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break
        H, W = frame_raw.shape[:2]
        frame_draw = frame_raw.copy()

        # person detect
        res1 = person_model.predict(
            source=frame_raw, imgsz=args.imgsz1, device=args.device,
            conf=args.conf1, verbose=False, classes=[args.person_class_id]
        )[0]

        person_boxes = []
        if res1.boxes is not None and len(res1.boxes) > 0:
            for b in res1.boxes:
                xyxy = b.xyxy[0].tolist()
                confp = float(b.conf[0].item()) if hasattr(b, "conf") else 1.0
                person_boxes.append((clip_xyxy(xyxy, W, H), confp))

        # for each person -> crop clean -> phone detect -> draw
        for i, (xyxy, confp) in enumerate(person_boxes):
            xyxy_exp = expand_box(xyxy, W, H, ratio=0.2)
            x1, y1, x2, y2 = map(int, xyxy_exp)
            crop = frame_raw[y1:y2, x1:x2]

            phone_found = False
            phone_boxes_global = []

            if crop.size > 0:
                res2 = phone_model.predict(
                    source=crop, imgsz=args.imgsz2, device=args.device,
                    conf=args.conf2, verbose=False
                )[0]
                if res2.boxes is not None and len(res2.boxes) > 0:
                    for pb in res2.boxes:
                        cx1, cy1, cx2, cy2 = pb.xyxy[0].tolist()
                        pconf = float(pb.conf[0].item()) if hasattr(pb, "conf") else 1.0
                        gx1, gy1 = int(cx1) + x1, int(cy1) + y1
                        gx2, gy2 = int(cx2) + x1, int(cy2) + y1
                        gxyxy = clip_xyxy([gx1, gy1, gx2, gy2], W, H)
                        phone_boxes_global.append((gxyxy, pconf))
                        phone_found = True

            # draw person
            color = ALERT_COLOR if phone_found else OK_COLOR
            draw_box(frame_draw, xyxy_exp, color=color, label=f"person {confp:.2f}")

            # draw phones
            if phone_found and args.draw_phone_box:
                for (gxyxy, pconf) in phone_boxes_global:
                    draw_box(frame_draw, gxyxy, color=PHONE_BOX_COLOR,
                             label=f"phone {pconf:.2f}", fill=True)

            # save crops
            if args.save_crops and crop_dir is not None:
                stem = f"f{frame_idx:06d}"
                save_crop(frame_raw, xyxy_exp, crop_dir, stem, i)

        # fps
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        put_fps(frame_draw, fps)

        # writer
        if args.save_video:
            if writer is None:
                outp = Path(args.save_video)
                outp.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(outp), fourcc, args.fps if args.fps>0 else 25.0, (W, H))
            writer.write(frame_draw)

        # show
        if args.show:
            if not safe_imshow(window_name, frame_draw):
                print("⚠️ Môi trường headless: tự tắt --show để tránh lỗi.")
                args.show = False
            else:
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    # thay dòng này:
    # cv2.destroyAllWindows()
    safe_destroy_all()
    print("[SUMMARY VID] done.")

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser("Unified pipeline: image/video Person -> Phone (clean crop first)")
    ap.add_argument("--mode", choices=["image","video"], required=True, help="Chọn 'image' hoặc 'video'")
    ap.add_argument("--source", required=True, help="Ảnh/Thư mục (image) hoặc file/0/RTSP (video)")
    ap.add_argument("--person-weights", required=True, help="YOLO .pt detect person")
    ap.add_argument("--phone-weights",  required=True, help="YOLO .pt detect using_phone")
    ap.add_argument("--person-class-id", type=int, default=0, help="ID class 'person' (COCO=0)")
    ap.add_argument("--imgsz1", type=int, default=768)
    ap.add_argument("--imgsz2", type=int, default=768)
    ap.add_argument("--conf1", type=float, default=0.45)
    ap.add_argument("--conf2", type=float, default=0.3)
    ap.add_argument("--device", type=str, default="0", help="'0' GPU0 hoặc 'cpu'")
    ap.add_argument("--outdir", type=str, default="./runs/pipeline_phone_use", help="Thư mục output (image) hoặc nơi chứa crops/video")
    # image-mode only
    ap.add_argument("--recursive", action="store_true", help="(image) Duyệt subfolders")
    ap.add_argument("--export-json", action="store_true", help="(image) Lưu JSON mỗi ảnh")
    ap.add_argument("--export-yolo-phone", action="store_true", help="(image) Xuất YOLO TXT phone")
    ap.add_argument("--yolo-phone-class", type=int, default=0, help="(image) Class id khi xuất YOLO TXT")
    # common
    ap.add_argument("--save-crops", action="store_true", help="Lưu crop người (từ frame/ảnh gốc)")
    ap.add_argument("--draw-phone-box", action="store_true", help="Vẽ box phone lên frame/ảnh gốc")
    # video-mode only
    ap.add_argument("--show", action="store_true", help="(video) Mở cửa sổ xem realtime")
    ap.add_argument("--save-video", type=str, default="", help="(video) Đường dẫn lưu mp4 annotate")
    ap.add_argument("--fps", type=float, default=0.0, help="(video) FPS ghi file (0=25)")
    args = ap.parse_args()

    # Load models
    person_model = YOLO(args.person_weights)
    phone_model  = YOLO(args.phone_weights)

    if args.mode == "image":
        run_image_mode(args, person_model, phone_model)
    else:
        run_video_mode(args, person_model, phone_model)

if __name__ == "__main__":
    main()
