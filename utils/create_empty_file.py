''' the purpose is that create the empty files for negative images which have no labels'''
import os
from pathlib import Path

# ---- Paths ----
img_dir = Path("/root/trunglm8/mobile_phone_detection/MUID-IITR/data_29_9/test/images")   # folder with images
lbl_dir = Path("/root/trunglm8/mobile_phone_detection/MUID-IITR/data_29_9/test/labels")   # folder where YOLO labels should go
lbl_dir.mkdir(parents=True, exist_ok=True)

# extensions to match
exts = [".jpg", ".jpeg", ".png"]

# ---- Loop over images ----
for img_path in img_dir.glob("*"):
    if img_path.suffix.lower() not in exts:
        continue  # skip non-image files

    label_path = lbl_dir / (img_path.stem + ".txt")

    # if label file does not exist, create empty one
    if not label_path.exists():
        label_path.touch()
        print(f"[INFO] Created empty label for: {img_path.name}")
