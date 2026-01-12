import os
import random
import shutil
from pathlib import Path

def create_subset_images(src_images, dst_root, x):
    """
    Create a subset dataset with x images only (no labels).

    Args:
        src_images (str): Path to images folder (e.g., 'dataset/images').
        dst_root   (str): Path to output subset folder.
        x          (int): Number of images to copy.
    """

    src_images = Path(src_images)
    dst_images = Path(dst_root)

    # Create output folder
    dst_images.mkdir(parents=True, exist_ok=True)

    # Get list of images
    all_images = list(src_images.glob("*.jpg")) + list(src_images.glob("*.png")) + list(src_images.glob("*.jpeg"))
    if len(all_images) == 0:
        raise ValueError(f"No images found in {src_images}")

    # Randomly select x images
    subset_images = random.sample(all_images, min(x, len(all_images)))

    # Copy images only
    for img_path in subset_images:
        shutil.copy(img_path, dst_images / img_path.name)

    print(f"✅ Created subset dataset with {len(subset_images)} images at {dst_root}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create subset dataset (images only)")
    parser.add_argument("--src_images", type=str, required=True, help="Path to images folder")
    parser.add_argument("--dst_root", type=str, required=True, help="Output subset folder")
    parser.add_argument("-x", type=int, required=True, help="Number of images to select")

    args = parser.parse_args()

    create_subset_images(args.src_images, args.dst_root, args.x)
