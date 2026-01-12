import kagglehub
import shutil
import os

# Đường dẫn bạn muốn lưu dataset
save_dir = "/root/trunglm8/mobile_phone_detection"
os.makedirs(save_dir, exist_ok=True)

# Download dataset (mặc định vào cache của kagglehub)
path = kagglehub.dataset_download("lakshyataragi/mobilephoneusagedatasetiitr")

print("Dataset cached at:", path)

# Copy toàn bộ dataset sang save_dir
dst_path = os.path.join(save_dir, os.path.basename(path))
if not os.path.exists(dst_path):
    shutil.copytree(path, dst_path)

print("Dataset copied to:", dst_path)
