import os

def add_foldername_to_files(folder_path):
    folder = os.path.basename(os.path.abspath(folder_path))
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Bỏ qua thư mục con
        if os.path.isdir(file_path):
            continue

        new_name = f"{folder}_{filename}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(file_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

# Ví dụ: thay './dataset' bằng đường dẫn thật
add_foldername_to_files("/root/trunglm8/mobile_phone_detection/crop_person_data/210_1")
