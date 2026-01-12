import os

def count_empty_txt(folder):
    empty_count = 0
    total = 0
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            total += 1
            path = os.path.join(folder, fname)
            if os.path.getsize(path) == 0:  # file rỗng
                empty_count += 1
    print(f"Tổng số file .txt: {total}")
    print(f"Số file .txt rỗng: {empty_count}")

# Ví dụ dùng
count_empty_txt("/root/trunglm8/mobile_phone_detection/MUID-IITR/data_29_9/test/labels")
