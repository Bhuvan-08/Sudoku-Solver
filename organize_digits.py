import os
import shutil

def organize_synthetic_digits(base_dir="synthetic_digits"):
    # Make subfolders for 0–9
    for digit in range(10):
        os.makedirs(os.path.join(base_dir, str(digit)), exist_ok=True)

    # Move files into corresponding folders
    for filename in os.listdir(base_dir):
        if filename.endswith(".png") and "_" in filename:
            digit_label = filename.split("_")[0]
            src = os.path.join(base_dir, filename)
            dst_dir = os.path.join(base_dir, digit_label)
            dst = os.path.join(dst_dir, filename)
            shutil.move(src, dst)

    print("✅ Synthetic digits organized into subfolders.")

if __name__ == "__main__":
    organize_synthetic_digits()
