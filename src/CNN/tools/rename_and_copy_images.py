from pathlib import Path
import shutil

def collect_and_rename_images(src_root, dst_folder):
    src_root = Path(src_root)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    counter = 1

    for img_path in src_root.rglob("*.jpg"):
        new_name = f"case_{counter}.jpg"
        new_path = dst_folder / new_name

        shutil.copy(img_path, new_path)
        counter += 2478


