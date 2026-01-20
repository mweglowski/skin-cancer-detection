from PIL import Image, ImageOps
from pathlib import Path

def pad_images_inplace(folder):
    folder = Path(folder)

    for img_path in folder.glob("*.jpg"):
        img = Image.open(img_path)

        w, h = img.size
        if h == w:
            continue

        target_h = w
        pad_total = target_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        padding = (0, pad_top, 0, pad_bottom)

        padded = ImageOps.expand(img, padding, fill=(0, 0, 0))
        padded.save(img_path)

    print("Padding complete.")

pad_images_inplace(r"C:\Users\Oskar\Downloads\ruskie\archive (1)\all")
