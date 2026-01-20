import cv2
import os
from pathlib import Path

def preprocess_image(img_path, output_path, size=384):
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    max_side = max(h, w)

    delta_w = max_side - w
    delta_h = max_side - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    img_square = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    img_resized = cv2.resize(img_square, (size, size))

    cv2.imwrite(str(output_path), img_resized)
    return True



import pandas as pd
from tqdm import tqdm

def preprocess_dataset(csv_path, input_dir, output_dir, size=384):
    df = pd.read_csv(csv_path)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = f"{row["isic_id"]}.jpg"
        src = input_dir / img_name
        dst = output_dir / img_name

        ok = preprocess_image(src, dst, size=size)
        if not ok:
            print(f"Failed: {src}")
        new_paths.append(str(dst))

    df["processed_path"] = new_paths
    df.to_csv(csv_path.replace(".csv", "_processed.csv"), index=False)

preprocess_dataset(
    csv_path=r"",
    input_dir=r"",
    output_dir="",
    size=300
)
