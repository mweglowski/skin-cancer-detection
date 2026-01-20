from PIL import Image, ImageDraw
import os

folder = r""

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)

        width, height = img.size
        if width != height:
            print(f"Skipping {filename}: not square")
            continue

        stripe_height = int(height * 0.125)

        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width, stripe_height], fill=(0, 0, 0))
        draw.rectangle([0, height - stripe_height, width, height], fill=(0, 0, 0))

        img.save(img_path)

print("Done!")
