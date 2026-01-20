from PIL import Image
import os

folder = r""

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)

        width, height = img.size

        new_size = min(width, height)

        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2

        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(img_path)

print("All images cropped to square inplace!")
