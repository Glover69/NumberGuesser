from PIL import Image, ImageFilter
import numpy as np

# img = Image.open("/Users/danielglover/Python/Forge/output/train/0_11.jpg").convert("L")
# arr = np.array(img)
# print("min:", arr.min(), "max:", arr.max(), "mean:", arr.mean())
# print(arr)

def add_margin(bbox, percent=45):
    left, top, right, bottom = bbox
    w, h = right - left, bottom - top
    pad_x, pad_y = w * percent / 100, h * percent / 100
    return left - pad_x, top - pad_y, right + pad_x, bottom + pad_y

def process_handdrawn(file):
    img = Image.open(file).convert('L')
    img_to_arr = np.array(img)

    bbox = find_bounding_box_of_nonzero_pixels(img_to_arr)

    padded_bbox = add_margin(bbox)

    cropped = img.crop(padded_bbox)

    width = padded_bbox[2]-padded_bbox[0]
    height = padded_bbox[3]-padded_bbox[1]

    scale = 20 / max(width, height)
    new_w, new_h = width * scale, height * scale
    resized = cropped.resize((int(new_w), int(new_h)))

    resized = resized.filter(ImageFilter.GaussianBlur(radius=1))  # start small, tune

    canvas = Image.new('L', (28, 28), 0)
    offset = ((28 - int(new_w)) // 2, (28 - int(new_h)) // 2)
    canvas.paste(resized, offset)

    final = np.array(canvas)
    return final

def find_bounding_box_of_nonzero_pixels(arr):
    rows, cols = np.where(arr > 0)   # indices of "ink" pixels
    return min(cols), min(rows), max(cols), max(rows)   # PIL crop box format

# processed = process_handdrawn('./data/hand-drawn/3_second.jpg')
# print(processed.min())
# print(processed.max())
# print(processed.mean())