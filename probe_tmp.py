import numpy as np, sys
from PIL import Image, ImageOps, ImageFilter

src = Image.open('assets/source/face.png').convert('RGBA')
w, h = src.size
crop_w = int(w * 0.74); crop_h = int(crop_w * (340 / 300))
left = max(0, int(w * 0.13)); top = int(h * 0.02)
crop = src.crop((left, top, min(w, left + crop_w), min(h, top + crop_h))).resize((300, 340), Image.Resampling.LANCZOS)
bg_img = Image.new('RGBA', crop.size, 'black'); bg_img.alpha_composite(crop)
gray = ImageOps.grayscale(bg_img.convert('RGB'))
g = np.asarray(gray).astype(np.float32) / 255.0
low = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=7.0))).astype(np.float32) / 255.0
feat = np.clip((g - low + 0.50) / 1.05, 0, 1)
print('feat percentiles:', np.percentile(feat, [90, 97, 99, 99.9]).round(3), 'max', round(float(feat.max()), 3))
print('feat > 0.78 px :', int((feat > 0.78).sum()))
