from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw

src = Image.open('assets/source/face2.png').convert('RGB')
w, h = src.size
print('src:', w, h)

# crop del retrato (persona ocupa la mitad izquierda: cara ~x300-740, y470-1120)
crop_w = 900
crop_h = int(crop_w * (340 / 300))
x0, y0 = 70, 280
crop = src.crop((x0, y0, min(w, x0 + crop_w), min(h, y0 + crop_h)))
crop = crop.resize((300, 340), Image.Resampling.LANCZOS)

# mascara oval del retrato (feather) centrada en la cabeza+hombros
mask = Image.new('L', crop.size, 0)
d = ImageDraw.Draw(mask)
d.ellipse((45, 18, 258, 332), fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(radius=6))

base = crop.copy()
base = ImageEnhance.Brightness(base).enhance(0.35)  # fondo oscurecido
base = Image.composite(crop, base, mask)
base.save('probe_mask.png')
print('saved probe_mask.png')
