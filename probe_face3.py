import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import sys

GRID = (300, 340)

def dither_face(infile, crop_box, outpng):
    src = Image.open(infile).convert("RGBA")
    crop = src.crop(crop_box)
    # recorte al ratio del grid 300/340
    bx0, by0, bx1, by1 = crop_box
    cw, ch = bx1 - bx0, by1 - by0
    target = 300 / 340
    if cw / ch > target:
        nh = int(cw / target); y0 = max(0, by0 - (nh - ch) // 2)
        crop = src.crop((bx0, y0, bx1, min(src.size[1], y0 + nh)))
    else:
        nw = int(ch * target); x0 = max(0, bx0 - (nw - cw) // 2)
        crop = src.crop((x0, by0, min(src.size[0], x0 + nw), by1))
    print('crop final:', crop.size)
    crop = crop.resize(GRID, Image.Resampling.LANCZOS)

    alpha = np.asarray(crop)[..., 3]
    person = alpha > 40

    bg_img = Image.new("RGBA", crop.size, "black")
    bg_img.alpha_composite(crop)
    gray = ImageOps.grayscale(bg_img.convert("RGB"))
    prepd = ImageOps.autocontrast(gray, cutoff=1)
    prepd = ImageEnhance.Contrast(prepd).enhance(1.08)
    prepd = prepd.filter(ImageFilter.UnsharpMask(radius=1.4, percent=100, threshold=2))

    field = np.asarray(prepd).astype(np.float32) / 255.0
    field = field ** 0.80  # levanta sombras: la mitad en sombra existe
    # self-lit de lo oscuro: pelo/marco de gafas reciben dots tenues
    # (si no, el pelo negro se pierde contra el fondo del panel)
    with np.errstate(invalid='ignore'):
        yy = np.arange(field.shape[0])[:, None] / field.shape[0]
        hairzone = (yy < 0.20)
        dark_fill = np.where(person & (field < 0.38) & hairzone, 0.20 + field * 0.25, field)
    field = np.where(person, dark_fill, field)
    hh = field.shape[0]
    ramp = np.clip((hh - 4 - np.arange(hh)) / 40.0, 0, 1)
    field = field * ramp[:, None]
    field = np.clip(field + np.random.default_rng(7).normal(0, 0.010, field.shape), 0, 1)

    # floyd-steinberg serpentine (igual que generate.py)
    H, W_ = field.shape
    err = (field * 255.0).astype(np.float32)
    bits = np.zeros((H, W_), dtype=bool)
    for y in range(H):
        row = err[y]
        if y % 2 == 0:
            rng_i = range(W_); dn = 1
        else:
            rng_i = range(W_ - 1, -1, -1); dn = -1
        for x in rng_i:
            v = row[x]
            b = v >= 128
            bits[y, x] = b
            e = v - (255.0 if b else 0.0)
            if x + dn < W_ and x + dn >= 0: row[x + dn] += e * 7 / 16
            if y + 1 < H:
                nxt = err[y + 1]
                if x - dn < W_ and x - dn >= 0: nxt[x - dn] += e * 3 / 16
                nxt[x] += e * 5 / 16
                if x + dn < W_ and x + dn >= 0: nxt[x + dn] += e * 1 / 16
    active = bits & person

    # speckles fuera
    from scipy.ndimage import label as ndi_label
    lab, n = ndi_label(active)
    if n:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        active = active & np.isin(lab, [int(np.argmax(sizes))] + [i for i in range(1, n + 1) if sizes[i] > 400])

    vis = np.full((H, W_, 3), 13, dtype=np.uint8)
    vis[..., 0] = 13; vis[..., 1] = 17; vis[..., 2] = 23
    vis[active] = (215, 219, 226)
    Image.fromarray(vis).resize((600, 680), Image.Resampling.NEAREST).save(outpng)
    print('dots:', int(active.sum()), '->', outpng)

dither_face('assets/source/face3_cut.png', (30, 40, 697, 780), 'probe_face3.png')
