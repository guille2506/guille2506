#!/usr/bin/env python3
"""MOTO readme artwork generator.

Port of the macu-dev technique (floyd-steinberg dither + optimal transport
morph) recolored to the MOTO theme 1 palette: blueprint dark + red accent.

Inputs (edit freely):
    assets/source/face.png      portrait photo (GitHub avatar)
    assets/source/maxwell.png   maxwell the cat (3d render capture)

Run from the repository root:
    python scripts/moto/generate.py

Requires: pip install -r scripts/moto/requirements.txt
Writes:   assets/moto-face.svg, assets/moto-maxwell.svg, assets/moto-morph.svg
"""

from __future__ import annotations

import html
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.ndimage import binary_erosion, label as ndi_label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets" / "source"
ASSETS = ROOT / "assets"

# --------------------------------------------------------------------------- #
# tema 1 MOTO
# --------------------------------------------------------------------------- #
MOTO = {
    "bg":      "#0d1117",
    "panel":   "#090b10",
    "panel2":  "#0b1220",
    "grid":    "#1d2946",
    "line":    "#26334d",
    "muted":   "#9aa0a8",
    "body":    "#c9cdd4",
    "ink":     "#f2f3f5",
    "ice":     "#d7dbe2",
    "accent":  "#ff2633",
    "red_dim": "#8f1d2a",
    "chrome":  "#27406b",
}

MONO = "ui-monospace,SFMono-Regular,Consolas,monospace"

YAML_ROWS = [
    (0, "profile", ""),
    (1, "subject", "Guillermo Illanes"),
    (1, "role", "Fullstack Developer"),
    (1, "origin", "San Juan, Argentina"),
    (1, "status", "open to work"),
    (1, "learning", "Java / React / TS"),
    (0, "stack", ""),
    (1, "frontend", "TS - React - Angular - Tailwind"),
    (1, "backend", "Java - Spring - Node - PHP"),
    (1, "database", "MySQL - MongoDB"),
    (0, "mascota", ""),
    (1, "model", "maxwell_the_cat_dingus.glb"),
    (1, "site", "nuevoportfolio.vercel.app"),
    (0, "contact", ""),
    (1, "github", "guille2506"),
]

SEED = 314159
TRAVELLER_COUNT = 900
HOLD_PARTICLE_COUNT = 2_400
FACE_GRID = (300, 340)
FRAME_INNER = (74, 154, 314, 340)   # x, y, w, h mapped dot area inside visual frame


# --------------------------------------------------------------------------- #
# helpers (port de macu-dev)
# --------------------------------------------------------------------------- #

def edge_connected(mask: np.ndarray) -> np.ndarray:
    """True for values in mask connected to an image edge (flood fill)."""
    connected = np.zeros(mask.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    h, w = mask.shape
    for x in range(w):
        queue.extend(((0, x), (h - 1, x)))
    for y in range(h):
        queue.extend(((y, 0), (y, w - 1)))
    while queue:
        y, x = queue.popleft()
        if connected[y, x] or not mask[y, x]:
            continue
        connected[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                queue.append((ny, nx))
    return connected


def color_flood_background(rgb: np.ndarray, tol: float = 30.0, blue_guard: bool = False) -> np.ndarray:
    """Flood the background color from the four corners (color tolerance)."""
    h, w, _ = rgb.shape
    corners = np.concatenate((
        rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
        rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)))
    base = np.median(corners, axis=0)
    bg = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(w):
        queue.extend(((0, x), (h - 1, x)))
    for y in range(h):
        queue.extend(((y, 0), (y, w - 1)))
    while queue:
        y, x = queue.popleft()
        if bg[y, x]:
            continue
        # el fondo MOTO/teal es azul-dominante: corta el crawl por
        # degradados calientes (glow rojo del rim del gato, sombras negras)
        if np.linalg.norm(rgb[y, x].astype(np.float32) - base) > tol:
            continue
        if not blue_guard:
            pass
        elif rgb[y, x, 2] < rgb[y, x, 0] + 6:
            continue
        bg[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not bg[ny, nx]:
                queue.append((ny, nx))
    return bg


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit Floyd-Steinberg; True = lit pixel."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    h, w = work.shape
    for y in range(h):
        ltr = y % 2 == 0
        xs = range(w) if ltr else range(w - 1, -1, -1)
        direction = 1 if ltr else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < w:
                work[y, nx] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - direction < w:
                    work[y + 1, x - direction] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < w:
                    work[y + 1, nx] += err * 1 / 16
    return out


def point_path(points) -> str:
    """Aggregate horizontal runs of dots into compact path segments."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def particle_path(points) -> str:
    """Independent one-px particles (no joining)."""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    return "".join(f"M{x} {y}h1" for x, y in unique)


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Order target points by min-cost assignment (hungarian)."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def place(points, area_x, area_y, area_w, area_h, bbox, fill=0.94):
    """Scale/center local-space points inside a target area (bbox-aware)."""
    x0, x1, y0, y1 = bbox
    scale = min(area_w / max(x1 - x0, 1), area_h / max(y1 - y0, 1)) * fill
    tx = area_x + (area_w - (x1 - x0) * scale) / 2 - x0 * scale
    ty = area_y + (area_h - (y1 - y0) * scale) / 2 - y0 * scale
    return np.column_stack((tx + points[:, 0] * scale, ty + points[:, 1] * scale))


def num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def time_num(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def animate_values(points: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in points)


def text_width(text: str, font_size: float) -> float:
    return len(text) * font_size * 0.605


# --------------------------------------------------------------------------- #
# fuentes de datos: cara y gato
# --------------------------------------------------------------------------- #

def face_points(rng: np.random.Generator) -> tuple:
    """Dithered portrait dots in LOCAL space (0..300 x 0..340) + bbox.

    Fuente: face2_cut.png (rembg U2-Net, silueta RGBA con fondo transparente).
    El corte hace innecesarios los floods de fondo: fuera de la silueta
    no hay pixeles y el panel queda limpio.
    """
    src = Image.open(SOURCE / "face2_cut.png").convert("RGBA")
    w, h = src.size
    crop_w = 900
    crop_h = int(crop_w * (340 / 300))
    x0, y0 = 70, 280
    crop = src.crop((x0, y0, min(w, x0 + crop_w), min(h, y0 + crop_h)))
    crop = crop.resize(FACE_GRID, Image.Resampling.LANCZOS)

    alpha = np.asarray(crop)[..., 3]
    person = alpha > 40

    bg_img = Image.new("RGBA", crop.size, "black")
    bg_img.alpha_composite(crop)
    gray = ImageOps.grayscale(bg_img.convert("RGB"))
    prepd = ImageOps.autocontrast(gray, cutoff=1)
    prepd = ImageEnhance.Contrast(prepd).enhance(1.08)
    prepd = prepd.filter(ImageFilter.UnsharpMask(radius=1.4, percent=100, threshold=2))

    field = np.asarray(prepd).astype(np.float32) / 255.0
    # fade del busto al borde inferior del crop
    hh = field.shape[0]
    ramp = np.clip((hh - 8 - np.arange(hh)) / 46.0, 0, 1)
    field = field * ramp[:, None]
    # un pelo de ruido para romper bandas planas
    field = np.clip(field + np.random.default_rng(SEED + 11).normal(0, 0.010, field.shape), 0, 1)

    bits = floyd_steinberg(field * 255.0)
    active = bits & person

    # speckles sueltos del borde alpha mueren
    lab, n = ndi_label(active)
    if n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep_labels = {int(np.argmax(sizes))}
        for lid in range(1, n + 1):
            if sizes[lid] > 400:
                keep_labels.add(lid)
        active = active & np.isin(lab, list(keep_labels))

    ys, xs = np.where(active)
    if len(xs) == 0:
        raise SystemExit("no face dots")
    points = np.column_stack((xs, ys)).astype(np.float32)
    red_flags = np.zeros(len(points), dtype=bool)  # cara limpia ice
    if len(points) > 13000:
        keep2 = rng.choice(len(points), 13000, replace=False)
        points = points[keep2]
    bbox = (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1)
    return points, bbox, red_flags


def cat_points() -> tuple:
    """Maxwell dot art in LOCAL space (0..440 x 0..440) + bbox.

    Intensity field: silhouette always present (dim red), variable density by
    luminance. Bright points classify red rim vs warm-white chest/whiskers.
    """
    src = Image.open(SOURCE / "maxwell.png").convert("RGB")
    rgb = np.asarray(src).astype(np.float32)
    bg = color_flood_background(rgb, tol=26.0, blue_guard=True)
    visible = ~bg

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = r - np.maximum(g, b)

    # el render es cálido en general: pecho y rim comparten familia naranja.
    # se separa por BRILLO (ice = luces/pecho/bigotes) y Saturación (rojo).
    ice_mask = visible & (lum > 135)
    warm_mask = visible & (~ice_mask) & (sat > 30)
    body_mask = visible & (~ice_mask) & (~warm_mask)

    ys, xs = np.where(visible)
    bbox = (int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1)

    def dots_from(mask, field):
        f = np.where(mask, field, 0.0)
        by, bx = np.where(floyd_steinberg(f * 255.0))
        return np.column_stack((bx, by)).astype(np.float32)

    ice = dots_from(ice_mask, np.full(visible.shape, 0.92))

    # contorno de la silueta: banda 2px. el px externo = linea SOLIDA (1.0),
    # el interno = trazo secundario mas tenue -> linea continua legible
    outline2 = visible & ~binary_erosion(visible, iterations=2)
    outline1 = visible & ~binary_erosion(visible, iterations=1)
    warm_mask = visible & (~ice_mask) & (sat > 26) & (~outline2)
    body_mask = visible & (~ice_mask) & (~warm_mask) & (~outline2)

    # rojo: linea solida en el borde, interior apenas insinuado
    red_field = np.where(outline2,
                         np.where(outline1, 1.0, np.clip(0.42 + lum / 300.0, 0, 0.62)),
                         np.clip(0.04 + lum / 620.0, 0, 0.18))
    red = dots_from(outline2 | warm_mask, red_field)
    # cuerpo: hatch muy tenue, la forma la llevan el contorno y el pecho
    dim = dots_from(body_mask, np.clip(0.12 + lum / 340.0, 0, 1))

    rng = np.random.default_rng(SEED)
    if len(red) > 3200:
        red = red[rng.choice(len(red), 3200, replace=False)]
    if len(dim) > 4200:
        dim = dim[rng.choice(len(dim), 4200, replace=False)]

    return red, ice, dim, bbox


# --------------------------------------------------------------------------- #
# paneles estaticos
# --------------------------------------------------------------------------- #

def chrome_defs(prefix: str) -> str:
    m = MOTO
    return (
        f'<pattern id="{prefix}grille" width="48" height="48" patternUnits="userSpaceOnUse">'
        f'<path d="M48 0V48H0" fill="none" stroke="{m["grid"]}" stroke-width="1" opacity="0.5"/></pattern>'
        f'<pattern id="{prefix}cross" width="192" height="192" patternUnits="userSpaceOnUse" x="88" y="62">'
        f'<path d="M90 96h12M96 90v12" fill="none" stroke="{m["accent"]}" '
        f'stroke-opacity="0.4" stroke-width="1.6"/></pattern>'
        f'<clipPath id="{prefix}clip"><rect x="28" y="58" width="374" height="352"/></clipPath>'
    )


def chrome_header(title: str, right_label: str, prefix: str) -> str:
    m = MOTO
    return (
        f'<rect width="430" height="410" fill="{m["bg"]}"/>'
        f'<rect width="430" height="410" fill="url(#{prefix}grille)"/>'
        f'<rect width="430" height="410" fill="url(#{prefix}cross)"/>'
        f'<rect x="12" y="12" width="406" height="386" fill="none" stroke="#ffffff" stroke-opacity="0.10"/>'
        f'<text x="28" y="36" font-size="13" letter-spacing="4" fill="{m["ink"]}">{title} '
        f'<tspan fill="{m["accent"]}">.</tspan></text>'
        f'<text x="402" y="36" font-size="10" letter-spacing="2" fill="{m["muted"]}" text-anchor="end">{right_label}</text>'
        f'<line x1="28" y1="48" x2="402" y2="48" stroke="#ffffff" stroke-opacity="0.12"/>'
    )


def chrome_footer(left_label: str, right_html: str) -> str:
    m = MOTO
    return (
        f'<line x1="28" y1="374" x2="402" y2="374" stroke="#ffffff" stroke-opacity="0.12"/>'
        f'<text x="28" y="392" font-size="10" letter-spacing="2" fill="{m["muted"]}">{left_label}</text>'
        f'<text x="402" y="392" font-size="10" letter-spacing="2" fill="{m["muted"]}" text-anchor="end">{right_html}</text>'
    )


FACE_AREA = (58, 74, 314, 288)   # x, y, w, h dentro del panel 430x410


def render_face_panel(points, bbox, red_flags) -> str:
    m = MOTO
    placed = place(points, *FACE_AREA, bbox, fill=0.85)
    groups = (
        f'<path d="{point_path(placed)}" fill="none" stroke="{m["accent"]}" stroke-width="1.08" stroke-opacity="0.95"/>'
    )
    if len(red_flags) and bool(np.asarray(red_flags).any()):
        red_p = place(points[red_flags], *FACE_AREA, bbox, fill=0.99)
        groups = (
            f'<path d="{point_path(placed[~red_flags])}" fill="none" stroke="{m["accent"]}" stroke-width="1.08" stroke-opacity="0.9"/>'
            + f'<path d="{point_path(red_p)}" fill="none" stroke="{m["ice"]}" stroke-width="1.2" stroke-opacity="0.95"/>'
        )
    body = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="430" height="410" viewBox="0 0 430 410" '
        f'font-family="{MONO}" role="img" aria-label="retrato 1-bit de Guillermo">'
        "<defs>" + chrome_defs("f") + "</defs>"
        + chrome_header("GUILLERMO", "1-BIT / DITHER", "f")
        + '<g clip-path="url(#fclip)" shape-rendering="crispEdges">'
        + groups
        + "</g>"
        + chrome_footer("src: selfie REMBG",
                        f'PTS {len(points):05d} <tspan fill="{m["accent"]}">&#9679; FS/SERPENTINE</tspan>')
        + "</svg>"
    )
    return body


def render_cat_panel(red, ice, dim, bbox) -> str:
    m = MOTO
    red_p = place(red, *FACE_AREA, bbox, fill=0.97)
    ice_p = place(ice, *FACE_AREA, bbox, fill=0.97)
    dim_p = place(dim, *FACE_AREA, bbox, fill=0.97)
    groups = (
        f'<path d="{point_path(dim_p)}" fill="none" '
        f'stroke="{m["red_dim"]}" stroke-width="1.05" stroke-opacity="0.55"/>'
        + f'<path d="{point_path(red_p)}" fill="none" stroke="{m["accent"]}" stroke-width="1.2" stroke-opacity="0.95"/>'
        + f'<path d="{point_path(ice_p)}" fill="none" stroke="{m["ice"]}" stroke-width="1.15" stroke-opacity="0.92"/>'
    )
    right = f'RENDER <tspan fill="{m["accent"]}">&#9679; LIVE</tspan>'
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="430" height="410" viewBox="0 0 430 410" '
        f'font-family="{MONO}" role="img" aria-label="maxwell en puntos">'
        "<defs>" + chrome_defs("c") + "</defs>"
        + chrome_header("MAXWELL", "3D MASCOT / GLB", "c")
        + '<g clip-path="url(#cclip)" shape-rendering="crispEdges">'
        + groups
        + "</g>"
        + chrome_footer("model: maxwell_the_cat_dingus.glb", right)
        + "</svg>"
    )


# --------------------------------------------------------------------------- #
# banner morph (cara -> maxwell)
# --------------------------------------------------------------------------- #

MORPH_W, MORPH_H = 1180, 610
INTRO_SECONDS = 3.2
TRANSITION_SECONDS = 1.4
HOLD_SECONDS = 4.0


def transport_to(source, target, rng, count):
    n = min(count, len(source), len(target))
    src = source[rng.choice(len(source), n, replace=False)]
    tgt = target[rng.choice(len(target), n, replace=False)]
    rows, cols = linear_sum_assignment(cdist(src, tgt, metric="sqeuclidean"))
    ordered = np.empty_like(tgt)
    ordered[rows] = tgt[cols]
    return src, ordered


def yaml_panel() -> str:
    m = MOTO
    parts = [
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{m["panel2"]}" '
        f'stroke="{m["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{m["line"]}"/>',
        f'<text x="490" y="111" fill="{m["chrome"]}" font-size="13" font-weight="700" '
        f'letter-spacing=".5">profile.yml</text>',
        f'<text x="580" y="111" fill="{m["muted"]}" font-size="11">[YAML]</text>',
        f'<rect x="996" y="94" width="132" height="24" rx="12" fill="{m["accent"]}" opacity=".14" '
        f'stroke="{m["accent"]}"/>',
        f'<text x="1062" y="111" text-anchor="middle" fill="{m["accent"]}" font-size="13" '
        f'font-weight="700">@guille2506</text>',
    ]
    row_y = 148.0
    for idx, (indent, key, value) in enumerate(YAML_ROWS, 1):
        line_num = f"{idx:2d}"
        if indent == 0:
            content = f'<tspan fill="{m["chrome"]}" font-weight="700">{html.escape(key)}:</tspan>'
            text_x = 525.0
        else:
            content = (f'<tspan fill="{m["accent"]}">{html.escape(key)}: </tspan>'
                       f'<tspan fill="{m["ink"]}">{html.escape(value)}</tspan>')
            text_x = 542.0
        parts.append(
            f'<text x="506" y="{num(row_y)}" text-anchor="end" fill="{m["muted"]}" opacity=".45" '
            f'font-size="13">{line_num}</text>')
        parts.append(
            f'<text x="{num(text_x)}" y="{num(row_y)}" font-size="13">{content}</text>')
        row_y += 21.5
    parts.append(f'<path d="M474 526H1146" stroke="{m["line"]}"/>')
    parts.append(f'<rect x="485" y="533" width="72" height="20" rx="3" fill="{m["accent"]}"/>')
    parts.append(f'<text x="521" y="547" text-anchor="middle" fill="{m["bg"]}" '
                 f'font-size="11" font-weight="700">NORMAL</text>')
    parts.append(f'<text x="569" y="547" fill="{m["ink"]}" font-size="12" font-weight="600">profile.yml</text>')
    parts.append(f'<text x="740" y="547" fill="{m["muted"]}" font-size="11">[utf-8]</text>')
    parts.append(f'<text x="1134" y="547" text-anchor="end" fill="{m["muted"]}" font-size="11">MOTO.SVG 100%</text>')
    return "".join(parts)


def render_morph(face, face_bbox, red_pts, ice_pts, dim_pts, cbbox, rng) -> None:
    m = MOTO
    fbbox = face_bbox
    face_placed = place(face, 74, 160, 330, 355, fbbox, fill=0.97)
    redL_p = place(red_pts, 74, 160, 330, 355, cbbox, fill=0.97)
    ice_p = place(ice_pts, 74, 160, 330, 355, cbbox, fill=0.97)
    dim_p = place(dim_pts, 74, 160, 330, 355, cbbox, fill=0.97)
    cat_all = np.vstack((redL_p, ice_p, dim_p))
    src, tgt = transport_to(face_placed, cat_all, rng, TRAVELLER_COUNT)

    times = [0.0, 3.0]
    frames = [src, src]
    times.extend((times[-1] + TRANSITION_SECONDS,
                  times[-1] + TRANSITION_SECONDS + HOLD_SECONDS))
    frames.extend((tgt, tgt))
    times.append(times[-1] + TRANSITION_SECONDS)
    frames.append(src)
    loop_seconds = times[-1]
    loop_duration = time_num(loop_seconds)
    key_times = ";".join(time_num(v / loop_seconds) for v in times)
    opacity_values = ";".join(["0", "0"] + ["1"] * (len(frames) - 3) + ["0"])

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{MORPH_W}" height="{MORPH_H}" viewBox="0 0 {MORPH_W} {MORPH_H}" '
        f'font-family="{MONO}" role="img" aria-labelledby="t d">',
        '<title id="t">Guillermo Illanes - perfil Readme</title>',
        '<desc id="d">Retrato 1-bit que se transforma en la mascota Maxwell.</desc>',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#02050b" '
        'flood-opacity=".28"/></filter>',
        f'<clipPath id="mclip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        f'<pattern id="bgrille" width="48" height="48" patternUnits="userSpaceOnUse">'
        f'<path d="M48 0V48H0" fill="none" stroke="{m["grid"]}" stroke-width="1" opacity="0.5"/></pattern>',
        f'<pattern id="bcross" width="192" height="192" patternUnits="userSpaceOnUse" x="88" y="62">'
        f'<path d="M90 96h12M96 90v12" fill="none" stroke="{m["accent"]}" '
        f'stroke-opacity="0.35" stroke-width="1.6"/></pattern>',
        "</defs>",
        f'<rect width="{MORPH_W}" height="{MORPH_H}" rx="18" fill="{m["bg"]}"/>',
        f'<rect width="{MORPH_W}" height="{MORPH_H}" rx="18" fill="url(#bgrille)" opacity="1"/>',
        f'<rect width="{MORPH_W}" height="{MORPH_H}" rx="18" fill="url(#bcross)"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{m["panel"]}" '
        f'stroke="{m["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{m["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#ff2633"/>'
        '<circle cx="59" cy="38" r="6" fill="#27406b"/>'
        '<circle cx="80" cy="38" r="6" fill="#1d2946"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{m["muted"]}" font-size="13" '
        f'letter-spacing=".4">vim profile.yml</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{m["panel2"]}" stroke="{m["line"]}"/>',
        f'<path d="M35 124H453" stroke="{m["line"]}"/>',
        f'<text x="49" y="111" fill="{m["chrome"]}" font-size="13" font-weight="700" '
        f'letter-spacing="1.2">VISUAL.DAT</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{m["muted"]}" font-size="11">300x340 / 1-BIT</text>',
        '<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        'M439 539h-12M439 539v-12" fill="none" stroke="{chrome}" opacity=".55"/>'
        .format(chrome=m["chrome"]),
        '<g clip-path="url(#mclip)" shape-rendering="crispEdges">',
        '<g opacity="1">',
    ]

    # capa loop: nieve de la cara completa (base visible desde t=0)
    first_centroid = tgt.mean(axis=0)
    band_ids = rng.integers(0, 94, size=len(face_placed))
    noise = rng.normal(0, 4, size=(94, 2))
    for band in range(94):
        pts = face_placed[band_ids == band]
        if not len(pts):
            continue
        centroid = pts.mean(axis=0)
        delta = (first_centroid - centroid) * 0.18 + noise[band]
        drift_positions = ["0 0", "0 0", f"{num(delta[0])} {num(delta[1])}",
                           f"{num(delta[0])} {num(delta[1])}"]
        drift_positions.extend(["0 0"] * (len(frames) - len(drift_positions)))
        drift_opacity = [".9", ".9", "0", "0"]
        drift_opacity.extend(["0"] * (len(frames) - len(drift_opacity) - 1))
        drift_opacity.append(".9")
        drift_position_values = ";".join(drift_positions)
        drift_opacity_values = ";".join(drift_opacity)
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{m["accent"]}" stroke-width="1" opacity=".9">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{loop_duration}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{drift_position_values}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" keyTimes="{key_times}" values="{drift_opacity_values}"/></path>'
        )

    # travellers: cara -> maxwell (asean cambios a rojo al llegar)
    for i in range(len(src)):
        positions = ";".join(
            f"{num(fr[i, 0])} {num(fr[i, 1])}" for fr in frames)
        parts.append(
            f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{m["accent"]}">'
            f'<animate attributeName="fill" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{m["accent"]};{m["accent"]};{m["accent"]};{m["accent"]};{m["accent"]}"/>'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{loop_duration}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{positions}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{opacity_values}"/></path>'
        )

    # nieve de hold del gato por color (dier la silueta densa durante el hold)
    hold = [
        (dim_p, m["red_dim"], "0.55"),
        (redL_p, m["accent"], "0.95"),
        (ice_p, m["ice"], "0.9"),
    ]
    for pts, color, opac in hold:
        visible = ["0"] * len(frames)
        visible[2] = opac
        visible[3] = opac
        parts.append(
            f'<path d="{particle_path(pts)}" fill="none" stroke="{color}" stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{";".join(visible)}"/></path>'
        )
    parts.append("</g>")

    # intro shimmer one-shot
    intro_ids = rng.integers(0, 60, size=len(face_placed))
    order = rng.permutation(60)
    starts = np.empty(60)
    starts[order] = np.linspace(0.05, 1.2, 60)
    for group in range(60):
        pts = face_placed[intro_ids == group]
        if not len(pts):
            continue
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{m["accent"]}" stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{num(starts[group])}s" dur=".8s" '
            'values="0;1" fill="freeze"/>'
            '<animate attributeName="opacity" begin="3.08s" dur=".12s" values="1;0" fill="freeze"/>'
            "</path>"
        )
    parts.append("</g>")

    parts.append(
        f'<text x="58" y="551" fill="{m["muted"]}" font-size="10">'
        f'PTS {len(face_placed):05d} - FS/SERPENTINE - MOTO.SVG v1</text>')
    parts.append(yaml_panel())
    parts.append("</svg>")

    out = ASSETS / "moto-morph.svg"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"{out.name}: {out.stat().st_size:,} bytes, viajeros {len(src)}")


def main() -> None:
    for required in ("face.png", "face2_cut.png", "maxwell.png"):
        if not (SOURCE / required).exists():
            raise SystemExit(f"falta la fuente: SOURCE/{required}")
    ASSETS.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    face, fbbox, face_red = face_points(rng)
    f = ASSETS / "moto-face.svg"
    f.write_text(render_face_panel(face, fbbox, face_red), encoding="utf-8")
    print(f"{f.name}: {f.stat().st_size:,} bytes, puntos cara {len(face):,} (rojo {int(face_red.sum()) if len(face_red) else 0})")

    warm_lattice, ice, dim, cbbox = cat_points()
    c = ASSETS / "moto-maxwell.svg"
    c.write_text(render_cat_panel(warm_lattice, ice, dim, cbbox), encoding="utf-8")
    print(f"{c.name}: {c.stat().st_size:,} bytes, rim {len(warm_lattice):,} / pecho {len(ice):,} / cuerpo {len(dim):,}")

    rng3 = np.random.default_rng(SEED + 100)
    render_morph(face, fbbox, warm_lattice, ice, dim, cbbox, rng3)


if __name__ == "__main__":
    main()
