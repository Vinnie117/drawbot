# make_smiley_grid_a4.py
# Creates smiley_grid_a4.svg: 10x10 grid of "smiley face" sketches on A4 page, transparent background.

import json, os, urllib.parse, urllib.request

CATEGORY = "smiley face"
ROWS = COLS = 10
ONLY_RECOGNIZED = True
NDJSON_PATH = "smiley_face.ndjson"
OUT_SVG = "smiley_grid_a4.svg"

# --- Page & cell sizes ---
# A4 in mm: 210 x 297. We’ll use mm units directly.
PAGE_W_MM, PAGE_H_MM = 210, 297
CELL_MM = 20        # cell size in mm
MARGIN_MM = 2       # margin inside each cell in mm
STROKE_WIDTH_MM = 0.5
STROKE_COLOR = "#000000"

def download_simplified_ndjson(category=CATEGORY, local_path=NDJSON_PATH):
    if os.path.exists(local_path):
        return local_path
    base = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
    url = base + urllib.parse.quote(category) + ".ndjson"
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, local_path)
    return local_path

def norm_bbox(strokes):
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = max(1, maxx - minx)
    h = max(1, maxy - miny)
    return minx, miny, w, h

def stroke_to_points(xlist, ylist, ox, oy, scale, minx, miny):
    pts = []
    for x, y in zip(xlist, ylist):
        X = ox + (x - minx) * scale
        Y = oy + (y - miny) * scale
        pts.append(f"{X:.2f},{Y:.2f}")
    return " ".join(pts)

def render_cell_svg(strokes, cell_x, cell_y, size=CELL_MM, margin=MARGIN_MM):
    minx, miny, w, h = norm_bbox(strokes)
    S = size - 2 * margin
    scale = S / max(w, h)
    drawing_w, drawing_h = w * scale, h * scale
    ox = cell_x + margin + (S - drawing_w) / 2
    oy = cell_y + margin + (S - drawing_h) / 2

    parts = []
    for xlist, ylist in strokes:
        if len(xlist) < 2:
            continue
        pts = stroke_to_points(xlist, ylist, ox, oy, scale, minx, miny)
        parts.append(
            f'<polyline points="{pts}" fill="none" '
            f'stroke="{STROKE_COLOR}" stroke-width="{STROKE_WIDTH_MM}mm" '
            f'stroke-linecap="round" stroke-linejoin="round" />'
        )
    return "\n".join(parts)

def main():
    ndjson_file = download_simplified_ndjson()
    drawings = []
    with open(ndjson_file, "r") as f:
        for line in f:
            rec = json.loads(line)
            if ONLY_RECOGNIZED and not rec.get("recognized", False):
                continue
            drawings.append(rec["drawing"])
            if len(drawings) >= ROWS * COLS:
                break

    width = COLS * CELL_MM
    height = ROWS * CELL_MM

    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{PAGE_W_MM}mm" height="{PAGE_H_MM}mm" '
        f'viewBox="0 0 {width} {height}">\n'
    )
    svg_parts = [svg_header]  # no background rectangle → transparent

    idx = 0
    for r in range(ROWS):
        for c in range(COLS):
            if idx < len(drawings):
                x = c * CELL_MM
                y = r * CELL_MM
                svg_parts.append(render_cell_svg(drawings[idx], x, y))
            idx += 1

    svg_parts.append("</svg>\n")

    with open(OUT_SVG, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

    print(f"Wrote {OUT_SVG} ({PAGE_W_MM}×{PAGE_H_MM} mm A4, transparent background).")

if __name__ == "__main__":
    main()
