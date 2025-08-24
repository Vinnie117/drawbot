import numpy as np
from io import StringIO
from skimage import measure

def _rdp(points, epsilon):
    """Douglas–Peucker simplification. points: (N,2) array. epsilon in same units."""
    if len(points) < 3:
        return points
    # Line from first to last
    p1, p2 = points[0], points[-1]
    v = p2 - p1
    v2 = np.dot(v, v)
    if v2 == 0:
        # all points equal
        dists = np.linalg.norm(points - p1, axis=1)
    else:
        # perpendicular distance
        dists = np.abs(np.cross(points - p1, v)) / np.sqrt(v2)
    idx = np.argmax(dists)
    dmax = dists[idx]
    if dmax > epsilon:
        left = _rdp(points[:idx+1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.vstack((p1, p2))

def contours_to_svg_centered_simplified(
    T,
    levels,
    paper="A4",                 # "A4" or "A5"
    orientation="portrait",     # "portrait" or "landscape"
    margin_mm=12,               # scalar, (h, v), or (l, r, t, b)
    simplify_tol_mm=0.2,        # DP tolerance in millimeters (≈ plotter resolution)
    min_poly_pts=8,             # drop tiny polylines
    sampling_stride=1,          # subsample array before tracing (e.g., 2 or 3 to reduce density)
    stroke="black",
    linewidth_pt=0.5
):
    """
    Vector (SVG) export for contour lines with page sizing (A4/A5),
    centered content, and geometry simplification for plotters.
    """
    # --- Page sizes (mm) ---
    paper_sizes_mm = {"A4": (210.0, 297.0), "A5": (148.0, 210.0)}
    if paper not in paper_sizes_mm:
        raise ValueError("paper must be 'A4' or 'A5'")
    w_mm, h_mm = paper_sizes_mm[paper]
    if orientation.lower() == "landscape":
        w_mm, h_mm = h_mm, w_mm

    # --- Margins ---
    def _parse_margins(m):
        if isinstance(m, (int, float)):
            return m, m, m, m
        if len(m) == 2:
            h, v = m; return h, h, v, v
        if len(m) == 4:
            l, r, t, b = m; return l, r, t, b
        raise ValueError("margin_mm must be a number, (h,v), or (l,r,t,b)")
    l_mm, r_mm, t_mm, b_mm = _parse_margins(margin_mm)

    content_w_mm = max(0.0, w_mm - l_mm - r_mm)
    content_h_mm = max(0.0, h_mm - t_mm - b_mm)

    # --- Optionally downsample the scalar field to reduce contour complexity ---
    T0 = T[::sampling_stride, ::sampling_stride]
    H, W = T0.shape
    arr_aspect = W / H
    box_aspect = content_w_mm / content_h_mm

    # Compute the plot box (centered) that preserves aspect
    if arr_aspect > box_aspect:
        plot_w_mm = content_w_mm
        plot_h_mm = content_w_mm / arr_aspect
    else:
        plot_h_mm = content_h_mm
        plot_w_mm = content_h_mm * arr_aspect
    offset_x_mm = l_mm + (content_w_mm - plot_w_mm) / 2
    offset_y_mm = b_mm + (content_h_mm - plot_h_mm) / 2

    # --- Generate contours as polylines (in array coords) ---
    # levels can be int (count) or array-like; skimage needs list of floats
    if np.isscalar(levels):
        vmin, vmax = float(np.nanmin(T0)), float(np.nanmax(T0))
        lvls = np.linspace(vmin, vmax, int(levels))
    else:
        lvls = np.asarray(levels, dtype=float)

    polylines = []
    for lv in lvls:
        cs = measure.find_contours(T0, lv)  # each c: array of (row, col) pairs in pixel coords
        for c in cs:
            # map (row, col) → (x_mm, y_mm) inside the centered plot box
            # c[:,0] in [0..H), c[:,1] in [0..W)
            x = (c[:,1] / (W - 1)) * plot_w_mm + offset_x_mm
            y = (1.0 - c[:,0] / (H - 1)) * plot_h_mm + offset_y_mm  # flip y to match SVG coords (top→bottom)
            poly = np.column_stack([x, y])

            # simplify
            if simplify_tol_mm > 0:
                poly = _rdp(poly, simplify_tol_mm)

            if len(poly) >= max(2, min_poly_pts):
                polylines.append(poly)

    # --- Build SVG ---
    # Header with explicit page size in mm
    buf = StringIO()
    buf.write(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm}mm" height="{h_mm}mm" '
        f'viewBox="0 0 {w_mm} {h_mm}">\n'
        f'  <rect x="0" y="0" width="{w_mm}" height="{h_mm}" fill="white"/>\n'
    )
    # Optional: draw a guides rectangle for margins (comment out if not needed)
    # buf.write(f'  <rect x="{offset_x_mm}" y="{offset_y_mm}" width="{plot_w_mm}" height="{plot_h_mm}" '
    #           f'fill="none" stroke="none"/>\n')

    # paths
    lw_mm = linewidth_pt * 0.352777778  # 1 pt = 1/72 in; 1 in = 25.4 mm → 1 pt ≈ 0.35278 mm
    for poly in polylines:
        d = f"M {poly[0,0]:.3f},{poly[0,1]:.3f} " + " ".join(f"L {x:.3f},{y:.3f}" for x, y in poly[1:])
        buf.write(f'  <path d="{d}" fill="none" stroke="{stroke}" stroke-width="{lw_mm:.3f}"/>\n')

    buf.write('</svg>\n')
    return buf.getvalue()



from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage import measure

def drop_tiny_contours_svg(
    T, levels,
    paper="A4", orientation="portrait",
    margin_mm=10, stroke="black",
    linewidth=0.25, scale_pct=100,
    min_path_len_mm=2.0,          # drop fragments shorter than this
    simplify_tol_mm=0.2,          # Douglas–Peucker tolerance
):
    paper_sizes_mm = {"A4": (210.0, 297.0), "A5": (148.0, 210.0)}
    if paper not in paper_sizes_mm:
        raise ValueError("paper must be 'A4' or 'A5'")
    w_mm, h_mm = paper_sizes_mm[paper]
    if orientation.lower() == "landscape":
        w_mm, h_mm = h_mm, w_mm

    def _parse_margins(m):
        if isinstance(m, (int, float)): return (m, m, m, m)
        if len(m) == 2: h, v = m; return (h, h, v, v)
        if len(m) == 4: l, r, t, b = m; return (l, r, t, b)
        raise ValueError("margin_mm must be a number, (h,v), or (l,r,t,b)")
    l_mm, r_mm, t_mm, b_mm = _parse_margins(margin_mm)

    content_w_mm = max(0.0, w_mm - l_mm - r_mm)
    content_h_mm = max(0.0, h_mm - t_mm - b_mm)

    arr_h, arr_w = T.shape
    arr_aspect = arr_w / arr_h
    content_aspect = content_w_mm / content_h_mm
    if arr_aspect > content_aspect:
        plot_w_mm = content_w_mm
        plot_h_mm = content_w_mm / arr_aspect
    else:
        plot_h_mm = content_h_mm
        plot_w_mm = content_h_mm * arr_aspect

    s = max(0.0, float(scale_pct) / 100.0)
    plot_w_mm *= s
    plot_h_mm *= s

    offset_x_mm = l_mm + (content_w_mm - plot_w_mm) / 2
    offset_y_mm = b_mm + (content_h_mm - plot_h_mm) / 2

    # mm/px (same in x & y because we kept aspect)
    mm_per_px = plot_w_mm / arr_w if arr_w else 1.0
    tol_px = (simplify_tol_mm / mm_per_px) if simplify_tol_mm else 0.0
    min_len_px = (min_path_len_mm / mm_per_px) if min_path_len_mm else 0.0

    # Build polylines with skimage (cheaper SVG than mpl.contour)
    polylines = []
    for L in levels:
        for seg in measure.find_contours(T, L):
            # seg is (row, col); convert to (x, y) with "image" origin
            xy = np.column_stack([seg[:, 1], (arr_h - 1) - seg[:, 0]])
            # simplify
            if tol_px > 0:
                xy = measure.approximate_polygon(xy, tolerance=tol_px)
            if xy.shape[0] < 2:
                continue
            # length filter
            if min_len_px > 0:
                d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1)).sum()
                if d < min_len_px:
                    continue
            polylines.append(xy)

    # Figure / axes sized to paper with centered content
    mm_to_in = 1.0 / 25.4
    w_in, h_in = w_mm * mm_to_in, h_mm * mm_to_in
    left_frac   = offset_x_mm / w_mm
    bottom_frac = offset_y_mm / h_mm
    ax_w_frac   = plot_w_mm / w_mm
    ax_h_frac   = plot_h_mm / h_mm

    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([left_frac, bottom_frac, ax_w_frac, ax_h_frac])
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(0, arr_w)
    ax.set_ylim(0, arr_h)

    lc = LineCollection(polylines, colors=stroke, linewidths=linewidth)
    ax.add_collection(lc)

    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return buf.getvalue().decode("utf-8")
