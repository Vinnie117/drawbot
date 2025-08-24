from io import BytesIO
import matplotlib.pyplot as plt

def contours_to_svg(T, levels, stroke="black", linewidth=0.3, figsize=8, pad_inches=0):
    """
    Render contour lines of the travel-time surface `T` as an SVG string.

    Parameters
    ----------
    T : 2D ndarray
        Travel time surface (e.g., from skfmm.travel_time).
    levels : int or 1D array-like
        Same as Matplotlib's `levels` in `plt.contour` (e.g., your contour_levels).
    stroke : str
        Line color in the SVG.
    linewidth : float
        Line width (in points).
    figsize : float
        Figure size in inches (square canvas).
    pad_inches : float
        Padding around the figure when exporting.

    Returns
    -------
    svg_text : str
        The SVG markup as a string.
    """
    # Create a clean, frameless figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(x=0, y=0)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Draw contours (origin='image' to match your previous plot orientation)
    ax.contour(T, levels=levels, colors=stroke, linewidths=linewidth, origin="image")

    # Export to SVG (in-memory)
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return buf.getvalue().decode("utf-8")


def save_svg(svg_text, path):
    """Write the SVG string to disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_text)



from io import BytesIO
import matplotlib.pyplot as plt

def contours_to_svg_paged(
    T,
    levels,
    paper="A4",                 # "A4" or "A5"
    orientation="portrait",     # "portrait" or "landscape"
    margin_mm=10,               # scalar, (h, v), or (left, right, top, bottom)
    stroke="black",
    linewidth=0.25
):
    """
    Render contour lines as an SVG sized exactly to A4/A5 with real margins.

    Parameters
    ----------
    T : 2D ndarray
        Travel-time surface (e.g., from skfmm.travel_time).
    levels : int or 1D array-like
        Contour levels (same as Matplotlib).
    paper : {"A4","A5"}
        Target paper size.
    orientation : {"portrait","landscape"}
        Page orientation.
    margin_mm : number | (h, v) | (l, r, t, b)
        Page margins in millimeters. If one value given, all sides use it.
        If two values, interpreted as (horizontal, vertical).
        If four values, interpreted as (left, right, top, bottom).
    stroke : str
        Line color.
    linewidth : float
        Line width (points).

    Returns
    -------
    svg_text : str
        SVG markup sized to the chosen paper with margins preserved.
    """
    # --- Page sizes (mm) ---
    paper_sizes_mm = {
        "A4": (210.0, 297.0),
        "A5": (148.0, 210.0),
    }
    if paper not in paper_sizes_mm:
        raise ValueError("paper must be 'A4' or 'A5'")

    w_mm, h_mm = paper_sizes_mm[paper]
    if orientation.lower() == "landscape":
        w_mm, h_mm = h_mm, w_mm

    # --- Margins handling ---
    def _parse_margins(m):
        if isinstance(m, (int, float)):
            return m, m, m, m                         # l, r, t, b
        if len(m) == 2:
            h, v = m
            return h, h, v, v
        if len(m) == 4:
            l, r, t, b = m
            return l, r, t, b
        raise ValueError("margin_mm must be a number, (h,v), or (l,r,t,b)")

    l_mm, r_mm, t_mm, b_mm = _parse_margins(margin_mm)

    # Clamp margins so content area stays positive
    l_mm = max(0.0, min(l_mm, w_mm/2))
    r_mm = max(0.0, min(r_mm, w_mm/2))
    t_mm = max(0.0, min(t_mm, h_mm/2))
    b_mm = max(0.0, min(b_mm, h_mm/2))

    # Fractions of the figure (0..1) for the Axes box
    left_frac   = l_mm / w_mm
    right_frac  = r_mm / w_mm
    top_frac    = t_mm / h_mm
    bottom_frac = b_mm / h_mm

    ax_width_frac  = max(0.0, 1.0 - left_frac - right_frac)
    ax_height_frac = max(0.0, 1.0 - top_frac - bottom_frac)

    # Convert to inches for Matplotlib figure size (1 in = 25.4 mm)
    mm_to_in = 1.0 / 25.4
    w_in, h_in = w_mm * mm_to_in, h_mm * mm_to_in

    # Create exact-size SVG page (no tight bbox — preserves page dimensions)
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([left_frac, bottom_frac, ax_width_frac, ax_height_frac])
    ax.set_axis_off()
    ax.set_aspect("equal")  # keep geometry square
    ax.contour(T, levels=levels, colors=stroke, linewidths=linewidth, origin="image")

    # Export SVG (keep full page size; no 'tight' bbox)
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return buf.getvalue().decode("utf-8")


# Optional: helper to save and auto-create folders
import os
def save_svg(svg_text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_text)


from io import BytesIO
import matplotlib.pyplot as plt
import os

def contours_to_svg_centered(
    T,
    levels,
    paper="A4",                 # "A4" or "A5"
    orientation="portrait",     # "portrait" or "landscape"
    margin_mm=10,               # scalar, (h, v), or (l, r, t, b)
    stroke="black",
    linewidth=0.5
):
    """
    Render contour lines as an SVG sized exactly to A4/A5,
    with the plot centered on the page inside given margins.
    """
    # --- Page sizes (mm) ---
    paper_sizes_mm = {
        "A4": (210.0, 297.0),
        "A5": (148.0, 210.0),
    }
    if paper not in paper_sizes_mm:
        raise ValueError("paper must be 'A4' or 'A5'")

    w_mm, h_mm = paper_sizes_mm[paper]
    if orientation.lower() == "landscape":
        w_mm, h_mm = h_mm, w_mm

    # --- Margins handling ---
    def _parse_margins(m):
        if isinstance(m, (int, float)):
            return m, m, m, m
        if len(m) == 2:
            h, v = m
            return h, h, v, v
        if len(m) == 4:
            l, r, t, b = m
            return l, r, t, b
        raise ValueError("margin_mm must be a number, (h,v), or (l,r,t,b)")

    l_mm, r_mm, t_mm, b_mm = _parse_margins(margin_mm)

    # Content area (mm)
    content_w_mm = max(0.0, w_mm - l_mm - r_mm)
    content_h_mm = max(0.0, h_mm - t_mm - b_mm)

    # --- Aspect ratio handling ---
    arr_h, arr_w = T.shape
    arr_aspect = arr_w / arr_h
    content_aspect = content_w_mm / content_h_mm

    if arr_aspect > content_aspect:
        # Limited by width
        plot_w_mm = content_w_mm
        plot_h_mm = content_w_mm / arr_aspect
    else:
        # Limited by height
        plot_h_mm = content_h_mm
        plot_w_mm = content_h_mm * arr_aspect

    # Center inside the content box
    offset_x_mm = l_mm + (content_w_mm - plot_w_mm) / 2
    offset_y_mm = b_mm + (content_h_mm - plot_h_mm) / 2

    # Convert to inches
    mm_to_in = 1.0 / 25.4
    w_in, h_in = w_mm * mm_to_in, h_mm * mm_to_in

    # Fractions for add_axes
    left_frac   = offset_x_mm / w_mm
    bottom_frac = offset_y_mm / h_mm
    ax_width_frac  = plot_w_mm / w_mm
    ax_height_frac = plot_h_mm / h_mm

    # --- Plot ---
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([left_frac, bottom_frac, ax_width_frac, ax_height_frac])
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.contour(T, levels=levels, colors=stroke, linewidths=linewidth, origin="image")

    # Export to SVG
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return buf.getvalue().decode("utf-8")


def save_svg(svg_text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_text)


from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage import measure

def simple_contours_to_svg_centered(
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
