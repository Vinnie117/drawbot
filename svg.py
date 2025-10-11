from io import BytesIO
import os
import matplotlib.pyplot as plt

# Optional: helper to save and auto-create folders
def save_svg(svg_text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_text)

def export_svg(fig):
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches=None, pad_inches=0)
    return buf.getvalue().decode("utf-8")

def export_png(fig, path, dpi=300, facecolor="white"):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.3, facecolor=facecolor)


def contours_to_svg_centered(*args, **kwargs):
    fig, ax = build_centered_contour_axes(*args, **kwargs)
    
    # Make backgrounds transparent
    fig.patch.set_alpha(0.0)     # Figure background
    ax.patch.set_alpha(0.0)      # Axes background

    svg_text = export_svg(fig)
    plt.close(fig)
    return svg_text


def build_centered_contour_axes(
    T,
    levels,
    paper="A4",
    orientation="portrait",
    margin_mm=25,
    stroke="black",
    linewidth=0.5,
):
    """
    Build a Matplotlib (fig, ax) layout centered on an A3/A4/A5 paper size.

    Returns
    -------
    fig, ax : tuple
        Matplotlib Figure and Axes objects, with contours already drawn.
    """

    # --- Page sizes (mm) ---
    paper_sizes_mm = {
        "A3": (297.0, 420.0),
        "A4": (210.0, 297.0),
        "A5": (148.0, 210.0),
    }
    if paper not in paper_sizes_mm:
        raise ValueError("paper must be 'A3', 'A4' or 'A5'")

    w_mm, h_mm = paper_sizes_mm[paper]
    if orientation.lower() == "landscape":
        w_mm, h_mm = h_mm, w_mm

    # --- Margins ---
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

    # --- Compute content area ---
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

    # Center inside content box
    offset_x_mm = l_mm + (content_w_mm - plot_w_mm) / 2
    offset_y_mm = b_mm + (content_h_mm - plot_h_mm) / 2

    # --- Convert to inches ---
    mm_to_in = 1.0 / 25.4
    w_in, h_in = w_mm * mm_to_in, h_mm * mm_to_in

    # Fractions for add_axes
    left_frac   = offset_x_mm / w_mm
    bottom_frac = offset_y_mm / h_mm
    ax_width_frac  = plot_w_mm / w_mm
    ax_height_frac = plot_h_mm / h_mm

    # --- Create figure and axes ---
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([left_frac, bottom_frac, ax_width_frac, ax_height_frac])
    ax.set_axis_off()
    ax.set_aspect("equal")

    # --- Draw the contours (correct orientation) ---
    ax.contour(T, levels=levels, colors=stroke, linewidths=linewidth, origin="image")

    return fig, ax

