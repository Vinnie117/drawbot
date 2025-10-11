import skfmm
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage import io, transform
from utils.draw import contours_to_centered_svg, contours_to_centered_svg2
from svg import contours_to_svg, save_svg, contours_to_svg_paged
from svg_simple import contours_to_svg_centered_simplified, drop_tiny_contours_svg
from skimage import measure
from io import BytesIO

''' Problem: too many separate (small?) parts
Option 1: Reduce contour count / number of levels

Option 2: Downsample before contouring (maybe dark parts more??)
T_small = transform.rescale(T, 0.5, anti_aliasing=True, preserve_range=True)
contour_levels = np.linspace(T_small.min(), T_small.max(), 200)

Option 3: Drop tiny contours and simplify with Douglas-Peucker

Option 4: Axidraw skip heavy optimization: ad.options.reordering = 4

'''

CONTOURS = 300


def contours_to_svg_centered(
    T,
    levels,
    paper="A4",                 # "A3", "A4", or "A5"
    orientation="portrait",     # "portrait" or "landscape"
    margin_mm=10,               # scalar, (h, v), or (l, r, t, b)
    stroke="black",
    linewidth=0.5
):

    """
    Render contour lines as an SVG sized exactly to A3/A4/A5,
    with the plot centered on the page inside given margins.
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

    # Make backgrounds transparent
    fig.patch.set_alpha(0.0)     # Figure background
    ax.patch.set_alpha(0.0)      # Axes background

    # Export to SVG

    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return buf.getvalue().decode("utf-8")

from skimage import filters, restoration, morphology
# 1) Load image as grayscale
image = io.imread(r"images\input\hokusai_wave.jpg", as_gray=True)
image = transform.rescale(image, 1, anti_aliasing=False)

# Light smoothing before fast marching
image_smooth = filters.gaussian(image, sigma=1.0, preserve_range=True)

# (optional) stronger but edge-preserving:
# image_smooth = restoration.denoise_tv_chambolle(image, weight=0.05)

phi = np.ones_like(image_smooth)
cy, cx = np.array(phi.shape) // 2
phi[cy, cx] = 0

T = skfmm.travel_time(phi, image_smooth)

# 4) Define contour levels
contour_levels = np.linspace(T.min(), T.max(), CONTOURS)

#After you compute T and contour_levels:
# svg = contours_to_svg(T, contour_levels, stroke="black", linewidth=0.5, figsize=8, pad_inches=0)

# svg = contours_to_svg_centered_simplified(T, contour_levels, "A5", margin_mm=12, simplify_tol_mm=0.2, min_poly_pts=6, sampling_stride=1)
# svg = drop_tiny_contours_svg(T, contour_levels, "A5", margin_mm=12, simplify_tol_mm=0.2, min_path_len_mm=2.0)


#### Do this to get rid of minimal contour points at the border
mask = np.zeros_like(T, dtype=bool)
mask[1:-1, 1:-1] = True  # keep only interior
T_masked = np.where(mask, T, np.nan)

svg = contours_to_svg_centered (T_masked, contour_levels, "A4", margin_mm=25, orientation="landscape")

save_svg(svg, "bot/test_wave.svg")