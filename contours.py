from io import BytesIO
import matplotlib.pyplot as plt
import skfmm
import numpy as np
from skimage import io

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


CONTOURS = 1000


# 1) Load image as grayscale
image = io.imread(r"images\input\foto.jpeg", as_gray=True)
#image = transform.rescale(image, 0.8, anti_aliasing=False)

# 2) Initialize the level set function
phi = np.ones_like(image)
cy, cx = np.array(phi.shape) // 2
phi[cy, cx] = 0  # set the image center as starting point

# 3) Compute the travel time using the Fast Marching Method
T = skfmm.travel_time(phi, image)

# 4) Define contour levels
contour_levels = np.linspace(T.min(), T.max(), CONTOURS)

#After you compute T and contour_levels:
svg = contours_to_svg(T, contour_levels, stroke="black", linewidth=0.5, figsize=8, pad_inches=0)
save_svg(svg, "images/output/test.svg")