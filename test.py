import skfmm
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage import io, transform
from utils.draw import contours_to_centered_svg, contours_to_centered_svg2
from svg import contours_to_svg, save_svg, contours_to_svg_paged, contours_to_svg_centered
from svg_simple import contours_to_svg_centered_simplified, drop_tiny_contours_svg

'''
Option 1: Reduce contour count
Option 2: Downsample before contouring (maybe dark parts more??)
Option 3: Drop tiny contours and simplify with Douglas-Peucker
Option 4: Axidraw skiü heavy optimization: ad.options.reordering = 0

'''

CONTOURS = 1000


# 1) Load image as grayscale
image = io.imread(r"images\input\foto.jpeg", as_gray=True)
image = transform.rescale(image, 0.5, anti_aliasing=False)

# 2) Initialize the level set function
phi = np.ones_like(image)
cy, cx = np.array(phi.shape) // 2
phi[cy, cx] = 0  # set the image center as starting point

# 3) Compute the travel time using the Fast Marching Method
T = skfmm.travel_time(phi, image)

# 4) Define contour levels
contour_levels = np.linspace(T.min(), T.max(), CONTOURS)

#After you compute T and contour_levels:
# svg = contours_to_svg(T, contour_levels, stroke="black", linewidth=0.5, figsize=8, pad_inches=0)
svg = contours_to_svg_centered_simplified(T, contour_levels, "A5", margin_mm=12, simplify_tol_mm=0.2, min_poly_pts=6, sampling_stride=1)


save_svg(svg, "bot/test2.svg")