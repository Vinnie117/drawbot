import cv2
import numpy as np
from typing import Union
import time
from utils.image_helper import apply_fixed_threshold, apply_otsu_threshold
from algos.traveling_salesman import approach_greedy
from numba_progress import ProgressBar
from scipy.signal import savgol_filter
from skimage import io, transform
import skfmm
from skimage import filters
import re

def create_drawing(style_config_type, style_config, img_path):

    drawing = None
    stroke_coords = None
    
    if style_config_type == "greedy_one_line":

        drawing, stroke_coords = create_greedy_one_line_drawing(
            img_path=img_path,
            resize_pct=style_config["RESIZE_PCT"],
            threshold=style_config["THRESHOLD"],
            method=style_config["METHOD"],
            points_sampled=style_config["POINTS_SAMPLED"],
            colour_sampled=style_config["COLOUR_SAMPLED"],
            smoothing=style_config["SMOOTH"]
            )
        
    if style_config_type == "contours":

        drawing, stroke_coords = create_contours_drawing(
            img_path=img_path,
            contours=style_config["LEVELS"],
            smooth_no_dots=style_config["SMOOTHING_NO_DOTS"]
        )
        
    return drawing, stroke_coords


def create_contours_drawing(contours: int, img_path, smooth_no_dots):
    contours = contours

    # 1) Load image as grayscale
    image = io.imread(img_path, as_gray=True)
    image = transform.rescale(image, 1, anti_aliasing=False)

    if smooth_no_dots == True:
        # Light smoothing before fast marching
        image = filters.gaussian(image, sigma=1.0, preserve_range=True)

        # (optional) stronger but edge-preserving:
        # image_smooth = restoration.denoise_tv_chambolle(image, weight=0.05)

    # 2) Initialize the level set function
    phi = np.ones_like(image)
    cy, cx = np.array(phi.shape) // 2
    phi[cy, cx] = 0  # set the image center as starting point

    # 3) Compute the travel time using the Fast Marching Method
    T = skfmm.travel_time(phi, image)

    # 4) Define contour levels
    contour_levels = np.linspace(T.min(), T.max(), contours)

    #### Do this to get rid of minimal contour points at the border
    mask = np.zeros_like(T, dtype=bool)
    mask[1:-1, 1:-1] = True  # keep only interior
    T_masked = np.where(mask, T, np.nan)

    return T_masked, contour_levels



def create_greedy_one_line_drawing(img_path: str, 
                   resize_pct: int,
                   threshold: Union[int, str], 
                   method: str,
                   points_sampled: int = None,
                   colour_sampled: str = None,
                   smoothing: dict = None):
    
    # validation of function call
    if method not in {"contour", "fill"}:
        raise ValueError("method must be either 'contour' or 'fill'")
    
    valid_threshold = (isinstance(threshold, int) and 0 <= threshold <= 255) or threshold == 'otsu'
    if not valid_threshold:
        raise ValueError("Threshold must be an integer (0-255) or 'otsu'")
    
    # Validate fill_value if method is 'fill'
    if method == "fill":
        # points_sampled can be int, float (0,1], or 'N%'
        if points_sampled is None:
            raise ValueError("When method is 'fill', 'points_sampled' must be provided.")
        if colour_sampled not in {"black", "white"}:
            raise ValueError("When method is 'fill', 'colour_sampled' must be either 'black' or 'white'.")

        
    # Check smoothing dict
    if smoothing is not None:
        allowed_keys = {"window_length", "poly_order"}
        for key in smoothing:
            if key not in allowed_keys:
                raise ValueError(f"Invalid parameter: '{key}'. Allowed keys are: {allowed_keys}")
        window_length = smoothing['window_length']
        poly_order = smoothing['poly_order']
        if not isinstance(window_length, int) or window_length < 3 or window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer ≥ 3")
        if not isinstance(poly_order, int) or poly_order < 0 or poly_order >= window_length:
            raise ValueError("poly_order must be a non-negative integer < window_length")


    drawing = None
    points = []

    # Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{img_path}' could not be loaded.")
    # Print original image size
    original_height, original_width = image.shape[:2]
    print(f"Original image size: {original_width} x {original_height}")

    # Define scale factor as a percentage (e.g., 50% of original size)
    scale_percent = resize_pct
    # Convert percent to scaling factor
    scale = scale_percent / 100.0
    # Compute new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    # Choose interpolation method
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)


    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)


    # determine which points are black and white
    match threshold:
        case int() as value if 0 <= value <= 255:
            binary  = apply_fixed_threshold(gray, threshold=value)
        case 'otsu':
            binary  = apply_otsu_threshold(gray)
    binary  = 255 - binary # Invert binary image


    if method == "contour":
        # (unchanged) collect all contour points
        fixed_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in fixed_contours:
            for pt in cnt:
                points.append(pt[0]) 
    else:
        # Sample black pixels (!=0) or white (==0)
        if colour_sampled == 'black':
            y_coords, x_coords = np.where(binary != 0)
        else:
            y_coords, x_coords = np.where(binary == 0)
        available = np.column_stack((x_coords, y_coords))
        total_available = len(available)

        # NEW: compute sample_count from int/float/'N%'
        sample_count = _parse_points_sampled(points_sampled, max_points=total_available)

        if total_available > sample_count:
            indices = np.random.choice(total_available, sample_count, replace=False)
            available = available[indices]

        points = available

    points = np.array(points)
    num_points = len(points)
    if num_points > 0:
        pct = 100 * num_points / num_points
        print(f"Computing {num_points} points (from {num_points} available = {pct:.2f}%)")
    else:
        print(f"Computing {num_points} points (no available points detected!)")

    # Calculate single strike path
    start = time.time()
    with ProgressBar(total=(num_points - 1) * num_points) as progress:  # outer loop * inner loop
        stroke_path = approach_greedy(points, num_points, progress)
    end = time.time()
    print(f"✅ {num_points} points computed in {end - start:.2f} seconds.")

    if smoothing is not None:
        ordered = np.array([points[i] for i in stroke_path])
        x_smooth = savgol_filter(ordered[:, 0], window_length, poly_order)  # e.g. window size 51, poly order 3
        y_smooth = savgol_filter(ordered[:, 1], window_length, poly_order)
        stroke_coords = list(zip(x_smooth.astype(int), y_smooth.astype(int)))
    else:
        stroke_coords = [tuple(map(int, points[i])) for i in stroke_path]

    # Draw the stroke path on a blank canvas
    drawing = np.ones_like(gray) * 255  # white background
    for i in range(1, len(stroke_coords)):
        pt1 = stroke_coords[i - 1]
        pt2 = stroke_coords[i]
        cv2.line(drawing, pt1, pt2, 0, 1)

    return drawing, stroke_coords


def path_to_centered_svg(
    stroke_coords,
    width, height,                  # inner artwork size in px/user units
    svg_path,
    stroke="black",
    stroke_width=1,
    paper="A4",                     # "A5", "A4", "A3" (or a (w_mm, h_mm) tuple)
    portrait=True,
    margin=0,                       # margin amount
    margin_unit="px",               # "px" or "mm"
    dpi=96                          # CSS px per inch; AxiDraw/inkscape default is 96
):
    """
    Create a single-root SVG sized to the chosen paper (A5/A4/A3 or custom mm tuple),
    and center your path without scaling by translating a <g>.

    - Keeps AxiDraw happy (no nested <svg>).
    - Uses mm for outer width/height (print size), and a px-based viewBox.
    - 'width'/'height' define your artwork's own coordinate system; no scaling applied.
    """

    # Paper sizes in millimeters
    PRESETS_MM = {
        "A5": (148.0, 210.0),
        "A4": (210.0, 297.0),
        "A3": (297.0, 420.0),
    }

    # Resolve paper size in mm
    if isinstance(paper, (tuple, list)) and len(paper) == 2:
        page_w_mm, page_h_mm = float(paper[0]), float(paper[1])
    else:
        key = str(paper).upper()
        if key not in PRESETS_MM:
            raise ValueError(f"Unsupported paper '{paper}'. Use 'A5', 'A4', 'A3', or (w_mm, h_mm).")
        page_w_mm, page_h_mm = PRESETS_MM[key]

    # Portrait vs landscape
    if not portrait:
        page_w_mm, page_h_mm = page_h_mm, page_w_mm

    # Unit conversions
    def mm_to_px(mm): return mm * dpi / 25.4

    page_w_px = mm_to_px(page_w_mm)
    page_h_px = mm_to_px(page_h_mm)

    # Margin handling
    margin_px = float(margin) if margin_unit == "px" else mm_to_px(float(margin))

    # Desired centered offsets
    cx = (page_w_px - width) / 2.0
    cy = (page_h_px - height) / 2.0

    # Clamp into page bounds (respect margin but never go negative / overflow)
    lower_x = max(0.0, margin_px)
    lower_y = max(0.0, margin_px)
    upper_x = max(0.0, page_w_px - width - margin_px)
    upper_y = max(0.0, page_h_px - height - margin_px)

    x = min(max(cx, lower_x), upper_x)
    y = min(max(cy, lower_y), upper_y)

    # Build path (or empty group if no coords)
    if stroke_coords:
        move = f"M {stroke_coords[0][0]} {stroke_coords[0][1]}"
        lines = " ".join(f"L {x} {y}" for (x, y) in stroke_coords[1:])
        d = f"{move} {lines}"
        path_elem = f'    <path d="{d}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" />\n'
    else:
        path_elem = ""

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'     width="{page_w_mm}mm" height="{page_h_mm}mm" '
        f'     viewBox="0 0 {page_w_px:.4f} {page_h_px:.4f}">\n'
        f'  <g transform="translate({x:.4f},{y:.4f})">\n'
        f'{path_elem}'
        f'  </g>\n'
        f'</svg>\n'
    )

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)



def _parse_points_sampled(points_sampled, max_points: int) -> int:
    """
    Accepts:
      - int (absolute count, except 1 = 100%)
      - float in (0, 1] (fraction of max_points)
      - str like '25%' (percentage of max_points)

    Returns a clamped positive int in [1, max_points].
    """
    if not isinstance(max_points, int) or max_points <= 0:
        raise ValueError("max_points must be a positive integer")

    # int count
    if isinstance(points_sampled, int):
        if points_sampled == 1:
            # Special case: interpret as 100%
            return max_points
        count = points_sampled

    # fractional float
    elif isinstance(points_sampled, float):
        if not (0 < points_sampled <= 1):
            raise ValueError("When float, points_sampled must be in (0, 1].")
        count = int(round(points_sampled * max_points))

    # percentage string like '25%'
    elif isinstance(points_sampled, str) and re.fullmatch(r"\s*(\d{1,3})\s*%\s*", points_sampled):
        p = int(re.fullmatch(r"\s*(\d{1,3})\s*%\s*", points_sampled).group(1))
        if not (0 < p <= 100):
            raise ValueError("Percentage must be between 1% and 100%.")
        count = int(round((p / 100.0) * max_points))

    else:
        raise ValueError("points_sampled must be an int, a float in (0,1], or a 'N%' string.")

    # clamp to [1, max_points]
    count = max(1, min(count, max_points))
    return count

