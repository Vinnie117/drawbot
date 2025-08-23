import cv2
import numpy as np
from typing import Union
import time
from utils.image_helper import apply_fixed_threshold, apply_otsu_threshold
from algos.traveling_salesman import approach_greedy
from numba_progress import ProgressBar
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

def create_drawing(img_path: str, 
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
        if not isinstance(points_sampled, int):
            raise ValueError("When method is 'fill', an integer 'points_sampled' must be provided.")
        if colour_sampled not in {"black", "white"}:
            raise ValueError("When method is 'fill', 'color_sampled' must be either 'black' or 'white'.")
        
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
        # combine all contour points into one list
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
        black_pixels = np.column_stack((x_coords, y_coords))
        sample_count = points_sampled  # adjust based on performance
        if len(black_pixels) > sample_count:
            indices = np.random.choice(len(black_pixels), sample_count, replace=False)
            black_pixels = black_pixels[indices]

        points = black_pixels

    points = np.array(points)
    num_points = len(points)
    print(f"Computing {num_points} points")

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
