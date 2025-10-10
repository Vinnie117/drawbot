import matplotlib.pyplot as plt
from utils.draw import create_drawing, path_to_centered_svg
import cv2
from datetime import datetime
import os
import yaml
from svg import contours_to_svg_centered, export_svg, export_png, build_centered_contour_axes, save_svg
from skimage import io
import numpy as np

# Load configs from YAML file
with open("config_local.yaml", "r") as f:
    yaml_data = yaml.safe_load(f)
configs = yaml_data.get("style", {})

# Unpack file paths
paths = yaml_data["paths"]
ROOT_FOLDER = paths["ROOT_FOLDER"]
INPUT_FOLDER = paths["INPUT_FOLDER"]
OUTPUT_FOLDER = paths["OUTPUT_FOLDER"]
BASE_IMAGE = paths["BASE_IMAGE"]
BASE_IMAGE_FILE = paths["BASE_IMAGE_FILE"]


# Prioritize whichever key is actually present
if "contours" in configs:
    active_style_config = configs["contours"]
    style_config_type = "contours"
elif "greedy_one_line" in configs:
    active_style_config = configs["greedy_one_line"]
    style_config_type = "greedy_one_line"
else:
    active_style_config = None
    style_config_type = None

print(f"Active config type: {style_config_type}")
print(active_style_config)


img_path=os.path.join(ROOT_FOLDER, INPUT_FOLDER, BASE_IMAGE_FILE)
for config in active_style_config:
    
    # Timestamp per run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # create folders to store results
    base_output_path = os.path.join(ROOT_FOLDER, OUTPUT_FOLDER, BASE_IMAGE)
    cv_output_path = os.path.join(base_output_path, 'cv')
    svg_output_path = os.path.join(base_output_path, 'svg')
    plt_output_path = os.path.join(base_output_path, 'plt')
    os.makedirs(cv_output_path, exist_ok=True)
    os.makedirs(plt_output_path, exist_ok=True)
    os.makedirs(svg_output_path, exist_ok=True)


    # Create the drawing
    test_image, stroke_coords = create_drawing(style_config_type, config, img_path)


    cv_filename = f'{BASE_IMAGE}_{timestamp}.jpg'
    cv_filepath = os.path.join(cv_output_path, cv_filename)
    cv2.imwrite(cv_filepath, test_image)
    height, width = test_image.shape

    svg_filename = f'{BASE_IMAGE}_{timestamp}.svg'
    svg_filepath = os.path.join(svg_output_path, svg_filename)
    h, w = test_image.shape[:2]


    h_now = datetime.now().strftime('%Y%m%d_%H%M')
    cv_filename  = f'{BASE_IMAGE}_{h_now}.png'
    cv_filepath  = os.path.join(cv_output_path,  cv_filename)

    svg_filename = f'{BASE_IMAGE}_{h_now}.svg'
    svg_filepath = os.path.join(svg_output_path, svg_filename)

    plt_filename = f'{BASE_IMAGE}_{h_now}.png'
    plt_filepath = os.path.join(plt_output_path, plt_filename)


    if style_config_type == "contours":
        # Unpack: for contours, create_drawing returns (T, contour_levels)
        T = test_image
        contour_levels = stroke_coords

        svg = contours_to_svg_centered(T, contour_levels, 
                                       paper=config["FORMAT"], 
                                       margin_mm=config["MARGIN_MM"], 
                                       orientation=config["ORIENTATION"])
        save_svg(svg, svg_filepath)
        fig, ax = build_centered_contour_axes(T, contour_levels, 
                                              paper=config["FORMAT"], 
                                              margin_mm=config["MARGIN_MM"], 
                                              orientation=config["ORIENTATION"])
        ax.set_title(plt_filename)
        export_png(fig, plt_filepath, dpi=300)
        plt.close(fig)


    if style_config_type == "greedy_one_line":
        path_to_centered_svg(stroke_coords, width, height, svg_filepath, paper="A5", portrait=True)

        plt.imshow(test_image, cmap='gray')
        plt.axis('off')
        plt.title(BASE_IMAGE_FILE)

        # Get current axes
        ax = plt.gca()

        # Add left-aligned text relative to the image (axes)
        annotation = f"RESIZE_PCT: {config['RESIZE_PCT']}\n" \
                    f"THRESHOLD: {config['THRESHOLD']}\n" \
                    f"POINTS_SAMPLED: {config['POINTS_SAMPLED']}\n" \
                    f"METHOD: {config['METHOD']}\n" \
                    f"COLOUR_SAMPLED: {config['COLOUR_SAMPLED']}"

        ax.text(0.0, -0.1, annotation,
                transform=ax.transAxes,
                ha='left', va='top', fontsize=10)
        
        # Right-aligned annotation (same y-coordinate)
        smooth_config = config['SMOOTH']
        if smooth_config is None:
            smooth_annotation = "SMOOTH: NONE"
        else:
            smooth_annotation = "SMOOTH:\n" + "\n".join(
                [f"  {k}: {v}" for k, v in smooth_config.items()]
            )
        ax.text(1.0, -0.1, smooth_annotation,
                transform=ax.transAxes,
                ha='right', va='top', fontsize=10)
        
        plt.subplots_adjust(bottom=0.25)  # Add space below for the text

        plt_filename = f'{BASE_IMAGE}_{timestamp}.png'
        plt_filepath = os.path.join(plt_output_path, plt_filename)
        plt.savefig(plt_filepath, dpi=300, bbox_inches='tight')
        plt.clf()