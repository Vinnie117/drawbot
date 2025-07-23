import matplotlib.pyplot as plt
from main import create_drawing
import cv2
from datetime import datetime
import os

# TODO
# - smooth path?
# - save the single stroke as txt

BASE_IMAGE = 'hokusai_wave'
BASE_IMAGE_FILE = BASE_IMAGE +'.jpg'
ROOT_FOLDER = 'images/'
INPUT_FOLDER = 'input/'
OUTPUT_FOLDER = 'output/'


# Define multiple configurations to run
configs = [
    {"RESIZE_PCT": 100, "THRESHOLD": 127, "METHOD": "fill", "POINTS_SAMPLED": 200000, "COLOUR_SAMPLED": "black", "SMOOTHING": None},
    {"RESIZE_PCT": 20, "THRESHOLD": 127, "METHOD": "fill", "POINTS_SAMPLED": 200000, "COLOUR_SAMPLED": "black", "SMOOTHING": None},
    {"RESIZE_PCT": 30, "THRESHOLD": 127, "METHOD": "fill", "POINTS_SAMPLED": 150000, "COLOUR_SAMPLED": "black", "SMOOTHING": None},
]


for config in configs:
    
    # Timestamp per run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # create folders to store results
    base_output_path = os.path.join(ROOT_FOLDER, OUTPUT_FOLDER, BASE_IMAGE)
    cv_output_path = os.path.join(base_output_path, 'cv')
    plt_output_path = os.path.join(base_output_path, 'plt')
    os.makedirs(cv_output_path, exist_ok=True)
    os.makedirs(plt_output_path, exist_ok=True)

    # Create the drawing
    test_image = create_drawing(
        img_path=os.path.join(ROOT_FOLDER, INPUT_FOLDER, BASE_IMAGE_FILE),
        resize_pct=config["RESIZE_PCT"],
        threshold=config["THRESHOLD"],
        method=config["METHOD"],
        points_sampled=config["POINTS_SAMPLED"],
        colour_sampled=config["COLOUR_SAMPLED"],
        smoothing=config["SMOOTHING"],
    )



    #cv2.imshow('Preview', test_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv_filename = f'{BASE_IMAGE}_{timestamp}.jpg'
    cv_filepath = os.path.join(cv_output_path, cv_filename)
    cv2.imwrite(cv_filepath, test_image)

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
    plt.subplots_adjust(bottom=0.25)  # Add space below for the text

    plt_filename = f'{BASE_IMAGE}_{timestamp}.png'
    plt_filepath = os.path.join(plt_output_path, plt_filename)
    plt.savefig(plt_filepath, dpi=300, bbox_inches='tight')
    plt.clf()