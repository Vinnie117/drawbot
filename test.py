import matplotlib.pyplot as plt
from main import create_drawing
import cv2

ROOT_FOLDER = 'images/'
INPUT_FOLDER = 'input/'
OUTPUT_FOLDER = 'output/'
BASE_IMAGE = 'hokusai_wave.jpg'
RESIZE_PCT = 30
THRESHOLD = 127
METHOD = 'fill'
POINTS_SAMPLED = 150000
COLOUR_SAMPLED = 'black'

test_image = create_drawing(img_path = ROOT_FOLDER + INPUT_FOLDER + BASE_IMAGE,
                      resize_pct=RESIZE_PCT,
                      threshold=THRESHOLD,
                      method=METHOD,
                      points_sampled=POINTS_SAMPLED,
                      colour_sampled=COLOUR_SAMPLED)


#cv2.imshow('Preview', test_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite(ROOT_FOLDER + OUTPUT_FOLDER + 'CV_' + BASE_IMAGE, test_image)


plt.imshow(test_image, cmap='gray')
plt.axis('off')
plt.title(BASE_IMAGE)

# Get current axes
ax = plt.gca()

# Add left-aligned text relative to the image (axes)
annotation = f"RESIZE_PCT: {RESIZE_PCT}\n" \
             f"THRESHOLD: {THRESHOLD}\n" \
             f"POINTS_SAMPLED: {POINTS_SAMPLED}\n" \
             f"METHOD: {METHOD}\n" \
             f"COLOUR_SAMPLED: {COLOUR_SAMPLED}"

ax.text(0.0, -0.1, annotation,
        transform=ax.transAxes,
        ha='left', va='top', fontsize=10)

plt.subplots_adjust(bottom=0.25)  # Add space below for the text
plt.savefig(ROOT_FOLDER + OUTPUT_FOLDER + 'PLT_' + BASE_IMAGE, dpi=300, bbox_inches='tight')
