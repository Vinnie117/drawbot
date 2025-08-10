import cv2
import numpy as np
import math
from scipy.spatial import KDTree
import time
from numba import njit
from numba_progress import ProgressBar
from tqdm import tqdm
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tqdm import tqdm
import numpy as np
from algos.traveling_salesman import greedy_path, greedy_path_numba, fast_greedy_path, approach_greedy
from algos.traveling_salesman import clustered_greedy_tsp, compute_penalized_distance_matrix, greedy_path_from_matrix
import matplotlib.pyplot as plt
from utils.image_helper import apply_fixed_threshold, apply_otsu_threshold, create_image

# Load the image
image = cv2.imread('images/hokusai_wave.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


################################################################
# Make Contours
binary  = apply_fixed_threshold(gray, 127)
binary  = 255 - binary   # black = draw

## Combine all contour points into one list
#fixed_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#points = []
#for cnt in fixed_contours:
#    for pt in cnt:
#        points.append(pt[0])  # pt

## For quick prototyping
#for cnt in fixed_contours:
#    epsilon = 3  # try 2–5 for simplification (higher epsilon -> less points)
#    approx = cv2.approxPolyDP(cnt, epsilon, True)
#    for pt in approx:
#        points.append(pt[0])


# Randomly sample a subset (e.g., 10,000 points)
#fill_pixels = np.column_stack(np.where(binary == 1))
#sample_count = 50000
#if len(fill_pixels) > sample_count:
#    indices = np.random.choice(len(fill_pixels), sample_count, replace=False)
#    fill_pixels = fill_pixels[indices]

# Sample black pixels (!=0) or white (==0)
y_coords, x_coords = np.where(binary != 0)
black_pixels = np.column_stack((x_coords, y_coords))
sample_count = 100000  # adjust based on performance
if len(black_pixels) > sample_count:
    indices = np.random.choice(len(black_pixels), sample_count, replace=False)
    black_pixels = black_pixels[indices]

points = black_pixels


################################################################
# Connect Contours into One Stroke: sort points to create a rough "single-stroke" path
points = np.array(points)
#points = np.vstack((points, fill_pixels)) # goal here is to use edge points and filled points
num_points = len(points)
print(f"Computing {num_points} points")

################################################################
##approach: greedy approach with traveling salesman 
start = time.time()
with ProgressBar(total=(num_points - 1) * num_points) as progress:  # outer loop * inner loop
    stroke_path = approach_greedy(points, num_points, progress)
end = time.time()
print(f"✅ {num_points} points computed in {end - start:.2f} seconds.")
stroke_coords = [tuple(map(int, points[i])) for i in stroke_path]


################################################################
# Approach: Clusters cntours, then local TSPs
#stroke_coords = clustered_greedy_tsp(points, n_clusters=100)

################################################################
# Draw the stroke path on a blank canvas

canvas = np.ones_like(gray) * 255  # white background
for i in range(1, len(stroke_coords)):
    pt1 = stroke_coords[i - 1]
    pt2 = stroke_coords[i]
    cv2.line(canvas, pt1, pt2, 0, 1)

# Resize the canvas (e.g., to 20% size)
scale_percent = 20
width = int(canvas.shape[1] * scale_percent / 100)
height = int(canvas.shape[0] * scale_percent / 100)
resized_canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)

# Show the smaller canvas
#cv2.imshow("Single Stroke Path", resized_canvas)
#cv2.imwrite('million_points_200k.jpg', resized_canvas)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#plt.figure(figsize=(8, 6))
plt.imshow(resized_canvas, cmap='gray')
plt.axis('off')
plt.title('Single Stroke Path')

# Get current axes
ax = plt.gca()

# Add left-aligned text relative to the image (axes)
annotation = "Bla\n" \
             "Bli\n" \
             "Blubb"

ax.text(0.0, -0.1, annotation,
        transform=ax.transAxes,
        ha='left', va='top', fontsize=10)

plt.subplots_adjust(bottom=0.25)  # Add space below for the text
plt.show()

