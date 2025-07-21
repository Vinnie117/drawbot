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
from algos import greedy_path, greedy_path_numba, fast_greedy_path, greedy_path_numba_pb, clustered_greedy_tsp

# Load the image
image = cv2.imread('test.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
def apply_fixed_threshold(gray_image, threshold=127):
    # 127 is the midpoint of the 8-bit grayscale range (0 to 255),
    _, bw = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return bw

def apply_otsu_threshold(gray_image):
    # Otsu’s method (automatic thresholding):
    _, bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

bw_fixed = apply_fixed_threshold(gray, 127)
bw_otsu = apply_otsu_threshold(gray)

def create_image(image, scale_percent):
    # Resize the image to make it smaller (e.g., 20% of original size)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


img_bw_fixed = create_image(bw_fixed, 20)
img_bw_otsu = create_image(bw_otsu, 20)


side_by_side = np.hstack((img_bw_fixed, img_bw_otsu))
#cv2.imshow('Fixed vs Otsu Threshold', side_by_side)
cv2.imwrite('test_bw_fixed.jpg', img_bw_fixed)
cv2.imwrite('test_bw_otsu.jpg', img_bw_otsu)


################################################################
# Make Contours
binary  = apply_fixed_threshold(gray, 127)
binary  = 255 - binary   # black = draw

fixed_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Combine all contour points into one list
points = []

for cnt in fixed_contours:
    for pt in cnt:
        points.append(pt[0])  # pt

#for cnt in fixed_contours:
#    epsilon = 3  # try 2–5 for simplification
#    approx = cv2.approxPolyDP(cnt, epsilon, True)
#    for pt in approx:
#        points.append(pt[0])


################################################################
# Connect Contours into One Stroke: sort points to create a rough "single-stroke" path
# greedy approach with traveling salesman 
points = np.array(points)
num_points = len(points)


print(f"Computing {num_points} points")
start = time.time()
with ProgressBar(total=(num_points - 1) * num_points) as progress:  # outer loop * inner loop
    stroke_path = greedy_path_numba_pb(points, num_points, progress)
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
cv2.imshow("Single Stroke Path", resized_canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
