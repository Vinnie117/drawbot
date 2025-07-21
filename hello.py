import cv2
import numpy as np
import math
from scipy.spatial import KDTree
import time
from numba import njit
from numba_progress import ProgressBar

# Load the image
image = cv2.imread('test.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding

#_, bw_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#_, bw_otu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

#cv2.waitKey(0)
#cv2.destroyAllWindows()


################################################################
# Make Contours
binary  = apply_fixed_threshold(gray, 127)
binary  = 255 - binary   # black = draw

fixed_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Combine all contour points into one list
points = []

#for cnt in fixed_contours:
#    for pt in cnt:
#        points.append(pt[0])  # pt

for cnt in fixed_contours:
    epsilon = 3  # try 2–5 for simplification
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    for pt in approx:
        points.append(pt[0])


################################################################
# Connect Contours into One Stroke: sort points to create a rough "single-stroke" path
# greedy or traveling salesman possible

def distance(p1, p2):
    # Euclidean distance in 2D
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

@njit
def distance_numba(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def greedy_path(points):
    path = [points[0]]
    used = [False] * len(points)
    used[0] = True
    for idx in range(1, len(points)):
        last = path[-1]
        next_index = min((i for i in range(len(points)) if not used[i]), key=lambda i: distance(last, points[i]))
        path.append(points[next_index])
        used[next_index] = True
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(points)}")
    return path

def fast_greedy_path(points):
    points = np.array(points)
    tree = KDTree(points)
    used = np.zeros(len(points), dtype=bool)
    path = [0]
    used[0] = True
    for _ in range(1, len(points)):
        dist, idx = tree.query(points[path[-1]], k=len(points))
        for i in idx:
            if not used[i]:
                path.append(i)
                used[i] = True
                break
        if _ % 100 == 0:
            print(f"Progress: {_}/{len(points)}")
    return points[path]


@njit(nogil=True)
def greedy_path_numba(points, progress_proxy):
    path = [0]
    used = np.zeros(len(points), dtype=np.bool_)
    used[0] = True
    for _ in range(1, len(points)):
        last = path[-1]
        min_dist = 1e9
        next_index = -1
        for i in range(len(points)):
            if not used[i]:
                d = distance_numba(points[last], points[i])
                if d < min_dist:
                    min_dist = d
                    next_index = i

            progress_proxy.update(1)

        path.append(next_index)
        used[next_index] = True
    return path

points = np.array(points)

start = time.time()
with ProgressBar(total=len(points) - 1) as progress:
    stroke_path = greedy_path_numba(points, progress)
end = time.time()
print(f"✅ Done in {end - start:.2f} seconds.")


################################################################
# Draw the stroke path on a blank canvas
stroke_coords = [tuple(map(int, points[i])) for i in stroke_path]
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
