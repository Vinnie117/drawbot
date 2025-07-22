import cv2
import numpy as np
from utils.image_helper import apply_otsu_threshold, apply_fixed_threshold, create_image

image = cv2.imread('images/hokusai_wave.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


bw_fixed = apply_fixed_threshold(gray, 127)
bw_otsu = apply_otsu_threshold(gray)
img_bw_fixed = create_image(bw_fixed, 20)
img_bw_otsu = create_image(bw_otsu, 20)


side_by_side = np.hstack((img_bw_fixed, img_bw_otsu))
#cv2.imshow('Fixed vs Otsu Threshold', side_by_side)
cv2.imwrite('images/test_bw_fixed.jpg', img_bw_fixed)
cv2.imwrite('images/test_bw_otsu.jpg', img_bw_otsu)
