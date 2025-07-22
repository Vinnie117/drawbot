import cv2
import numpy as np
from utils.image_helper import apply_otsu_threshold, apply_fixed_threshold, create_image

FOLDER = 'images/'
BASE_IMAGE = 'vermeers_earring.jpg'

image = cv2.imread(FOLDER + BASE_IMAGE)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


bw_fixed = apply_fixed_threshold(gray, 150)
bw_otsu = apply_otsu_threshold(gray)
img_bw_fixed = create_image(bw_fixed, 50)
img_bw_otsu = create_image(bw_otsu, 50)


side_by_side = np.hstack((img_bw_fixed, img_bw_otsu))
cv2.imshow('BW Preview', side_by_side)
#cv2.imwrite('images/test_bw_fixed.jpg', img_bw_fixed)
#cv2.imwrite('images/test_bw_otsu.jpg', img_bw_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()