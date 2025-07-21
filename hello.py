import cv2
import numpy as np

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
cv2.imshow('Fixed vs Otsu Threshold', side_by_side)
cv2.imwrite('test_bw_fixed.jpg', img_bw_fixed)
cv2.imwrite('test_bw_otsu.jpg', img_bw_otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()
