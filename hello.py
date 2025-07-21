import cv2

# Load the image
image = cv2.imread('test.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, bw_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Resize the image to make it smaller (e.g., 50% of original size)
scale_percent = 20  # percent of original size
width = int(bw_image.shape[1] * scale_percent / 100)
height = int(bw_image.shape[0] * scale_percent / 100)
dim = (width, height)
bw_image_resized = cv2.resize(bw_image, dim, interpolation=cv2.INTER_AREA)

# Save or display the result
cv2.imwrite('black_and_white.jpg', bw_image)
cv2.imshow('Black and White', bw_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
