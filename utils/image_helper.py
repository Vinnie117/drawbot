import cv2

def apply_fixed_threshold(gray_image, threshold=127):
    # 127 is the midpoint of the 8-bit grayscale range (0 to 255),
    _, bw = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return bw

def apply_otsu_threshold(gray_image):
    # Otsu’s method (automatic thresholding):
    _, bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def create_image(image, scale_percent):
    # Resize the image to make it smaller (e.g., 20% of original size)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)