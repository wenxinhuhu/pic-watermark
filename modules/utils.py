import cv2

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def save_image(image, image_path):
    cv2.imwrite(image_path, image)