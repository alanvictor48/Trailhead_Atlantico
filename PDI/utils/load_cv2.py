import cv2

def load_image(file: str):
    return cv2.imread('imgs/'+file)

def save_image(img, name: str):
    cv2.imwrite(name, img)

def show_image(image, title: str = 'image'):
    cv2.imshow(title, image)