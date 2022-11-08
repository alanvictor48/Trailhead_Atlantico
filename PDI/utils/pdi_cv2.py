import cv2
import mahotas
import numpy as np

def bgr2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def bgr2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def bgr2lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def equalize_hist(hist):
    return cv2.equalizeHist(hist)

def equalize_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def mean_filter(image, kernel):
    return cv2.blur(image, (kernel, kernel))

def median_filter(image, kernel):
    return cv2.medianBlur(image, kernel)

def bilateral_filter(image, color, space):
    return cv2.bilateralFilter(image, color, space, space)

def bin(image, low):
    return cv2.threshold(image, low, 255, cv2.THRESH_BINARY)[1]

def adaptative_bin(image, kernel, filter):
    if filter=='mean':
        return cv2.adaptiveThreshold(image, 255,  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, kernel)
    else:
        return cv2.adaptiveThreshold(image, 255,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, kernel)

def otsu_bin(image):
    T = mahotas.thresholding.otsu(image)
    return image > T

def sobel(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    return cv2.bitwise_or(sobelX, sobelY)

def laplacian(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return np.uint8(np.absolute(lap))

def canny(image, low, high):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, low, high)

def countour_detection(image, kernel):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    suave = cv2.blur(img, (kernel, kernel))
    
    T = mahotas.thresholding.otsu(suave)
    bin = suave.copy()
    bin = bin > T
    
    bordas = cv2.Canny(bin, 70, 150)
    
    (lx, objetos, lx) = cv2.findContours(bordas.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img = image.copy()
    cv2.drawContours(img, objetos, -1, (255, 0, 0), 2)

    return img